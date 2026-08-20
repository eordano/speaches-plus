import atexit
from dataclasses import fields
from time import perf_counter

import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nano_vllm.config import Config
from nano_vllm.engine.model_runner import ModelRunner
from nano_vllm.engine.scheduler import (
    SCHEDULE_MODE_PREFILL,
    Scheduler,
)
from nano_vllm.engine.sequence import Sequence
from nano_vllm.layers.grammar import GrammarBackend, accept_tokens
from nano_vllm.sampling_params import SamplingParams
from nano_vllm.spec_decode.base import Proposer
from nano_vllm.spec_decode.eagle3 import load_eagle3_from_hf
from nano_vllm.spec_decode.eagle_proposer import EagleProposer
from nano_vllm.spec_decode.ngram import NgramProposer

class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        assert tokenizer is not None, f"AutoTokenizer.from_pretrained returned None for {config.model!r}"
        assert tokenizer.eos_token_id is not None, f"tokenizer for {config.model!r} has no eos_token_id"
        self.tokenizer = tokenizer
        config.eos = tokenizer.eos_token_id
        assert config.hf_config is not None
        vocab_size: int = config.hf_config.vocab_size
        self.scheduler = Scheduler(config)
        self.grammar_backend = GrammarBackend(tokenizer, vocab_size)
        self.config = config
        self.matchers: dict[int, object] = {}
        self.proposer: Proposer | None = self._build_proposer(config)
        atexit.register(self.exit)

    @staticmethod
    def _build_proposer(config: Config) -> Proposer | None:
        if config.enable_eagle3_spec_decode:
            assert config.eagle3_speculator_path is not None
            assert config.hf_config is not None
            draft_model = load_eagle3_from_hf(config.eagle3_speculator_path)
            target_dtype: torch.dtype = config.hf_config.dtype
            draft_model = draft_model.to(device="cuda", dtype=target_dtype).eval()
            return EagleProposer(draft_model, config.eagle3_num_drafts)
        if config.enable_ngram_spec_decode:
            return NgramProposer(engine_default_enabled=True)
        return NgramProposer(engine_default_enabled=False)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        assert isinstance(prompt, list), f"tokenizer.encode returned {type(prompt).__name__}, expected list[int]"
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        matcher = self.grammar_backend.compile(sampling_params)
        if matcher is not None:
            if self.config.tensor_parallel_size > 1:
                raise NotImplementedError(
                    "grammar-guided decoding is not supported with "
                    "tensor_parallel_size>1: xgrammar matchers are not picklable "
                    "and cannot be shipped to subprocess ranks."
                )
            self.matchers[seq.seq_id] = matcher

    def step(self):
        seqs, mode = self.scheduler.schedule()
        if not seqs:
            return [], 0
        if mode == SCHEDULE_MODE_PREFILL:
            num_tokens = sum(seq.num_scheduled_tokens for seq in seqs)
        else:
            num_tokens = -len(seqs)
        step_matchers = (
            [self.matchers.get(seq.seq_id) for seq in seqs] if self.matchers else None
        )
        outputs_raw, runner_state = self.model_runner.call("run", seqs, mode, step_matchers)
        accepted = self.scheduler.postprocess(seqs, outputs_raw, mode)
        if accepted and self.matchers:
            advance_matchers = [self.matchers.get(seq_id) for seq_id, _ in accepted]
            advance_token_ids = [token_id for _, token_id in accepted]
            accept_tokens(advance_matchers, advance_token_ids)
        if self.proposer is not None:
            drafts = self.proposer.propose(seqs, runner_state)
            for seq in seqs:
                if seq.seq_id in drafts:
                    had = bool(seq.draft_tokens)
                    seq.set_drafts(drafts[seq.seq_id])
                    has = bool(seq.draft_tokens)
                    self.scheduler.note_drafts_set(had, has)
        outputs = []
        for seq in seqs:
            if seq.is_finished:
                outputs.append((seq.seq_id, seq.completion_token_ids))
                self.matchers.pop(seq.seq_id, None)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict[str, str | list[int]]]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        ordered_token_ids: list[list[int]] = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        results: list[dict[str, str | list[int]]] = []
        for tids in ordered_token_ids:
            decoded = self.tokenizer.decode(tids)
            assert isinstance(decoded, str), f"tokenizer.decode returned {type(decoded).__name__}"
            results.append({"text": decoded, "token_ids": tids})
        return results
