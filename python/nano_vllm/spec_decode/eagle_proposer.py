from __future__ import annotations

from typing import Any

import torch

from nano_vllm.engine.sequence import Sequence
from nano_vllm.spec_decode.base import Proposer
from nano_vllm.spec_decode.eagle3 import Eagle3DraftModel
from nano_vllm.utils.pinned_scratch import host_view

class EagleProposer(Proposer):

    def __init__(self, draft_model: Eagle3DraftModel, num_drafts: int):
        if num_drafts < 1:
            raise ValueError(f"num_drafts must be >= 1, got {num_drafts}")
        self.draft_model = draft_model
        self.num_drafts = num_drafts

    @torch.inference_mode()
    def propose_tokens(
        self,
        last_tokens: torch.Tensor,
        aux_per_seq: torch.Tensor,
        position_offsets: torch.Tensor | None = None,
    ) -> list[list[int]]:
        if last_tokens.dim() != 1:
            raise ValueError(f"last_tokens must be 1-D [batch], got shape {tuple(last_tokens.shape)}")
        if aux_per_seq.dim() != 2:
            raise ValueError(
                f"aux_per_seq must be 2-D [batch, num_aux*hidden], got shape {tuple(aux_per_seq.shape)}"
            )
        if last_tokens.size(0) != aux_per_seq.size(0):
            raise ValueError(
                f"batch mismatch: last_tokens={last_tokens.size(0)} aux_per_seq={aux_per_seq.size(0)}"
            )
        batch = last_tokens.size(0)
        if batch == 0:
            return []
        device = next(self.draft_model.parameters()).device
        param_dtype = next(self.draft_model.parameters()).dtype
        last_tokens = last_tokens.to(device=device, dtype=torch.long)
        aux_per_seq = aux_per_seq.to(device=device, dtype=param_dtype)
        if position_offsets is None:
            position_offsets = torch.zeros(batch, dtype=torch.long, device=device)
        else:
            position_offsets = position_offsets.to(device=device, dtype=torch.long)

        chain_tokens: list[torch.Tensor] = []
        current_tokens = last_tokens
        chain_input = aux_per_seq
        fuse = True

        for step in range(self.num_drafts):
            positions_step = (position_offsets + step).unsqueeze(1)
            logits, midlayer_out = self.draft_model(
                current_tokens.unsqueeze(1),
                chain_input.unsqueeze(1),
                positions_step,
                fuse_aux=fuse,
            )
            next_tokens = self.draft_model.propose_token_ids(logits[:, -1, :])
            chain_tokens.append(next_tokens)
            current_tokens = next_tokens
            chain_input = midlayer_out[:, -1, :]
            fuse = False

        stacked = torch.stack(chain_tokens, dim=1)
        return [list(per_seq_chain) for per_seq_chain in stacked.tolist()]

    def propose(
        self,
        seqs: list[Sequence],
        runner_state: dict[str, Any],
    ) -> dict[int, list[int]]:
        active_indices = [i for i, seq in enumerate(seqs) if not seq.is_finished]
        if not active_indices:
            return {}
        aux = runner_state.get("last_aux_hidden_states")
        if aux is None:
            for i in active_indices:
                seqs[i].clear_drafts()
            return {}
        if aux.dim() == 3:
            aux = aux[:, -1, :]
        if aux.dim() != 2:
            raise AssertionError(
                f"EAGLE-3 aux tensor must be rank 2 or 3, got rank {aux.dim()} shape {tuple(aux.shape)}"
            )
        if aux.size(0) != len(seqs):
            raise AssertionError(
                f"EAGLE-3 aux tensor row count {aux.size(0)} does not match input seq count "
                f"{len(seqs)}; expected one row per seq."
            )
        device = aux.device
        active = [seqs[i] for i in active_indices]
        n = len(active)
        idx_h = host_view("eagle_idx", torch.long, n)
        last_h = host_view("eagle_last", torch.long, n)
        pos_h = host_view("eagle_pos", torch.long, n)
        idx_np = idx_h.numpy()
        last_np = last_h.numpy()
        pos_np = pos_h.numpy()
        for i, (orig_idx, seq) in enumerate(zip(active_indices, active)):
            idx_np[i] = orig_idx
            last_np[i] = seq.last_token
            pos_np[i] = seq.num_tokens
        non_blocking = device.type == "cuda"
        idx_t = idx_h.to(device, non_blocking=non_blocking)
        aux_per_seq = aux.index_select(0, idx_t)
        last_tokens = last_h.to(device, non_blocking=non_blocking)
        position_offsets = pos_h.to(device, non_blocking=non_blocking)
        drafts_per_seq = self.propose_tokens(last_tokens, aux_per_seq, position_offsets)
        return {seq.seq_id: drafts for seq, drafts in zip(active, drafts_per_seq)}
