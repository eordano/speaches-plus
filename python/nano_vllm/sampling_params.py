from dataclasses import dataclass
from typing import Any

TOP_K_DISABLED = -1
TOP_P_DISABLED = 1.0
TOP_P_LOWER_BOUND = 0.0
MIN_TEMPERATURE = 1e-10

@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    top_k: int = TOP_K_DISABLED
    top_p: float = TOP_P_DISABLED
    guided_json: dict[str, Any] | str | None = None
    guided_grammar: str | None = None
    guided_choice: list[str] | None = None
    guided_regex: str | None = None
    enable_ngram_spec_decode: bool = False
    ngram_max_n: int = 8
    ngram_min_n: int = 2
    ngram_num_drafts: int = 5

    def __post_init__(self):
        assert self.temperature > MIN_TEMPERATURE, "greedy sampling is not permitted"
        assert self.top_k == TOP_K_DISABLED or self.top_k > 0, "top_k must be -1 or positive"
        assert TOP_P_LOWER_BOUND < self.top_p <= TOP_P_DISABLED, "top_p must be in (0, 1]"
        guided_count = sum(
            spec is not None
            for spec in (self.guided_json, self.guided_grammar, self.guided_choice, self.guided_regex)
        )
        assert guided_count <= 1, "at most one of guided_json/grammar/choice/regex may be set"
        if self.enable_ngram_spec_decode:
            assert self.ngram_min_n >= 1, "ngram_min_n must be >= 1"
            assert self.ngram_max_n >= self.ngram_min_n, "ngram_max_n must be >= ngram_min_n"
            assert self.ngram_num_drafts >= 1, "ngram_num_drafts must be >= 1"

    @property
    def has_guided_decoding(self) -> bool:
        return any(
            spec is not None
            for spec in (self.guided_json, self.guided_grammar, self.guided_choice, self.guided_regex)
        )
