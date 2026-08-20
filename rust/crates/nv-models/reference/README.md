# Reference notes: vLLM gemma-4

The inference-side spec we port into `src/gemma4_e4b.rs` (the gemma-4 E4B /
12B Per-Layer-Embedding text model) and the compressed-tensors / w4a16 loader
lives upstream in
[vllm-project/vllm](https://github.com/vllm-project/vllm)
`vllm/model_executor/models/gemma4{,_mm}.py` (Apache-2.0). Verbatim copies
were vendored here until 2026-08-06; fetch fresh from upstream when porting
against a new vLLM gemma-4 revision.

Architecture notes captured from these files + HF transformers `modeling_gemma4`:

- **Per-Layer Embeddings (PLE):** `embed_tokens_per_layer` -> reshape
  `[T, num_layers, hidden_per_layer=256]`; `per_layer_model_projection(embeds) *
  hidden_size**-0.5` -> `per_layer_projection_norm` (RMSNorm over 256); combine
  `(proj + ple) * 2**-0.5`. Per layer: `gate=gelu_tanh(input_gate(h)); h += post_per_layer_input_norm(projection(gate * per_layer_input))`.
- **KV sharing:** last `num_kv_shared_layers` reuse the K/V of the last
  non-shared layer of the same `layer_type`; shared layers have no k/v proj/norm.
- **Hybrid attention:** sliding layers use `head_dim`; full layers use
  `global_head_dim`; per-type RoPE (`default` vs `proportional` partial 0.25);
  `v_norm` is a weightless RMSNorm; attention `scaling = 1.0`.
- **Sandwich norms** + `final_logit_softcapping = 30.0` + `layer_scalar` per layer.
- E4B / 12B are dense (`enable_moe_block=false`); 26B-A4B adds the MoE block.
