# Ep 12: How LLM tool calling works (Qwen3-8B on an A100)

- `measure_tool_calling.py` runs all 5 GPU benchmarks and writes `tool_calling_results.json`.
- `make_plots.py` draws the 5 publication figures from `tool_calling_results.json`.

Run: `python measure_tool_calling.py` on a CUDA GPU with `torch` and `transformers` installed (we used an NVIDIA A100-SXM4-40GB, PyTorch 2.11).

## What the results cover
1. Schema injection (`<tools>`), the two-call tool handshake (`151657` `<tool_call>` to `151658` `</tool_call>`), server-side in-engine interception, and 5-step prefix KV cache reuse (`m1_two_call_handshake_and_prefix_cache`)
2. 36-layer Logit Lens phase transition of token `151657` (`0.08%` at Layer 32 -> `8.05%` at Layer 33 -> `97.07%` at Layer 34 -> `100.0%` at Layer 35) and schema attention routing (`m2_token_151657_logit_lens_and_attention`)
3. Sequential 5-call single-tool loop (`2,135` prefill tokens, `16.04 s`) vs native 2-call parallel tool calling (`791` prefill tokens, `11.76 s`, `2.70x` token reduction) (`m3_sequential_vs_parallel_tool_calls`)
4. Tool catalog scaling from `1` tool (`213` tokens, `69.85 ms`, `29.95 MiB` KV) to `30` tools (`5,176` tokens, `1,304.51 ms`, `727.88 MiB` KV) and `50` tools (`8,624` tokens, `1,212.75 MiB` / `1.18 GiB` KV, eager-attention CUDA OOM on 40 GB A100 with `792.24 ms` SDPA fallback), plus disjoint vs overlapping schema accuracy (`8/8` vs `2/8`) (`m4_tool_bloat_scaling_and_confusion`)
5. Constrained FSM enum logit masking at decode step 18 (`0/20` unconstrained vs `20/20` masked) and next-turn `<tool_response>` error recovery (`0/8` generic vs `7/8` actionable) (`m5_constrained_tools_and_self_correction`)

## Caveats and Methodological Notes
- **Simulated Python Backend Stubs (`0.005 ms`):** To isolate GPU prefill, decode, and KV cache latency from external network variance, the PostgreSQL, Redis, Kubernetes, and PagerDuty tools in `measure_tool_calling.py` are deterministic in-memory Python stubs returning fixed JSON payloads (`0.005 ms` execution).
- **Sequential 5-Call Harness (`m3`):** Even when prompted to call one tool per turn, `Qwen/Qwen3-8B`'s parallel tool prior emitted all 4 `<tool_call>` blocks (`92` tokens) in Call 1. Our sequential benchmark harness (`parsed[0]`) executes only the first tool call per turn across Calls 1 to 4 to measure the cost of a single-tool-per-turn loop vs dispatching all 4 in Call 1.
- **50-Tool Eager OOM vs SDPA (`m4`):** The `40 GB` CUDA OOM at `50` tools (`8,624` tokens) occurs under eager attention (`attn_implementation="eager"`), which materializes the `[1, 32, 8624, 8624]` score matrix (`~8.9 GB` per layer). With PyTorch SDPA / FlashAttention (`sdpa_fallback_after_eager_oom`), `50` tools completes prefill in `792.24 ms` while still consuming `1,212.75 MiB` (`1.18 GiB`) of KV cache per active request.
