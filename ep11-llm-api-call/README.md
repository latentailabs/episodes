# Ep 11: What happens in 1 LLM API call (Qwen3-8B on an A100)

- `measure_api_call.py` runs all measurements and writes `api_call_results.json`.
- `make_plots.py` draws the article figures from that JSON.

Run: `python measure_api_call.py` on a CUDA GPU with `torch` and `transformers` installed (we used an A100-SXM4-40GB, PyTorch 2.11).

## What the results cover
1. Stateless calls vs history, and 10-turn re-prefill vs prefix KV cache (`m1_*`)
2. Chat template tokens, raw / wrong-template / ChatML accuracy on 5 prompts, attention from the last prompt token to control-token positions (`m2_*`)
3. Prefill vs decode timing for 64 to 2,048 prompt tokens (`m3_*`)
4. JSON extraction: naive prompt, strict system prompt, FSM logit mask, 80 calls each (`m4_m5_*`)

## Caveats
- Decode was measured with Hugging Face transformers, eager attention, batch 1. Effective bandwidth was about 240 GB/s, far below the A100's 1,555 GB/s spec, so optimized servers decode faster.
- The strict system prompt also reached 80/80 valid JSON on this simple schema. The logit mask is the guarantee, not the only thing that worked.
- The attention share includes position 0 (`<|im_start|>`), a known attention sink. It is a head mean over one 38-token prompt.
