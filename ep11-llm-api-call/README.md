# Ep 11: What happens in 1 LLM API call (Qwen3-8B on an A100)

- `measure_api_call.py` runs all measurements and writes `api_call_results.json`.
- `m1_prefix_recheck.py` re-runs the 10-turn prefix-cache test correctly (see the correction below) and writes `m1_prefix_recheck.json`.
- `make_plots.py` draws the article figures from those two JSON files (Figure 1 uses the re-check).

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

## Correction (4 Nov 2026): prefix-cache test in `m1_*`
The first version of the 10-turn prefix-cache test (`measure_api_call.py`, `delta_ids = ids[:, prev_len:]`) assumed each turn's prompt is a prefix of the next one. With Qwen3's chat template it is not: every prompt ends with an empty `<think>\n\n</think>\n\n` block that the template drops once that turn becomes history, so the two prompts differ 4 tokens before the reply. The cache therefore held 4 stale tokens per turn and 4 reply tokens were never fed.

`m1_prefix_recheck.py` fixes this: it reuses only the longest common prefix of the previous and current prompt, times each turn 5 times (median), and checks that cached and full re-prefill pick the same next token (they do on all 10 turns; max last-logit difference 1.19 in bfloat16).

Corrected numbers: 5,549 vs 999 prefill tokens over 10 turns (5.55x); Turn 10 reuses 886 of 963 tokens and its time to first token drops from 138.64 ms to 64.18 ms (2.16x); summed over 10 turns 868.34 ms vs 642.30 ms (1.35x). The old `m1_*.ten_turn_benchmark` values in `api_call_results.json` (890 reused, 137.92 -> 66.29 ms) are kept for the record but superseded.

Also: Call 1 in the stateless demo is 55 tokens (with the system message); `m1_prefix_recheck.json` stores 55 with a note, because the script first ran without the system message.
