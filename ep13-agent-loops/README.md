# Ep 13: How AI agent loops work (Qwen3-8B on an A100)

Code and raw results for How AI Agents Actually Work, Episode 3: https://youtu.be/a31Ci7qqTuM

By Arun Prabhakaran (@LatentAILabs).

## Files
- `measure_agent_loops.py` runs the original tests and writes `agent_loops_results.json`. The video uses its M1 (history tax) and M2 (ReAct vs act-only) results.
- `measure_rerun_m3_m5.py` is the 12 Nov 2026 rerun of the plan-and-execute and empty-result tests. It writes `agent_loops_rerun_v2.json`, and with the `m5v3` / `m5v4` arguments it produces the data in `agent_loops_rerun_m5v3.json` and `agent_loops_rerun_m5v4.json`.
- The four JSON files are the raw results behind every number below.

## Setup
- Model: `Qwen/Qwen3-8B`, bfloat16, SDPA attention, greedy decoding (`gpu_environment` in each JSON).
- GPU: one NVIDIA A100-SXM4-40GB. PyTorch 2.14.1+cu130, transformers 5.22.1 (the transformers version is recorded in the rerun files only).
- Tools: a real on-disk SQLite database, created fresh by the script at `/tmp/incident_telemetry_ep13.db`, with 5 tables (`alerts`, `service_dependencies`, `db_metrics`, `deployments`, `error_logs`) and 17 seeded rows. Both scripts use the same database code.

## What the results cover

### 1. The while-loop history tax
Script: `measure_agent_loops.py`. JSON: `agent_loops_results.json`, key `m1_agent_while_loop_and_history_tax` (shortened to `m1` below).

- The agent ran 6 turns: 5 tool calls and 1 final answer. The 5 SQLite queries took 1.9975 ms in total (sum of `m1.canonical_trajectory[*].sqlite_exec_ms`: 0.5376 + 0.3462 + 0.3962 + 0.3552 + 0.3623).
- The 6 LLM turns took 42,729 ms of GPU time with stripped think blocks and a prefix KV cache (`m1.history_and_kv_modes.mode_c_strip_prior_think_prefix_kv.total_gpu_ms` = 42729.01).
- Re-sending the whole conversation every turn, with old think blocks kept, prefilled 8,794 prompt tokens over 6 steps (`mode_a_keep_full_think_stateless.cumulative_prefill_tokens_computed`). Stripping old think blocks and reusing the prefix KV cache prefilled 1,655 tokens (`mode_c_strip_prior_think_prefix_kv.cumulative_prefill_tokens_computed`). That is 5.31x fewer, calculated, not measured (`m1.comparisons.prefill_tokens_full_think_vs_prefix_kv_ratio`).
- Prefill time fell from 820 ms to 522 ms (`mode_a...cumulative_prefill_ms` = 820.38, `mode_c...cumulative_prefill_ms` = 521.55), 1.57x, calculated (`m1.comparisons.prefill_ms_speedup_full_think_vs_prefix_kv`). Total GPU time only went from 42.82 s to 42.73 s (`mode_a...total_gpu_ms` = 42822.34, `mode_c...total_gpu_ms` = 42729.01). Almost all of it is decoding (`cumulative_decode_ms` = 42001.96 and 42207.46).
- The step 6 KV cache size, 332.3 MiB with old think blocks and 263.7 MiB without (`mode_a...final_step_kv_mib` = 332.3, `mode_c...final_step_kv_mib` = 263.67), is calculated from the token count times 147,456 bytes per token (`gpu_environment.kv_bytes_per_token`). It is not a memory measurement.
- The system prompt listed the 5 tools in the order to call them, so this run measures the cost of the loop, not the model's planning.

### 2. ReAct vs act-only (8 red-herring incident tasks)
Script: `measure_agent_loops.py`. JSON: `agent_loops_results.json`, key `m2_react_vs_act_only_and_logit_lens.red_herring_benchmark`.

- Thinking off: 3/8 correct, 1.75 s per task on average (`act_only_no_think.correct_count` = 3, `mean_latency_ms` = 1751.78).
- Thinking on, 200-token reply limit: 0/8. All 8 were cut off inside the think block (`react_unbounded_200_tok_cap.correct_count` = 0, `truncated_in_think_count` = 8).
- 135-token think budget, 280-token reply limit: 5/8 (`react_bounded_think.correct_count` = 5). When the budget is reached the script inserts `</think>` so the model has to act.
- The two reply limits differ (200 vs 280), so the second and third rows are not a like-for-like comparison. The 120 / 200 / 280 reply limits and the 135 think budget are set in the script (`measure_m2_react_vs_act_only_and_logit_lens`), not stored in the JSON.

### 3. 5 calls vs 2 calls: ReAct vs Plan-and-Execute (median of 3 runs)
Script: `measure_rerun_m3_m5.py` (no argument). JSON: `agent_loops_rerun_v2.json`, key `m3v2_react_vs_parsed_plan_and_execute` (shortened to `m3v2` below).

- ReAct: 5 LLM calls, 5,174 prompt tokens, 710 output tokens, 37.8 s (`m3v2.sequential_react.llm_calls`, `cumulative_prefill_tokens`, `cumulative_output_tokens`; time is `m3v2.median_total_gpu_ms.sequential_react` = 37825.26). ReAct used a 75-token think budget, set in the script.
- Plan-and-Execute: the planner sees the same tool schemas (721 tokens in, 172 out, 9.1 s: `m3v2.plan_and_execute.planner_call`). Python parses its JSON plan (`planner_raw`) and runs it exactly as written: 4 real SQLite queries, plus 1 step naming a tool that does not exist (`executed_steps[4]`, `summarize_root_cause_and_rollback`, `valid_tool` = false). The solver reads all results (805 in, 209 out, 11.0 s: `solver_call`). Total 1,526 prompt tokens, 381 output tokens, 20.1 s (`cumulative_prefill_tokens`, `cumulative_output_tokens`, `m3v2.median_total_gpu_ms.plan_and_execute` = 20125.37).
- 3.39x fewer prompt tokens and 1.88x faster, both calculated, not measured (`m3v2.ratios.prefill_token_reduction_x`, `median_gpu_time_speedup_x`).
- Both answers named the release, the error code and the rollback command (`final_answer_checks` in both arms: `release`, `error_code`, `rollback` all true).
- Times are the median of 3 repeats (`m3v2.all_repeat_total_gpu_ms`). Token counts and per-call times are from repeat 1; token counts were identical in all 3 repeats (`m3v2.token_counts_identical_across_repeats`). Both arms re-prefill the full prompt every call, with no prefix KV reuse, and each call is capped at 512 new tokens. No call hit the cap.

### 4. Empty results: agents quit, they don't loop
Scripts: `measure_rerun_m3_m5.py` (no argument for the first bullet, `m5v4` for the rest). JSON: `agent_loops_rerun_v2.json` and `agent_loops_rerun_m5v4.json`.

- Asked to start on a date with 0 rows, with no "keep trying" line in the prompt, the agent made 1 call and answered that no error codes were found, at repetition penalty 1.00 and 1.35 (`agent_loops_rerun_v2.json`, `m4b_empty_date_trap_neutral_prompt.runs[*]`: one `query_error_logs` call with `row_count` 0, then `FINAL_ANSWER`; `duplicate_calls` = 0).
- An earlier prompt that said "keep trying" produced the same call 3 times (`agent_loops_results.json`, `m4_infinite_loop_traps_and_repetition_penalty.repetition_penalty_sweep[*].trace`). In that earlier run the tool returned a fixed empty payload instead of real SQLite rows.
- 8 trap tasks, up to 6 turns each (`SCENARIOS` in the script: 3 empty-date traps, 4 wrong-target traps, 1 empty-service-logs trap). Without checks: 0/8 (`m5v4_neutral_prompt_continue.mode_0_no_guardrail.solved_count` = 0, and the same for `m5v4_persistence_prompt_continue`).
- With a generic note added to the tool response when a call returns 0 rows ("this call returned no matching rows ... change the tool, target/service or date_prefix", `GUARD_EMPTY` in the script): 3/8 with a neutral system prompt, all 3 date traps, L1 to L3 (`m5v4_neutral_prompt_continue.mode_1_action_hash_guardrail.solved_count` = 3). 1/8 with a "keep using tools" system prompt that we wrote after seeing the first rerun (`m5v4_persistence_prompt_continue.mode_1_action_hash_guardrail.solved_count` = 1). Both system prompts are stored under `system_prompt`.
- A SHA-256 duplicate-call check never fired: 0 repeated calls in 32 runs (2 prompts x 2 modes x 8 tasks; `total_duplicate_calls` = 0 and `total_blocked_calls` = 0 in all four blocks of `agent_loops_rerun_m5v4.json`).
- Wrong-target traps (L4, L5, L6, L8): 0/4 in every configuration. One of them, L4, answered with a `kubectl rollout undo` rollback command that is not in the database (`cases[3].final_answer`).
- Each case ran once, greedy, thinking off, with a 300-token cap per call. Treat these as results on our setup, not general rates.

## How to rerun
You need a CUDA GPU (we used an A100-SXM4-40GB; the bf16 weights alone are 16.38 GB, `gpu_environment.model_weights_gb_decimal`), `torch` and `transformers` installed, and access to `Qwen/Qwen3-8B` on Hugging Face.

```bash
# Sections 1 and 2. Creates projects/38_agent_loops/data/agent_loops_results.json under the current directory.
python measure_agent_loops.py

# Section 3 and the first bullet of section 4. Writes data/agent_loops_rerun_v2.json under the current directory.
python measure_rerun_m3_m5.py

# Section 4 trap tasks. Prints the result instead of writing a file (see below).
python measure_rerun_m3_m5.py m5v4
```

The `m5v4` and `m5v3` modes print the JSON as a zlib-compressed, base64-encoded block between the lines `BEGIN_RERUN_B64` and `END_RERUN_B64`. Save the block to a file and decode it with:

```bash
python -c "import sys,zlib,base64,json; print(json.dumps(json.loads(zlib.decompress(base64.b64decode(open(sys.argv[1]).read()))), indent=2))" blob.txt > agent_loops_rerun_m5v4.json
```

Greedy decoding should give the same token counts on the same software versions. Timings vary between runs: the three ReAct repeats in section 3 took 39.6 s, 37.8 s and 37.7 s.

## Limitations
- Timings are single wall-clock runs, except section 3, which is the median of 3 repeats.
- The 8 red-herring tasks (section 2) and the 8 trap tasks (section 4) were each run once. With 8 tasks, one task changes a result by 12.5 points.
- Ratios (5.31x, 1.57x, 3.39x, 1.88x) are calculated from two measured values. KV cache sizes in MiB are calculated from token counts, not measured.
- Section 1 runs the agent once to record a 6-turn trajectory, then replays that same trajectory under each history mode: each mode prefills its own prompt and then feeds the recorded output tokens one at a time. Decode time is therefore almost the same in every mode by design, and the 42,729 ms is the replay time with stripped think blocks and the prefix KV cache.
- The tasks, the database and the system prompts were written by us. The section 2 tasks are built so that the headline service is a red herring.
- `agent_loops_rerun_m5v3.json` and the `m5v2_generic_action_hash_guardrail` block in `agent_loops_rerun_v2.json` are intermediate reruns kept for completeness. It stopped a case as soon as the model replied without a tool call, which ended many runs while the model was saying it would try another date. `m5v4` keeps going in that case (up to 6 calls) and is the version used in the video.
