# Ep 14: How AI agent memory works (Qwen3-8B on an L4)

Code and raw results for How AI Agents Actually Work, Episode 4.

By Arun Prabhakaran (@LatentAILabs).

## Files
- `measure_agent_memory.py` runs all four tests and writes `agent_memory_results.json`.
- `agent_memory_results.json` holds the raw results behind every number below.

## Setup
- Models: `Qwen/Qwen3-8B` (bfloat16, greedy decoding) and `Qwen/Qwen3-Embedding-0.6B`.
- GPU: one NVIDIA L4 (22.03 GiB), PyTorch 2.14.1+cu130 (`m0_hardware_and_preflight`).
- Memory store: an on-disk SQLite database created by the script.

## What the results cover

### 1. Sliding window vs append-only prefix cache (one 12-turn chat)
Key: `m1_sliding_window_vs_prefix_cache`.
- Suffix prefill tokens in turns 5–12 (the tokens the model had to read again):
  - append-only: 971
  - sliding window (keep last 4 turns): 4,126
  - 4.25x more for the sliding window (`comparisons.sliding_window_vs_append_suffix_prefill_ratio_turns_5_to_12`)
- Mean cache hit in turns 5–12: 88.9% vs 29.2%.
- Measured prefill time in turns 5–12 (sum of `steps[].prefill_ms` for turns 5–12): 766 ms vs 1,870 ms, 2.44x, calculated.
- Total GPU time over all 12 turns: 28.48 s vs 29.63 s (`total_gpu_ms`). Decoding dominates, so the wall-clock difference is small.
- Both runs answered turn 12 correctly (`turn12_solved_all_3_facts`).
- Threshold compaction once after turn 6:
  - 2,489 suffix prefill tokens in total, including the compaction prefill
  - 137.11 MiB peak KV cache, against 226.27 MiB for append-only

### 2. Compaction at an equal 256-token budget (8 ten-turn incidents)
Key: `m2_compaction_prose_vs_schema`.
- Full history: 7/8.
- Keep last 3 turns: 0/8 (wrong port and a bad restart on all 8).
- Prose summary: 0/8. The port was kept 8/8, but the agent restarted the healthy component on all 8.
- 4-field state schema: 4/8.

### 3. Knowledge updates across 64 sessions (8 services)
Key: `m3_knowledge_updates_and_lineage`.
- Top-1 cosine search ranked the outdated setup note first on 6/8 services and answered 2/8 current-state queries.
- A SQLite fact table with a `supersedes_fact_id` column answered 8/8 current and 8/8 historical queries.

### 4. Pre-turn hook vs agent-called search tool
Key: `m4_control_policy_hook_vs_tool`.
- One hop: hook 4/4 in 1.32 s, tool loop 4/4 in 1.72 s (1.31x, `single_hop_4_tasks.latency_speedup_hook_vs_tool`).
- Two hops: hook alone 0/4, hook plus tool 4/4.
