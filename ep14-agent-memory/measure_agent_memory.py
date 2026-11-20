#!/usr/bin/env python3
"""Episode 14 GPU Telemetry Benchmark: How AI Agent Memory Works (`projects/39_agent_memory`).

Runs 4 reproducible, non-rigged GPU + SQLite + Dense-Embedding benchmarks on `Qwen/Qwen3-8B` (bfloat16)
and `Qwen/Qwen3-Embedding-0.6B`:
  - M0: Hardware, Model & LCP `DynamicCache` Correctness Preflight
  - M1: 12-Turn Agent Session — Append-Only Prefix Cache vs Rolling Sliding Window (W=4) vs Rolling Top-Summary vs Threshold Compaction
  - M2: Equal-Budget (256-tok) Compaction Quality Across 8 10-Turn Debugging Trajectories (Full vs Window vs Prose vs 4-Field Schema)
  - M3: Cross-Session Knowledge Updates & Historical Queries Across 64 SQLite + Vector Memories (Top-1 vs Top-5 vs Qwen3-Extracted Fact Lineage)
  - M4: Memory Control Policy Across 4 Single-Hop + 4 Two-Hop Tasks (Pre-Turn Hook vs Agent-Directed Memory Tools vs Hybrid)
"""
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

_t0_hb = time.time()
_stop_hb = threading.Event()


def _hb():
    while not _stop_hb.wait(10.0):
        print(f"[hb] Ep14 GPU benchmark running... {int(time.time() - _t0_hb)}s", flush=True)


threading.Thread(target=_hb, daemon=True).start()

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
assert DEV == "cuda", "measure_agent_memory.py requires a CUDA GPU"

LLM_ID = "Qwen/Qwen3-8B"
EMBED_CANDIDATES = ["Qwen/Qwen3-Embedding-0.6B", "BAAI/bge-base-en-v1.5"]
DB_PATH = "/tmp/agent_memory_ep14.db"
OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "agent_memory_results.json"


def sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def render_chat_ids(tokenizer, messages, enable_thinking=False, add_generation_prompt=True):
    txt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    # Ensure prior turns keep a deterministic serialization so prefix matching is purely a function of message list mutations
    return tokenizer.encode(txt, add_special_tokens=False)


def lcp_length(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


import copy


def clone_dynamic_cache(cache: DynamicCache, crop_len: int) -> DynamicCache:
    if cache is None or crop_len <= 0 or cache.get_seq_length() == 0:
        return DynamicCache()
    new_c = copy.deepcopy(cache)
    if hasattr(new_c, "layers"):
        for layer in new_c.layers:
            if layer.keys is not None and layer.keys.shape[-2] > crop_len:
                layer.keys = layer.keys[..., :crop_len, :].contiguous()
                layer.values = layer.values[..., :crop_len, :].contiguous()
    elif hasattr(new_c, "key_cache"):
        for i in range(len(new_c.key_cache)):
            if new_c.key_cache[i].shape[-2] > crop_len:
                new_c.key_cache[i] = new_c.key_cache[i][..., :crop_len, :].contiguous()
                new_c.value_cache[i] = new_c.value_cache[i][..., :crop_len, :].contiguous()
    if hasattr(new_c, "_seen_tokens"):
        new_c._seen_tokens = min( getattr(new_c, "_seen_tokens", crop_len), crop_len )
    return new_c


def lcp_prefill_and_decode(model, tokenizer, prev_ids: list[int], prev_cache: DynamicCache | None, curr_ids: list[int], max_new_tokens: int = 64):
    """Explicit Longest-Common-Prefix (LCP) KV cache reuse + timed prefill and greedy decode."""
    lcp = lcp_length(prev_ids, curr_ids) if prev_cache is not None else 0
    # Need at least 1 new token to run forward() and produce logits for curr_ids[-1]
    if lcp >= len(curr_ids):
        lcp = max(0, len(curr_ids) - 1)

    cache = clone_dynamic_cache(prev_cache, lcp) if (prev_cache is not None and lcp > 0) else DynamicCache()
    suffix_ids = curr_ids[lcp:]
    inp = torch.tensor([suffix_ids], dtype=torch.long, device=DEV)

    t0 = sync_time()
    with torch.inference_mode():
        out = model(input_ids=inp, past_key_values=cache, use_cache=True)
    t_prefill_ms = (sync_time() - t0) * 1000.0

    next_tok = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
    gen_ids = [next_tok]
    eos_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}

    t1 = sync_time()
    with torch.inference_mode():
        for _ in range(max_new_tokens - 1):
            if next_tok in eos_ids:
                break
            step_inp = torch.tensor([[next_tok]], dtype=torch.long, device=DEV)
            out = model(input_ids=step_inp, past_key_values=cache, use_cache=True)
            next_tok = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
            gen_ids.append(next_tok)
    t_decode_ms = (sync_time() - t1) * 1000.0

    full_seq_ids = curr_ids + gen_ids
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return {
        "prompt_tokens": len(curr_ids),
        "reused_prefix_tokens": lcp,
        "suffix_prefill_tokens": len(suffix_ids),
        "prefill_ms": round(t_prefill_ms, 3),
        "decode_tokens": len(gen_ids),
        "decode_ms": round(t_decode_ms, 3),
        "total_ms": round(t_prefill_ms + t_decode_ms, 3),
        "gen_text": gen_text,
        "full_seq_ids": full_seq_ids,
        "cache": cache,
    }


def main():
    print(f"[M0] Loading {LLM_ID} on {torch.cuda.get_device_name(0)}...", flush=True)
    tok = AutoTokenizer.from_pretrained(LLM_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_ID, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    ).eval()

    embed_id = None
    embed_tok = None
    embed_model = None
    for cand in EMBED_CANDIDATES:
        try:
            embed_tok = AutoTokenizer.from_pretrained(cand)
            embed_model = AutoModel.from_pretrained(cand, torch_dtype=torch.bfloat16, device_map="cuda").eval()
            embed_id = cand
            print(f"[M0] Loaded embedding model: {embed_id}", flush=True)
            break
        except Exception as e:
            print(f"[M0] Could not load {cand}: {e}", flush=True)
    assert embed_model is not None, "Failed to load any embedding model"

    def encode_texts(texts: list[str]) -> np.ndarray:
        batch = embed_tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEV)
        with torch.inference_mode():
            out = embed_model(**batch)
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            normed = F.normalize(pooled, p=2, dim=-1)
        return normed.cpu().numpy()

    # Warmup GPU
    _ = lcp_prefill_and_decode(model, tok, [], None, tok.encode("Warmup GPU pass."), max_new_tokens=4)

    # M0: Hardware + Model + LCP Preflight Check
    cfg = model.config
    n_params = sum(p.numel() for p in model.parameters())
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    kv_bytes_per_tok = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads) * 2

    pf_msgs = [
        {"role": "system", "content": "You are a senior site reliability agent. Follow CLAUDE.md rules strictly."},
        {"role": "user", "content": "Check the current PgBouncer port for inventory-db."},
    ]
    pf_ids = render_chat_ids(tok, pf_msgs, enable_thinking=False)
    split_p = len(pf_ids) - 6
    with torch.inference_mode():
        full_logits = model(torch.tensor([pf_ids], device=DEV)).logits[:, -1, :].float()
        c_pf = DynamicCache()
        _ = model(torch.tensor([pf_ids[:split_p]], device=DEV), past_key_values=c_pf, use_cache=True)
        cached_logits = model(torch.tensor([pf_ids[split_p:]], device=DEV), past_key_values=c_pf, use_cache=True).logits[:, -1, :].float()
    max_logit_diff = float(torch.max(torch.abs(full_logits - cached_logits)).item())
    cos_sim = float(F.cosine_similarity(full_logits, cached_logits, dim=-1).item())
    top1_match = bool(torch.argmax(full_logits, dim=-1).item() == torch.argmax(cached_logits, dim=-1).item())
    assert top1_match and max_logit_diff <= 1.0 and cos_sim > 0.999, (
        f"Preflight failed: max_logit_diff={max_logit_diff}, cos_sim={cos_sim}, top1_match={top1_match}"
    )

    m0_info = {
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_total_gib": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
        "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
        "pytorch_version": torch.__version__,
        "llm_id": LLM_ID,
        "embedding_id": embed_id,
        "num_parameters": n_params,
        "weights_gib": round(weight_bytes / (1024**3), 2),
        "weights_gb": round(weight_bytes / 1e9, 2),
        "num_layers": cfg.num_hidden_layers,
        "num_query_heads": cfg.num_attention_heads,
        "num_kv_heads": cfg.num_key_value_heads,
        "head_dim": getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads),
        "kv_bytes_per_token": kv_bytes_per_tok,
        "kv_kib_per_token": round(kv_bytes_per_tok / 1024.0, 1),
        "preflight_lcp_cache_check": {
            "total_tokens": len(pf_ids),
            "prefix_tokens": split_p,
            "suffix_tokens": len(pf_ids) - split_p,
            "max_abs_logit_diff_bf16_ulp": round(max_logit_diff, 5),
            "logit_cosine_similarity": round(cos_sim, 6),
            "top1_token_match": top1_match,
        },
    }
    print(f"[M0] Preflight OK: max_logit_diff={max_logit_diff:.5f}, top1_match={top1_match}", flush=True)

    # =========================================================================
    # M1: 12-Turn Agent Session — Prefix Cache vs Sliding Window vs Rolling Top-Summary vs Threshold Compaction
    # =========================================================================
    print("[M1] Running 12-turn KV cache & compaction benchmark...", flush=True)
    CLAUDE_MD_SYSTEM = (
        "You are an autonomous production engineering agent.\n"
        "# CLAUDE.md Project Rules\n"
        "- Primary database cluster: `inventory-db` (PostgreSQL 16.4, region `us-east-1`).\n"
        "- Connection pooler: PgBouncer (`pgbouncer.ini`, default transaction pooling mode).\n"
        "- Cache cluster: `redis-session-cache` (Redis 7.2, port 6379, TLS enabled).\n"
        "- Deployment tool: `deploy_ctl --service <name> --port <int> --pool-size <int>`.\n"
        "- Never restart a healthy service whose p99 latency is under 2.0 ms.\n"
        "- Reply concisely with the exact next diagnostic or remediation step."
    )

    SESSION_TURNS = [
        (
            "Turn 1: Alert INC-8801 fired: checkout-api HTTP 502 rate spiked to 31.4% at 14:02 UTC. Inspect active alerts:\n"
            "[TOOL_OUTPUT] alert_id=INC-8801 service=checkout-api upstream_error='db_pool_exhausted' p99_ms=4120.0 status=FIRING. Summarize finding in 1 sentence.",
            "Alert INC-8801 confirms checkout-api is failing with db_pool_exhausted and 4,120 ms p99 latency.",
        ),
        (
            "Turn 2: Check the recent network migration log for inventory-db:\n"
            "[TOOL_OUTPUT] migration_id=MIG-204 timestamp='2026-11-15T13:55:00Z' service=inventory-db action='Moved PgBouncer listener from legacy port 5432 to port 6432 and set max_client_conn=400'. Summarize finding in 1 sentence.",
            "Migration MIG-204 moved inventory-db's PgBouncer listener from port 5432 to port 6432 with max_client_conn=400.",
        ),
        (
            "Turn 3: Check checkout-api's current environment configuration in Kubernetes:\n"
            "[TOOL_OUTPUT] deployment=checkout-api-v42 DB_HOST='inventory-db.internal' DB_PORT=5432 DB_POOL_SIZE=64 DB_USER='vault_iam_role' pod_restarts=14. Summarize finding in 1 sentence.",
            "The checkout-api-v42 deployment is still configured with legacy DB_PORT=5432 instead of the new PgBouncer port 6432.",
        ),
        (
            "Turn 4: Before changing checkout-api, verify whether redis-session-cache is also degraded:\n"
            "[TOOL_OUTPUT] service=redis-session-cache port=6379 connected_clients=182 p99_latency_ms=0.38 error_rate=0.00% status=HEALTHY_DO_NOT_RESTART. Summarize finding in 1 sentence.",
            "The redis-session-cache cluster has 0.38 ms p99 latency and 0% errors, so it is completely healthy and must not be restarted.",
        ),
        (
            "Turn 5: Inspect PostgreSQL pg_stat_activity on inventory-db:\n"
            "[TOOL_OUTPUT] active_backends=19 max_connections=200 wait_event='None' pgbouncer_port_6432_listeners=4 legacy_port_5432_listeners=0. Summarize finding in 1 sentence.",
            "PostgreSQL on inventory-db has only 19 active backends and zero listeners on legacy port 5432 while port 6432 has 4 active listeners.",
        ),
        (
            "Turn 6: Check recent error logs from checkout-api pods:\n"
            "[TOOL_OUTPUT] 2026-11-15T14:03:11Z ERROR [db_pool] dial tcp 10.24.8.19:5432: connect: connection refused (retries=3, timeout=4000ms) service=checkout-api. Summarize finding in 1 sentence.",
            "The checkout-api pods are getting connection refused on 10.24.8.19:5432 because nothing is listening on port 5432.",
        ),
        (
            "Turn 7: Check payment-worker dependency health to see if any other service uses port 5432:\n"
            "[TOOL_OUTPUT] service=payment-worker DB_HOST='inventory-db.internal' DB_PORT=6432 DB_POOL_SIZE=32 p99_latency_ms=4.2 status=HEALTHY. Summarize finding in 1 sentence.",
            "The payment-worker service is already using DB_PORT=6432 and running normally at 4.2 ms p99 latency.",
        ),
        (
            "Turn 8: Check TLS certificate validity on inventory-db PgBouncer port 6432:\n"
            "[TOOL_OUTPUT] cert_cn='inventory-db.internal' issuer='vault-pki-ca' valid_until='2027-02-15T00:00:00Z' tls_handshake_ms=1.1 status=VALID. Summarize finding in 1 sentence.",
            "The TLS certificate on inventory-db port 6432 is valid through February 2027 with 1.1 ms handshake latency.",
        ),
        (
            "Turn 9: Check connection pool sizing rules for checkout-api across 6 replicas:\n"
            "[TOOL_OUTPUT] replicas=6 current_pool_per_replica=64 total_requested=384 pgbouncer_max_client_conn=400 recommended_pool_per_replica=48. Summarize finding in 1 sentence.",
            "Across 6 replicas, setting pool_size=48 per replica keeps total connections at 288, safely below PgBouncer's 400 limit.",
        ),
        (
            "Turn 10: Verify Canary pod test with DB_PORT=6432 and DB_POOL_SIZE=48:\n"
            "[TOOL_OUTPUT] pod=checkout-api-canary-1 DB_PORT=6432 DB_POOL_SIZE=48 http_502_rate=0.00% p99_latency_ms=11.4 status=PASS. Summarize finding in 1 sentence.",
            "The checkout-api canary pod with DB_PORT=6432 and DB_POOL_SIZE=48 passed with 0% 502 errors and 11.4 ms p99 latency.",
        ),
        (
            "Turn 11: Check deployment freeze window and rollback tag for checkout-api:\n"
            "[TOOL_OUTPUT] current_release='rel-2026-42' next_release='rel-2026-43' freeze_window='NONE' approval_ticket='OPS-9102'. Summarize finding in 1 sentence.",
            "Release rel-2026-43 is approved under ticket OPS-9102 with no active deployment freeze.",
        ),
        (
            "Turn 12: Based on all findings in this session, state the exact deploy_ctl command for checkout-api (including --port and --pool-size) and state whether redis-session-cache should be restarted.",
            None,
        ),
    ]

    def run_m1_mode(mode_name: str, window_k: int | None = None, rolling_top_summary: bool = False, compact_at_turn: int | None = None):
        sys_msg = {"role": "system", "content": CLAUDE_MD_SYSTEM}
        history = []  # list of (user_msg, assistant_msg) pairs
        prev_ids = []
        prev_cache = None
        steps_log = []
        compaction_overhead = {"calls": 0, "prefill_tokens": 0, "decode_tokens": 0, "gpu_ms": 0.0}
        pinned_checkpoint_msg = None

        for t_idx, (u_text, canonical_reply) in enumerate(SESSION_TURNS, start=1):
            # If threshold compaction triggers right before turn `compact_at_turn` (e.g. after Turn 6 completes, before Turn 7)
            if compact_at_turn is not None and t_idx == compact_at_turn + 1:
                comp_prompt = [
                    sys_msg,
                    {
                        "role": "user",
                        "content": (
                            "Compact the completed debugging turns below into a concise structured checkpoint "
                            "(Active Ports/Configs, Verified Findings, Ruled-Out Healthy Services) in under 90 words:\n\n"
                            + "\n".join(f"U: {u}\nA: {a}" for u, a in history)
                        ),
                    },
                ]
                c_ids = render_chat_ids(tok, comp_prompt, enable_thinking=False)
                c_res = lcp_prefill_and_decode(model, tok, prev_ids, prev_cache, c_ids, max_new_tokens=120)
                compaction_overhead["calls"] += 1
                compaction_overhead["prefill_tokens"] += c_res["suffix_prefill_tokens"]
                compaction_overhead["decode_tokens"] += c_res["decode_tokens"]
                compaction_overhead["gpu_ms"] += c_res["total_ms"]
                pinned_checkpoint_msg = {
                    "role": "assistant",
                    "content": f"[COMPACTED CHECKPOINT TURNS 1-{compact_at_turn}]: {c_res['gen_text']}",
                }
                history = []  # older turns replaced by pinned_checkpoint_msg
                # Keep prev_ids / prev_cache from the system prefix so Turn 7 reuses the system prefix!

            # Construct current turn messages
            msgs = [sys_msg]
            if rolling_top_summary and len(history) > 0:
                # Rewrites a rolling summary right after sys_msg on every turn (mutating token prefix right after sys_msg)
                rolling_str = f"[ROLLING SUMMARY AFTER TURN {t_idx - 1} ({len(history)} turns completed)]: " + " | ".join(
                    f"T{i+1}: {a[:48]}" for i, (_, a) in enumerate(history)
                )
                msgs.append({"role": "assistant", "content": rolling_str})
                for u, a in history[-3:]:
                    msgs.append({"role": "user", "content": u})
                    msgs.append({"role": "assistant", "content": a})
            elif window_k is not None:
                active_hist = history[-window_k:]
                for u, a in active_hist:
                    msgs.append({"role": "user", "content": u})
                    msgs.append({"role": "assistant", "content": a})
            else:
                if pinned_checkpoint_msg is not None:
                    msgs.append(pinned_checkpoint_msg)
                for u, a in history:
                    msgs.append({"role": "user", "content": u})
                    msgs.append({"role": "assistant", "content": a})

            msgs.append({"role": "user", "content": u_text})
            curr_ids = render_chat_ids(tok, msgs, enable_thinking=False)
            res = lcp_prefill_and_decode(model, tok, prev_ids, prev_cache, curr_ids, max_new_tokens=48)

            reply_used = canonical_reply if canonical_reply is not None else res["gen_text"]
            # Update prev_ids and prev_cache to represent the state at the end of this turn
            msgs_with_reply = msgs + [{"role": "assistant", "content": reply_used}]
            end_ids = render_chat_ids(tok, msgs_with_reply, enable_thinking=False, add_generation_prompt=False)
            # Prefill the assistant reply tokens into cache so next turn can reuse the full turn
            lcp_end = lcp_length(curr_ids, end_ids)
            end_cache = clone_dynamic_cache(res["cache"], lcp_end)
            rem_ids = end_ids[lcp_end:]
            if len(rem_ids) > 0:
                with torch.inference_mode():
                    _ = model(torch.tensor([rem_ids], dtype=torch.long, device=DEV), past_key_values=end_cache, use_cache=True)
            prev_ids = end_ids
            prev_cache = end_cache
            history.append((u_text, reply_used))

            kv_mib = round((len(end_ids) * kv_bytes_per_tok) / (1024**2), 2)
            steps_log.append({
                "turn": t_idx,
                "prompt_tokens": res["prompt_tokens"],
                "reused_prefix_tokens": res["reused_prefix_tokens"],
                "suffix_prefill_tokens": res["suffix_prefill_tokens"],
                "cache_hit_pct": round(100.0 * res["reused_prefix_tokens"] / max(1, res["prompt_tokens"]), 1),
                "prefill_ms": res["prefill_ms"],
                "decode_tokens": res["decode_tokens"],
                "decode_ms": res["decode_ms"],
                "kv_cache_mib": kv_mib,
                "gen_text": res["gen_text"] if t_idx == 12 else reply_used,
            })

        tot_prompt = sum(s["prompt_tokens"] for s in steps_log)
        tot_reused = sum(s["reused_prefix_tokens"] for s in steps_log)
        tot_suffix_prefill = sum(s["suffix_prefill_tokens"] for s in steps_log) + compaction_overhead["prefill_tokens"]
        tot_prefill_ms = sum(s["prefill_ms"] for s in steps_log) + compaction_overhead["gpu_ms"]
        tot_decode_ms = sum(s["decode_ms"] for s in steps_log)
        post_window_suffix = sum(s["suffix_prefill_tokens"] for s in steps_log[4:])  # Turns 5..12
        final_ans = steps_log[-1]["gen_text"]
        solved_t12 = ("6432" in final_ans) and ("48" in final_ans) and any(
            k in final_ans.lower() for k in ("not restart", "do not", "no", "healthy")
        )
        return {
            "mode": mode_name,
            "cumulative_prompt_tokens": tot_prompt,
            "cumulative_reused_prefix_tokens": tot_reused,
            "cumulative_suffix_prefill_tokens": tot_suffix_prefill,
            "turns_5_to_12_suffix_prefill_tokens": post_window_suffix,
            "mean_cache_hit_pct_turns_5_to_12": round(float(np.mean([s["cache_hit_pct"] for s in steps_log[4:]])), 1),
            "cumulative_prefill_ms": round(tot_prefill_ms, 2),
            "cumulative_decode_ms": round(tot_decode_ms, 2),
            "total_gpu_ms": round(tot_prefill_ms + tot_decode_ms, 2),
            "peak_kv_cache_mib": max(s["kv_cache_mib"] for s in steps_log),
            "compaction_overhead": compaction_overhead,
            "turn12_answer": final_ans,
            "turn12_solved_all_3_facts": solved_t12,
            "steps": steps_log,
        }

    m1_append = run_m1_mode("mode_a_append_only_prefix_cache")
    m1_window = run_m1_mode("mode_b_sliding_window_last_4", window_k=4)
    m1_rolling = run_m1_mode("mode_c_rolling_top_summary", rolling_top_summary=True)
    m1_compact = run_m1_mode("mode_d_threshold_compact_turn6", compact_at_turn=6)

    m1_summary = {
        "system_prefix_tokens": len(render_chat_ids(tok, [{"role": "system", "content": CLAUDE_MD_SYSTEM}], enable_thinking=False, add_generation_prompt=False)),
        "mode_a_append_only": m1_append,
        "mode_b_sliding_window": m1_window,
        "mode_c_rolling_summary": m1_rolling,
        "mode_d_threshold_compact": m1_compact,
        "comparisons": {
            "sliding_window_vs_append_suffix_prefill_ratio_all_12_turns": round(
                m1_window["cumulative_suffix_prefill_tokens"] / max(1, m1_append["cumulative_suffix_prefill_tokens"]), 2
            ),
            "sliding_window_vs_append_suffix_prefill_ratio_turns_5_to_12": round(
                m1_window["turns_5_to_12_suffix_prefill_tokens"] / max(1, m1_append["turns_5_to_12_suffix_prefill_tokens"]), 2
            ),
            "rolling_summary_vs_append_suffix_prefill_ratio_all_12_turns": round(
                m1_rolling["cumulative_suffix_prefill_tokens"] / max(1, m1_append["cumulative_suffix_prefill_tokens"]), 2
            ),
            "sliding_window_vs_threshold_compact_suffix_prefill_ratio": round(
                m1_window["cumulative_suffix_prefill_tokens"] / max(1, m1_compact["cumulative_suffix_prefill_tokens"]), 2
            ),
            "threshold_compact_peak_kv_savings_pct_vs_append": round(
                100.0 * (1.0 - m1_compact["peak_kv_cache_mib"] / max(1e-6, m1_append["peak_kv_cache_mib"])), 1
            ),
        },
    }
    print(
        f"[M1] Done! Suffix Prefill Tokens (12 turns): Append={m1_append['cumulative_suffix_prefill_tokens']}, "
        f"Window={m1_window['cumulative_suffix_prefill_tokens']} "
        f"(Turns 5-12 ratio={m1_summary['comparisons']['sliding_window_vs_append_suffix_prefill_ratio_turns_5_to_12']}x), "
        f"RollingSum={m1_rolling['cumulative_suffix_prefill_tokens']}, "
        f"ThresholdCompact={m1_compact['cumulative_suffix_prefill_tokens']}",
        flush=True,
    )

    # =========================================================================
    # M2: Equal-Budget (256-Token) Compaction Benchmark Across 8 10-Turn Incidents (C1..C8)
    # =========================================================================
    print("[M2] Running 8-task equal-budget compaction benchmark (C1..C8)...", flush=True)
    M2_TASKS = [
        {
            "id": "C1",
            "service": "checkout-api",
            "param_name": "port",
            "new_val": "6432",
            "old_val": "5432",
            "turn2_fact": "Migration MIG-204 moved inventory-db PgBouncer listener from legacy port 5432 to port 6432.",
            "ruled_out_service": "redis-session-cache",
            "turn4_neg": "Diagnostic check on redis-session-cache: 0.38 ms p99 latency, 0% errors. Status: HEALTHY. Do NOT restart redis-session-cache.",
        },
        {
            "id": "C2",
            "service": "auth-gateway",
            "param_name": "port",
            "new_val": "8443",
            "old_val": "8080",
            "turn2_fact": "Security cutover SEC-119 moved token-signer mTLS listener from legacy port 8080 to port 8443.",
            "ruled_out_service": "ldap-directory-sync",
            "turn4_neg": "Diagnostic check on ldap-directory-sync: 0.72 ms p99 latency, 0% errors. Status: HEALTHY. Do NOT restart ldap-directory-sync.",
        },
        {
            "id": "C3",
            "service": "payment-worker",
            "param_name": "port",
            "new_val": "5671",
            "old_val": "5672",
            "turn2_fact": "Broker upgrade BRK-308 moved rabbitmq-ledger AMQP-TLS listener from port 5672 to port 5671.",
            "ruled_out_service": "fraud-scorer-rpc",
            "turn4_neg": "Diagnostic check on fraud-scorer-rpc: 1.15 ms p99 latency, 0% errors. Status: HEALTHY. Do NOT restart fraud-scorer-rpc.",
        },
        {
            "id": "C4",
            "service": "search-indexer",
            "param_name": "port",
            "new_val": "9250",
            "old_val": "9200",
            "turn2_fact": "Cluster re-shard IDX-412 moved opensearch-catalog ingest coordinator from port 9200 to port 9250.",
            "ruled_out_service": "kafka-clickstream",
            "turn4_neg": "Diagnostic check on kafka-clickstream: 0.90 ms p99 latency, 0 consumer lag. Status: HEALTHY. Do NOT restart kafka-clickstream.",
        },
        {
            "id": "C5",
            "service": "billing-cron",
            "param_name": "port",
            "new_val": "7432",
            "old_val": "5432",
            "turn2_fact": "Failover cutover DB-509 movedtimescale-ledger writer endpoint from port 5432 to port 7432.",
            "ruled_out_service": "invoice-pdf-worker",
            "turn4_neg": "Diagnostic check on invoice-pdf-worker: 0.55 ms queue latency, 0% errors. Status: HEALTHY. Do NOT restart invoice-pdf-worker.",
        },
        {
            "id": "C6",
            "service": "notification-hub",
            "param_name": "port",
            "new_val": "6380",
            "old_val": "6379",
            "turn2_fact": "TLS migration CACHE-601 moved valkey-pubsub cluster listener from plaintext port 6379 to TLS port 6380.",
            "ruled_out_service": "smtp-relay-proxy",
            "turn4_neg": "Diagnostic check on smtp-relay-proxy: 0.64 ms p99 latency, 0% bounce errors. Status: HEALTHY. Do NOT restart smtp-relay-proxy.",
        },
        {
            "id": "C7",
            "service": "analytics-etl",
            "param_name": "port",
            "new_val": "9009",
            "old_val": "8123",
            "turn2_fact": "Protocol upgrade CH-715 moved clickhouse-events native TCP listener from HTTP port 8123 to port 9009.",
            "ruled_out_service": "s3-parquet-sink",
            "turn4_neg": "Diagnostic check on s3-parquet-sink: 0.49 ms write ack, 0% errors. Status: HEALTHY. Do NOT restart s3-parquet-sink.",
        },
        {
            "id": "C8",
            "service": "Order-fulfillment",
            "param_name": "port",
            "new_val": "15432",
            "old_val": "5432",
            "turn2_fact": "Proxy cutover PRX-820 moved citus-shard-router listener from port 5432 to port 15432.",
            "ruled_out_service": "memcache-shipping-rates",
            "turn4_neg": "Diagnostic check on memcache-shipping-rates: 0.31 ms p99 latency, 0% errors. Status: HEALTHY. Do NOT restart memcache-shipping-rates.",
        },
    ]

    PROSE_COMPACT_PROMPT = (
        "Summarize the debugging conversation so far in a concise paragraph so the agent can continue "
        "the investigation and take the final remediation action in the next turn."
    )
    SCHEMA_COMPACT_PROMPT = (
        "Compress the debugging conversation so far into this 4-field structured checkpoint:\n"
        "1. ACTIVE_CONFIG_AND_PORTS: exact service names, new ports, and old ports.\n"
        "2. VERIFIED_ROOT_CAUSE: what caused the connection failures.\n"
        "3. RULED_OUT_HEALTHY_COMPONENTS: healthy services that were checked and must NOT be restarted.\n"
        "4. NEXT_ACTION: pending remediation step."
    )

    def build_10turn_history(task: dict) -> list[tuple[str, str]]:
        s = task["service"]
        r = task["ruled_out_service"]
        nv, ov = task["new_val"], task["old_val"]
        return [
            (f"Turn 1: Alert fired on {s}: elevated timeout rate. Suspects mentioned in alert paging channel: upstream port config or {r}.", f"Acknowledged alert on {s}; will inspect recent migrations and {r}."),
            (f"Turn 2: Fetch recent infrastructure change log:\n[TOOL_OUTPUT] {task['turn2_fact']}", f"Noted: listener moved from {ov} to {nv}."),
            (f"Turn 3: Inspect {s} deployment env vars:\n[TOOL_OUTPUT] service={s} UPSTREAM_PORT={ov} retries=3 status=CRASHLOOP.", f"{s} is still pointing to legacy port {ov}."),
            (f"Turn 4: Run health probe on suspect {r} before touching it:\n[TOOL_OUTPUT] {task['turn4_neg']}", f"{r} is healthy and ruled out."),
            (f"Turn 5: Collect stack trace sample from {s} pod-01:\n[TOOL_OUTPUT] " + " ".join([f"frame_{i}: net.dial.timeout({ov}) at worker.go:{100+i}" for i in range(18)]), "Stack trace confirms connection timeouts on the old port."),
            (f"Turn 6: Collect Prometheus histogram buckets for {s}:\n[TOOL_OUTPUT] " + " ".join([f"le_{b}ms={b*3}" for b in (10, 25, 50, 100, 250, 500, 1000, 2500, 5000)]), "Latency histogram shows all failed requests hitting the timeout ceiling."),
            (f"Turn 7: Check node kernel dmesg and conntrack table:\n[TOOL_OUTPUT] nf_conntrack_count=4120 nf_conntrack_max=262144oom_kills=0 nic_drops=0.", "Kernel conntrack and network interface metrics are clean."),
            (f"Turn 8: Check {r} connection pool metrics again because the initial pager alert mentioned {r}:\n[TOOL_OUTPUT] component={r} active_conns=42 idle_conns=22 cpu_pct=6.4.", f"Reviewed {r} pool metrics."),
            (f"Turn 9: Verify change window approval for {s}:\n[TOOL_OUTPUT] ticket=CHG-991 status=APPROVED.", f"Change ticket CHG-991 is approved for {s}."),
        ]

    m2_results = {"full_history": [], "sliding_window_3": [], "prose_compact_256": [], "schema_compact_256": []}

    for task in M2_TASKS:
        hist9 = build_10turn_history(task)
        final_q = (
            f"Turn 10: Output your final remediation JSON with keys "
            f'{{"service": "{task["service"]}", "target_port": <int>, "restart_secondary_component": "<name or NONE>"}}. '
            f"Only restart `{task['ruled_out_service']}` if it was NOT ruled out as healthy earlier."
        )
        sys_m2 = {"role": "system", "content": "You are a site reliability agent. Answer Turn 10 with the exact JSON requested."}

        for arm_key in ("full_history", "sliding_window_3", "prose_compact_256", "schema_compact_256"):
            summary_txt = ""
            port_in_sum = False
            neg_in_sum = False
            if arm_key == "full_history":
                msgs = [sys_m2]
                for u, a in hist9:
                    msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
                msgs.append({"role": "user", "content": final_q})
            elif arm_key == "sliding_window_3":
                msgs = [sys_m2]
                for u, a in hist9[-3:]:
                    msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
                msgs.append({"role": "user", "content": final_q})
            else:
                # Compress Turns 1..6 (which contain Turn 2 port fact, Turn 4 ruled-out fact, and Turns 5..6 logs) under equal 256-tok budget
                c_instr = PROSE_COMPACT_PROMPT if arm_key == "prose_compact_256" else SCHEMA_COMPACT_PROMPT
                c_msgs = [
                    sys_m2,
                    {
                        "role": "user",
                        "content": c_instr + "\n\n" + "\n".join(f"{u}\nAssistant: {a}" for u, a in hist9[:6]),
                    },
                ]
                c_ids = render_chat_ids(tok, c_msgs, enable_thinking=False)
                c_out = lcp_prefill_and_decode(model, tok, [], None, c_ids, max_new_tokens=256)
                summary_txt = c_out["gen_text"]
                port_in_sum = task["new_val"] in summary_txt
                neg_in_sum = (task["ruled_out_service"].lower() in summary_txt.lower()) and any(
                    w in summary_txt.lower() for w in ("healthy", "not restart", "ruled out", "do not", "0%")
                )
                msgs = [
                    sys_m2,
                    {"role": "assistant", "content": f"[COMPACTED HISTORY TURNS 1-6]:\n{summary_txt}"},
                ]
                for u, a in hist9[6:]:
                    msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
                msgs.append({"role": "user", "content": final_q})

            p_ids = render_chat_ids(tok, msgs, enable_thinking=False)
            out = lcp_prefill_and_decode(model, tok, [], None, p_ids, max_new_tokens=96)
            ans = out["gen_text"]
            correct_port = (task["new_val"] in ans) and (f'"{task["old_val"]}"' not in ans and f": {task['old_val']}" not in ans)
            avoided_bad_restart = ("none" in ans.lower()) or (task["ruled_out_service"].lower() not in ans.lower()) or ("not" in ans.lower())
            # Check JSON field specifically if present
            m_rest = re.search(r'"restart_secondary_component"\s*:\s*"([^"]+)"', ans)
            if m_rest:
                avoided_bad_restart = m_rest.group(1).strip().upper() == "NONE"

            passed = bool(correct_port and avoided_bad_restart)
            m2_results[arm_key].append({
                "task_id": task["id"],
                "prompt_tokens": len(p_ids),
                "summary_text": summary_txt,
                "port_retained_in_summary": port_in_sum,
                "neg_retained_in_summary": neg_in_sum,
                "turn10_output": ans,
                "correct_port": correct_port,
                "avoided_bad_restart": avoided_bad_restart,
                "passed": passed,
            })

    m2_summary = {}
    for arm_key, rows in m2_results.items():
        m2_summary[arm_key] = {
            "passed_count": sum(1 for r in rows if r["passed"]),
            "total_tasks": len(rows),
            "accuracy_pct": round(100.0 * sum(1 for r in rows if r["passed"]) / len(rows), 1),
            "correct_port_count": sum(1 for r in rows if r["correct_port"]),
            "avoided_bad_restart_count": sum(1 for r in rows if r["avoided_bad_restart"]),
            "port_retained_in_summary_count": sum(1 for r in rows if r["port_retained_in_summary"]),
            "neg_retained_in_summary_count": sum(1 for r in rows if r["neg_retained_in_summary"]),
            "mean_prompt_tokens": round(float(np.mean([r["prompt_tokens"] for r in rows])), 1),
            "tasks": rows,
        }
    print(
        f"[M2] Done! Full={m2_summary['full_history']['passed_count']}/8, "
        f"Window3={m2_summary['sliding_window_3']['passed_count']}/8, "
        f"Prose={m2_summary['prose_compact_256']['passed_count']}/8, "
        f"Schema={m2_summary['schema_compact_256']['passed_count']}/8",
        flush=True,
    )

    # =========================================================================
    # M3: Cross-Session Knowledge Updates & Historical Queries (64 SQLite Sessions + GPU Embeddings)
    # =========================================================================
    print("[M3] Building 64-session SQLite + GPU Embedding memory store and running U1..U8...", flush=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE session_memories (session_id INTEGER PRIMARY KEY, session_date TEXT, service TEXT, note_text TEXT)"
    )
    cur.execute(
        "CREATE TABLE fact_lineage (fact_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER, session_date TEXT, entity TEXT, attribute TEXT, value TEXT, supersedes_fact_id INTEGER)"
    )

    U_SPECS = [
        ("U1", "checkout-api", "inventory-db", "5432", "6432", "app_ro", "vault_iam_role"),
        ("U2", "auth-gateway", "user-profile-db", "5432", "6433", "auth_reader", "mtls_spiffe_id"),
        ("U3", "payment-worker", "ledger-postgres", "5432", "7432", "pay_legacy", "kms_token_role"),
        ("U4", "search-indexer", "catalog-opensearch", "9200", "9250", "basic_admin", "oidc_service_acct"),
        ("U5", "billing-cron", "timescale-metrics", "5432", "15432", "cron_user", "iam_rds_signer"),
        ("U6", "notification-hub", "valkey-pubsub", "6379", "6380", "default_acl", "tls_client_cert"),
        ("U7", "analytics-etl", "clickhouse-events", "8123", "9009", "etl_readonly", "kerberos_principal"),
        ("U8", "order-fulfillment", "citus-coordinator", "5432", "16432", "order_rw", "vault_dynamic_creds"),
    ]

    session_rows = []
    sid = 1
    # 1. Insert initial architecture setup notes (2026-10-04 .. 2026-10-08)
    for idx, (uid, svc, db, old_p, new_p, old_u, new_u) in enumerate(U_SPECS):
        dt = f"2026-10-0{min(9, 4 + (idx % 5))}"
        txt = (
            f"Architecture reference for {svc} and {db}: what port and database identity does {svc} use to connect to {db}? "
            f"{svc} connects to {db} on port {old_p} using database identity {old_u}."
        )
        session_rows.append((sid, dt, svc, txt))
        sid += 1

    # 2. Insert 20 distractor sessions (2026-10-10 .. 2026-10-16)
    for d_i in range(20):
        svc = U_SPECS[d_i % len(U_SPECS)][1]
        dt = f"2026-10-{10 + (d_i % 7):02d}"
        txt = f"Routine CI build and latency check #{d_i+100} on {svc}: container image scanned, 0 CVEs found, CPU utilization at {32 + (d_i % 15)}%."
        session_rows.append((sid, dt, svc, txt))
        sid += 1

    # 3. Insert port cutover changelog notes (2026-10-18 .. 2026-10-22)
    for idx, (uid, svc, db, old_p, new_p, old_u, new_u) in enumerate(U_SPECS):
        dt = f"2026-10-{18 + (idx % 5):02d}"
        txt = (
            f"INCIDENT CUTOVER CHANGELOG ({svc} -> {db}): deprecated legacy listener {old_p}; "
            f"cut over {svc} upstream pooler for {db} to port {new_p}."
        )
        session_rows.append((sid, dt, svc, txt))
        sid += 1

    # 4. Insert 20 more distractor sessions (2026-10-24 .. 2026-10-30)
    for d_i in range(20):
        svc = U_SPECS[d_i % len(U_SPECS)][1]
        dt = f"2026-10-{24 + (d_i % 7):02d}"
        txt = f"Weekly autoscaler review #{d_i+200} for {svc}: HPA min_replicas=4 max_replicas=16, memory request 2Gi."
        session_rows.append((sid, dt, svc, txt))
        sid += 1

    # 5. Insert credential rotation notes (2026-11-02 .. 2026-11-06)
    for idx, (uid, svc, db, old_p, new_p, old_u, new_u) in enumerate(U_SPECS):
        dt = f"2026-11-0{2 + (idx % 5)}"
        txt = (
            f"SECURITY ROTATION NOTICE ({svc} -> {db}): revoked static account {old_u}; "
            f"rotated {svc} database authentication identity for {db} to {new_u} (port remains {new_p})."
        )
        session_rows.append((sid, dt, svc, txt))
        sid += 1

    cur.executemany("INSERT INTO session_memories VALUES (?, ?, ?, ?)", session_rows)
    conn.commit()

    # Use Qwen3-8B to extract structured (entity, attribute, value) rows from the 24 non-distractor state-change sessions into `fact_lineage`
    for s_id, s_dt, s_svc, s_txt in session_rows:
        if "Routine CI build" in s_txt or "Weekly autoscaler" in s_txt:
            continue
        ext_msgs = [
            {"role": "system", "content": "Extract database connection facts as JSON: {\"service\": \"...\", \"port\": \"...\", \"identity\": \"...\"}. Use null if a field is not mentioned."},
            {"role": "user", "content": f"Session Date: {s_dt}\nNote: {s_txt}"},
        ]
        e_ids = render_chat_ids(tok, ext_msgs, enable_thinking=False)
        e_out = lcp_prefill_and_decode(model, tok, [], None, e_ids, max_new_tokens=64)["gen_text"]
        m_p = re.search(r'"port"\s*:\s*"?(\d+)"?', e_out)
        m_id = re.search(r'"identity"\s*:\s*"([^"]+)"', e_out)
        if m_p:
            pval = m_p.group(1)
            cur.execute("SELECT fact_id FROM fact_lineage WHERE entity=? AND attribute='port' ORDER BY fact_id DESC LIMIT 1", (s_svc,))
            prev_row = cur.fetchone()
            sup_id = prev_row[0] if prev_row else None
            cur.execute("INSERT INTO fact_lineage (session_id, session_date, entity, attribute, value, supersedes_fact_id) VALUES (?, ?, ?, 'port', ?, ?)", (s_id, s_dt, s_svc, pval, sup_id))
        if m_id and m_id.group(1).lower() != "null":
            ival = m_id.group(1)
            cur.execute("SELECT fact_id FROM fact_lineage WHERE entity=? AND attribute='identity' ORDER BY fact_id DESC LIMIT 1", (s_svc,))
            prev_row = cur.fetchone()
            sup_id = prev_row[0] if prev_row else None
            cur.execute("INSERT INTO fact_lineage (session_id, session_date, entity, attribute, value, supersedes_fact_id) VALUES (?, ?, ?, 'identity', ?, ?)", (s_id, s_dt, s_svc, ival, sup_id))
    conn.commit()

    # Embed all 64 session notes on GPU
    all_notes = [r[3] for r in session_rows]
    note_vecs = encode_texts(all_notes)

    m3_curr_results = {"top1_cosine": [], "top5_cosine": [], "fact_lineage": []}
    m3_hist_results = {"top1_cosine": [], "top5_cosine": [], "fact_lineage": []}

    for uid, svc, db, old_p, new_p, old_u, new_u in U_SPECS:
        q_curr = f"Current-State Query (Date: 2026-11-15): What port and database identity does {svc} use to connect to {db}?"
        q_hist = f"Historical Query: Back on 2026-10-12 (before the mid-October cutover), what port and database identity did {svc} use to connect to {db}?"

        qv_curr = encode_texts([q_curr])[0]
        sims_curr = note_vecs @ qv_curr
        top5_idx_curr = np.argsort(-sims_curr)[:5]

        qv_hist = encode_texts([q_hist])[0]
        sims_hist = note_vecs @ qv_hist
        top5_idx_hist = np.argsort(-sims_hist)[:5]

        for arm_name in ("top1_cosine", "top5_cosine", "fact_lineage"):
            # 1. Current-state evaluation
            if arm_name == "top1_cosine":
                i0 = int(top5_idx_curr[0])
                ctx_curr = f"[Session #{session_rows[i0][0]} | {session_rows[i0][1]} | cos={sims_curr[i0]:.3f}]: {session_rows[i0][3]}"
                top1_is_stale = old_p in session_rows[i0][3]
            elif arm_name == "top5_cosine":
                ctx_curr = "\n".join(
                    f"[Session #{session_rows[int(i)][0]} | {session_rows[int(i)][1]} | cos={sims_curr[int(i)]:.3f}]: {session_rows[int(i)][3]}"
                    for i in top5_idx_curr
                )
                top1_is_stale = old_p in session_rows[int(top5_idx_curr[0])][3]
            else:
                cur.execute(
                    """
                    SELECT attribute, value, session_date, session_id, supersedes_fact_id
                    FROM fact_lineage f1
                    WHERE entity=? AND NOT EXISTS (
                        SELECT 1 FROM fact_lineage f2 WHERE f2.supersedes_fact_id = f1.fact_id
                    )
                    """,
                    (svc,),
                )
                active_facts = cur.fetchall()
                ctx_curr = "ACTIVE NON-SUPERSEDED FACTS IN SQLITE:\n" + "\n".join(
                    f"- {svc}.{a} = {v} (since {dt}, session #{s_i}, supersedes_fact_id={sup})" for a, v, dt, s_i, sup in active_facts
                )
                top1_is_stale = False

            msgs_c = [
                {"role": "system", "content": "Answer with JSON: {\"port\": \"...\", \"identity\": \"...\"} based on the retrieved memory."},
                {"role": "user", "content": f"{ctx_curr}\n\n{q_curr}"},
            ]
            ans_c = lcp_prefill_and_decode(model, tok, [], None, render_chat_ids(tok, msgs_c, enable_thinking=False), max_new_tokens=48)["gen_text"]
            ok_c = bool(
                re.search(rf"\b{re.escape(new_p)}\b", ans_c)
                and (new_u in ans_c)
                and not re.search(rf"\b{re.escape(old_p)}\b", ans_c)
            )
            m3_curr_results[arm_name].append({
                "id": uid, "service": svc, "top1_is_stale": top1_is_stale,
                "context": ctx_curr, "output": ans_c, "passed": ok_c,
            })

            # 2. Historical-date evaluation (2026-10-12)
            if arm_name == "top1_cosine":
                i0 = int(top5_idx_hist[0])
                ctx_hist = f"[Session #{session_rows[i0][0]} | {session_rows[i0][1]}]: {session_rows[i0][3]}"
            elif arm_name == "top5_cosine":
                ctx_hist = "\n".join(
                    f"[Session #{session_rows[int(i)][0]} | {session_rows[int(i)][1]}]: {session_rows[int(i)][3]}"
                    for i in top5_idx_hist
                )
            else:
                cur.execute(
                    """
                    SELECT attribute, value, session_date, session_id
                    FROM fact_lineage f1
                    WHERE entity=? AND session_date <= '2026-10-12'
                    AND NOT EXISTS (
                        SELECT 1 FROM fact_lineage f2 WHERE f2.supersedes_fact_id = f1.fact_id AND f2.session_date <= '2026-10-12'
                    )
                    """,
                    (svc,),
                )
                hist_facts = cur.fetchall()
                ctx_hist = "POINT-IN-TIME FACTS AS OF 2026-10-12:\n" + "\n".join(
                    f"- {svc}.{a} = {v} (recorded {dt}, session #{s_i})" for a, v, dt, s_i in hist_facts
                )

            msgs_h = [
                {"role": "system", "content": "Answer with JSON: {\"port\": \"...\", \"identity\": \"...\"} for the requested date."},
                {"role": "user", "content": f"{ctx_hist}\n\n{q_hist}"},
            ]
            ans_h = lcp_prefill_and_decode(model, tok, [], None, render_chat_ids(tok, msgs_h, enable_thinking=False), max_new_tokens=48)["gen_text"]
            ok_h = bool(
                re.search(rf"\b{re.escape(old_p)}\b", ans_h)
                and (old_u in ans_h)
                and not re.search(rf"\b{re.escape(new_p)}\b", ans_h)
            )
            m3_hist_results[arm_name].append({
                "id": uid, "service": svc, "context": ctx_hist, "output": ans_h, "passed": ok_h,
            })

    m3_summary = {
        "embedding_model": embed_id,
        "total_sqlite_sessions": len(session_rows),
        "top1_ranked_stale_first_count": sum(1 for r in m3_curr_results["top1_cosine"] if r["top1_is_stale"]),
        "current_state_queries": {
            k: {
                "passed_count": sum(1 for r in v if r["passed"]),
                "total": len(v),
                "accuracy_pct": round(100.0 * sum(1 for r in v if r["passed"]) / len(v), 1),
                "rows": v,
            }
            for k, v in m3_curr_results.items()
        },
        "historical_date_queries": {
            k: {
                "passed_count": sum(1 for r in v if r["passed"]),
                "total": len(v),
                "accuracy_pct": round(100.0 * sum(1 for r in v if r["passed"]) / len(v), 1),
                "rows": v,
            }
            for k, v in m3_hist_results.items()
        },
    }
    print(
        f"[M3] Done! Stale Ranked #1={m3_summary['top1_ranked_stale_first_count']}/8 | "
        f"Current Acc: Top1={m3_summary['current_state_queries']['top1_cosine']['passed_count']}/8, "
        f"Top5={m3_summary['current_state_queries']['top5_cosine']['passed_count']}/8, "
        f"Lineage={m3_summary['current_state_queries']['fact_lineage']['passed_count']}/8 | "
        f"Historical Acc: Lineage={m3_summary['historical_date_queries']['fact_lineage']['passed_count']}/8",
        flush=True,
    )

    # =========================================================================
    # M4: Control Policy Comparison — Pre-Turn Retrieval Hook vs Agent-Directed Memory Tool Loop (4 1-Hop + 4 2-Hop)
    # =========================================================================
    print("[M4] Running Pre-Turn Retrieval Hook vs Agent-Directed Tool Calling (1-Hop & 2-Hop)...", flush=True)
    cur.execute("CREATE TABLE alert_routing (alert_code TEXT PRIMARY KEY, symptom TEXT, backing_service TEXT)")
    ALERT_MAP = [
        ("INC-901", "HTTP 502 spike on edge-checkout- ingress", "checkout-api", "6432", "vault_iam_role"),
        ("INC-902", "Token verification timeout on mobile-sso-edge", "auth-gateway", "6433", "mtls_spiffe_id"),
        ("INC-903", "Webhook settlement lag on stripe-ingest-edge", "payment-worker", "7432", "kms_token_role"),
        ("INC-904", "Autocomplete timeout on storefront-graphql-edge", "search-indexer", "9250", "oidc_service_acct"),
    ]
    for acode, symp, b_svc, _, _ in ALERT_MAP:
        cur.execute("INSERT INTO alert_routing VALUES (?, ?, ?)", (acode, symp, b_svc))
    conn.commit()

    def python_memory_lookup(query_str: str) -> tuple[str, float]:
        t0 = time.perf_counter()
        out_lines = []
        # Check if an alert code or service name is in query_str
        for acode, symp, b_svc, _, _ in ALERT_MAP:
            if acode.lower() in query_str.lower():
                out_lines.append(f"[ALERT_ROUTING] {acode} ({symp}) -> backing_service={b_svc}")
        for _, svc, db, _, _, _, _ in U_SPECS:
            if svc.lower() in query_str.lower():
                cur.execute(
                    """
                    SELECT attribute, value, session_date FROM fact_lineage f1
                    WHERE entity=? AND NOT EXISTS (
                        SELECT 1 FROM fact_lineage f2 WHERE f2.supersedes_fact_id = f1.fact_id
                    )
                    """,
                    (svc,),
                )
                for a, v, dt in cur.fetchall():
                    out_lines.append(f"[ACTIVE_FACT] {svc}.{a} = {v} (updated {dt})")
        db_ms = (time.perf_counter() - t0) * 1000.0
        return ("\n".join(out_lines) if out_lines else "[NO_DIRECT_ENTITY_MATCH]"), db_ms

    m4_1hop_hook, m4_1hop_tool = [], []
    m4_2hop_hook, m4_2hop_tool = [], []

    for acode, symp, b_svc, exp_port, exp_id in ALERT_MAP:
        # --- 1-HOP TASK (Query names `b_svc` directly) ---
        q1 = f"What is the current active port and database identity for {b_svc}? Reply in JSON {{\"port\": \"...\", \"identity\": \"...\"}}."
        # Policy A: Pre-Turn Hook
        mem_txt, db_ms = python_memory_lookup(q1)
        msgs_h1 = [
            {"role": "system", "content": "You are an SRE agent. Use the pre-retrieved memory to answer in JSON."},
            {"role": "user", "content": f"PRE-TURN MEMORY:\n{mem_txt}\n\nQuestion: {q1}"},
        ]
        r_h1 = lcp_prefill_and_decode(model, tok, [], None, render_chat_ids(tok, msgs_h1, enable_thinking=False), max_new_tokens=48)
        m4_1hop_hook.append({
            "service": b_svc, "llm_calls": 1, "db_ms": round(db_ms, 3),
            "gpu_ms": r_h1["total_ms"], "total_tokens": r_h1["prompt_tokens"] + r_h1["decode_tokens"],
            "passed": (exp_port in r_h1["gen_text"]) and (exp_id in r_h1["gen_text"]),
            "output": r_h1["gen_text"],
        })

        # Policy B: Agent-Directed Memory Tool Loop (Call 1 emits archival_search, Call 2 answers)
        msgs_t1 = [
            {
                "role": "system",
                "content": (
                    "You are an SRE agent with Letta-style memory tools. "
                    "To look up a service or alert in archival memory, output `SEARCH: <entity_or_alert>`. "
                    "Once you have the facts, output JSON `{\"port\": \"...\", \"identity\": \"...\"}`."
                ),
            },
            {"role": "user", "content": q1},
        ]
        c1_res = lcp_prefill_and_decode(model, tok, [], None, render_chat_ids(tok, msgs_t1, enable_thinking=False), max_new_tokens=32)
        obs1, db_ms1 = python_memory_lookup(c1_res["gen_text"] + " " + b_svc)
        msgs_t1_step2 = msgs_t1 + [
            {"role": "assistant", "content": c1_res["gen_text"]},
            {"role": "user", "content": f"TOOL_RESPONSE:\n{obs1}\nNow output the final JSON."},
        ]
        c2_res = lcp_prefill_and_decode(model, tok, [], None, render_chat_ids(tok, msgs_t1_step2, enable_thinking=False), max_new_tokens=48)
        m4_1hop_tool.append({
            "service": b_svc, "llm_calls": 2, "db_ms": round(db_ms1, 3),
            "gpu_ms": round(c1_res["total_ms"] + c2_res["total_ms"], 3),
            "total_tokens": c1_res["prompt_tokens"] + c1_res["decode_tokens"] + c2_res["prompt_tokens"] + c2_res["decode_tokens"],
            "passed": (exp_port in c2_res["gen_text"]) and (exp_id in c2_res["gen_text"]),
            "output": c2_res["gen_text"],
        })

        # --- 2-HOP TASK (Query only names `acode`, NOT `b_svc`!) ---
        q2 = f"Alert {acode} ({symp}) is firing. Which backing service is affected, and what is its current database port and identity? Reply in JSON."
        # Policy A: Single Pre-Turn Hook (only matches `acode` -> discovers `b_svc`, but does NOT have `b_svc`'s port/identity in the single pre-turn lookup!)
        mem_txt2, db_ms2 = python_memory_lookup(q2)
        msgs_h2 = [
            {"role": "system", "content": "You are an SRE agent. Use the pre-retrieved memory to answer in JSON."},
            {"role": "user", "content": f"PRE-TURN MEMORY:\n{mem_txt2}\n\nQuestion: {q2}"},
        ]
        r_h2 = lcp_prefill_and_decode(model, tok, [], None, render_chat_ids(tok, msgs_h2, enable_thinking=False), max_new_tokens=64)
        m4_2hop_hook.append({
            "alert": acode, "backing_service": b_svc, "llm_calls": 1, "db_ms": round(db_ms2, 3),
            "gpu_ms": r_h2["total_ms"], "total_tokens": r_h2["prompt_tokens"] + r_h2["decode_tokens"],
            "passed": (exp_port in r_h2["gen_text"]) and (exp_id in r_h2["gen_text"]),
            "output": r_h2["gen_text"],
        })

        # Policy B: Hybrid Pre-Turn Hook + Agent-Directed Follow-Up Tool Call (Hop 1 via pre-turn hook reveals `b_svc`, Call 1 searches `b_svc`, Call 2 answers!)
        msgs_t2 = [
            {
                "role": "system",
                "content": (
                    "You are an SRE agent with archival search. If the pre-turn memory shows a backing_service whose port is missing, "
                    "output `SEARCH: <backing_service>`. Once you have the port and identity, output JSON."
                ),
            },
            {"role": "user", "content": f"PRE-TURN MEMORY:\n{mem_txt2}\n\nQuestion: {q2}"},
        ]
        s1_res = lcp_prefill_and_decode(model, tok, [], None, render_chat_ids(tok, msgs_t2, enable_thinking=False), max_new_tokens=32)
        obs2, db_ms_hop2 = python_memory_lookup(s1_res["gen_text"])
        msgs_t2_step2 = msgs_t2 + [
            {"role": "assistant", "content": s1_res["gen_text"]},
            {"role": "user", "content": f"TOOL_RESPONSE:\n{obs2}\nNow output the final JSON with service, port, and identity."},
        ]
        s2_res = lcp_prefill_and_decode(model, tok, [], None, render_chat_ids(tok, msgs_t2_step2, enable_thinking=False), max_new_tokens=64)
        m4_2hop_tool.append({
            "alert": acode, "backing_service": b_svc, "llm_calls": 2, "db_ms": round(db_ms2 + db_ms_hop2, 3),
            "gpu_ms": round(s1_res["total_ms"] + s2_res["total_ms"], 3),
            "total_tokens": s1_res["prompt_tokens"] + s1_res["decode_tokens"] + s2_res["prompt_tokens"] + s2_res["decode_tokens"],
            "passed": (exp_port in s2_res["gen_text"]) and (exp_id in s2_res["gen_text"]),
            "output": s2_res["gen_text"],
        })

    m4_summary = {
        "single_hop_4_tasks": {
            "pre_turn_hook": {
                "passed_count": sum(1 for r in m4_1hop_hook if r["passed"]),
                "mean_gpu_ms": round(float(np.mean([r["gpu_ms"] for r in m4_1hop_hook])), 2),
                "mean_db_ms": round(float(np.mean([r["db_ms"] for r in m4_1hop_hook])), 3),
                "mean_llm_calls": 1.0,
                "rows": m4_1hop_hook,
            },
            "agent_tool_loop": {
                "passed_count": sum(1 for r in m4_1hop_tool if r["passed"]),
                "mean_gpu_ms": round(float(np.mean([r["gpu_ms"] for r in m4_1hop_tool])), 2),
                "mean_db_ms": round(float(np.mean([r["db_ms"] for r in m4_1hop_tool])), 3),
                "mean_llm_calls": 2.0,
                "rows": m4_1hop_tool,
            },
            "latency_speedup_hook_vs_tool": round(
                float(np.mean([r["gpu_ms"] for r in m4_1hop_tool])) / max(1e-6, float(np.mean([r["gpu_ms"] for r in m4_1hop_hook]))), 2
            ),
        },
        "two_hop_4_tasks": {
            "pre_turn_hook_only": {
                "passed_count": sum(1 for r in m4_2hop_hook if r["passed"]),
                "mean_gpu_ms": round(float(np.mean([r["gpu_ms"] for r in m4_2hop_hook])), 2),
                "rows": m4_2hop_hook,
            },
            "hybrid_hook_plus_search_tool": {
                "passed_count": sum(1 for r in m4_2hop_tool if r["passed"]),
                "mean_gpu_ms": round(float(np.mean([r["gpu_ms"] for r in m4_2hop_tool])), 2),
                "rows": m4_2hop_tool,
            },
        },
    }
    print(
        f"[M4] Done! 1-Hop Hook={m4_summary['single_hop_4_tasks']['pre_turn_hook']['passed_count']}/4 "
        f"({m4_summary['single_hop_4_tasks']['pre_turn_hook']['mean_gpu_ms']} ms) vs "
        f"Tool={m4_summary['single_hop_4_tasks']['agent_tool_loop']['passed_count']}/4 "
        f"({m4_summary['single_hop_4_tasks']['agent_tool_loop']['mean_gpu_ms']} ms, "
        f"{m4_summary['single_hop_4_tasks']['latency_speedup_hook_vs_tool']}x) | "
        f"2-Hop HookOnly={m4_summary['two_hop_4_tasks']['pre_turn_hook_only']['passed_count']}/4 vs "
        f"HybridTool={m4_summary['two_hop_4_tasks']['hybrid_hook_plus_search_tool']['passed_count']}/4",
        flush=True,
    )

    all_results = {
        "m0_hardware_and_preflight": m0_info,
        "m1_sliding_window_vs_prefix_cache": m1_summary,
        "m2_compaction_prose_vs_schema": m2_summary,
        "m3_knowledge_updates_and_lineage": m3_summary,
        "m4_control_policy_hook_vs_tool": m4_summary,
    }
    OUT_JSON.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    _stop_hb.set()
    print(f"=== SAVED ALL EPISODE 14 TELEMETRY TO {OUT_JSON} ===", flush=True)


if __name__ == "__main__":
    main()
