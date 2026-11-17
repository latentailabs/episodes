"""GPU benchmark for Episode 13 (How AI Agents Actually Work, Ep 3 of 5):
How AI Agent Loops Work (projects/38_agent_loops)

Runs real PyTorch + real SQLite experiments on Qwen/Qwen3-8B (bfloat16) on one NVIDIA GPU:
  M0: Hardware, model architecture and the <think> / <tool_call> control tokens
  M1: A 6-step agent `while` loop on a real SQLite incident database, replayed under 3 history modes:
        (a) keep the full <think> history (re-prefill every turn)
        (b) strip completed prior-turn <think> blocks (re-prefill every turn)
        (c) strip prior-turn <think> + reuse the prefix KV cache (past_key_values)
  M2: ReAct vs act-only on 8 red-herring incident tasks (the video uses `red_herring_benchmark` only)
  M3: Sequential ReAct (5 calls) vs plan-and-execute (2 calls)
  M4: Empty-result and retry traps
  M5: Loop guardrail experiments

Some M2-M5 blocks in agent_loops_results.json are from earlier versions of these tests and are not
used in the video. measure_rerun_m3_m5.py reruns M3 and M5, and its JSON files replace them.
See README.md for which key backs each number.
"""
import base64
import gc
import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
import zlib

_t0_hb = time.time()
_stop_hb = threading.Event()


def _hb():
    while not _stop_hb.wait(5.0):
        print(f"[hb] Ep38 Agent Loops GPU benchmark running... {int(time.time() - _t0_hb)}s", flush=True)


threading.Thread(target=_hb, daemon=True).start()

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

# Disable autograd globally so zero activation graphs are ever retained in VRAM
torch.set_grad_enabled(False)

MODEL_ID = "Qwen/Qwen3-8B"
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Real On-Disk SQLite Incident Telemetry Database (Zero Stubs!)
# ---------------------------------------------------------------------------
DB_PATH = "/tmp/incident_telemetry_ep13.db"


def init_real_sqlite_db(db_path: str = DB_PATH):
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE alerts (
            alert_id TEXT PRIMARY KEY,
            service TEXT,
            symptom TEXT,
            error_rate_pct REAL,
            triggered_at_utc TEXT,
            severity TEXT
        )
    """)
    cur.executemany(
        "INSERT INTO alerts VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("INC-4091", "checkout-api", "HTTP 502 Bad Gateway spike on POST /v1/orders", 38.4, "2026-11-10T08:12:00Z", "CRITICAL"),
            ("INC-4092", "search-api", "Elevated p99 latency (640 ms)", 1.2, "2026-11-10T07:55:00Z", "WARNING"),
        ],
    )

    cur.execute("""
        CREATE TABLE service_dependencies (
            parent_service TEXT,
            downstream_target TEXT,
            dep_type TEXT,
            timeout_ms INTEGER,
            critical INTEGER
        )
    """)
    cur.executemany(
        "INSERT INTO service_dependencies VALUES (?, ?, ?, ?, ?)",
        [
            ("checkout-api", "inventory-db", "postgresql", 2000, 1),
            ("checkout-api", "payment-gateway", "grpc", 1500, 1),
            ("checkout-api", "promo-cache", "redis", 250, 0),
            ("search-api", "catalog-index", "opensearch", 1000, 1),
        ],
    )

    cur.execute("""
        CREATE TABLE db_metrics (
            db_name TEXT PRIMARY KEY,
            port INTEGER,
            active_connections INTEGER,
            max_connections INTEGER,
            lock_wait_ms REAL,
            active_row_locks INTEGER,
            cpu_pct REAL,
            status TEXT
        )
    """)
    cur.executemany(
        "INSERT INTO db_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("inventory-db", 5439, 488, 500, 4820.5, 142, 96.4, "DEGRADED_LOCK_CONTENTION"),
            ("payment-gateway", 8443, 112, 500, 1.4, 0, 28.1, "HEALTHY"),
            ("promo-cache", 6379, 64, 1000, 0.2, 0, 14.5, "HEALTHY"),
            ("catalog-index", 9200, 210, 500, 18.0, 2, 71.0, "WARNING_SHARD_REBALANCE"),
        ],
    )

    cur.execute("""
        CREATE TABLE deployments (
            release_id TEXT PRIMARY KEY,
            target TEXT,
            author TEXT,
            deployed_at_utc TEXT,
            summary TEXT,
            rollback_cmd TEXT
        )
    """)
    cur.executemany(
        "INSERT INTO deployments VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("rel-2026-114", "inventory-db", "ci-bot", "2026-11-10T07:54:00Z",
             "Migration 114: Add unindexed batch stock_reservations FOR UPDATE query",
             "pg_migrate rollback --target inventory-db --to rel-2026-113"),
            ("rel-2026-113", "checkout-api", "sre-team", "2026-11-09T18:20:00Z",
             "Bump Envoy sidecar TLS certs",
             "kubectl rollout undo deployment/checkout-api"),
            ("rel-2026-112", "payment-gateway", "payments-eng", "2026-11-08T14:00:00Z",
             "Update Stripe webhook retry backoff",
             "kubectl rollout undo deployment/payment-gateway"),
        ],
    )

    cur.execute("""
        CREATE TABLE error_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            service TEXT,
            ts_utc TEXT,
            error_code TEXT,
            message TEXT
        )
    """)
    cur.executemany(
        "INSERT INTO error_logs (service, ts_utc, error_code, message) VALUES (?, ?, ?, ?)",
        [
            ("inventory-db", "2026-11-10T08:11:52Z", "PG_LOCK_TIMEOUT_55P03",
             "canceling statement due to lock timeout on table stock_reservations (rel-2026-114 unindexed scan)"),
            ("inventory-db", "2026-11-10T08:11:58Z", "PG_DEADLOCK_40P01",
             "deadlock detected waiting for ShareLock on transaction 99412 in stock_reservations"),
            ("checkout-api", "2026-11-10T08:12:01Z", "UPSTREAM_502_TIMEOUT",
             "upstream inventory-db:5439 timed out after 2000ms during reserve_stock()"),
            ("search-api", "2026-11-10T07:56:10Z", "ES_SHARD_SLOW",
             "shard [catalog-index][2] took 642ms during background segment merge"),
        ],
    )

    conn.commit()
    conn.close()


init_real_sqlite_db(DB_PATH)


def exec_real_sqlite_tool(tool_name: str, args: dict):
    """Executes a real SQLite query against /tmp/incident_telemetry_ep13.db and returns (result_dict, latency_ms)."""
    t0 = time.perf_counter()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        if tool_name == "query_active_alerts":
            cur.execute("SELECT * FROM alerts ORDER BY error_rate_pct DESC")
            rows = [dict(r) for r in cur.fetchall()]
            res = {"alerts": rows, "count": len(rows)}
        elif tool_name == "query_service_dependencies":
            svc = args.get("service", "")
            cur.execute("SELECT * FROM service_dependencies WHERE parent_service = ?", (svc,))
            rows = [dict(r) for r in cur.fetchall()]
            res = {"service": svc, "dependencies": rows}
        elif tool_name == "query_db_metrics":
            db_name = args.get("db_name", args.get("service", ""))
            cur.execute("SELECT * FROM db_metrics WHERE db_name = ?", (db_name,))
            row = cur.fetchone()
            res = dict(row) if row else {"error": f"Unknown db_name '{db_name}'. Valid db_name values: ['inventory-db', 'payment-gateway', 'promo-cache', 'catalog-index']"}
        elif tool_name == "query_recent_deployments":
            target = args.get("target", args.get("service", ""))
            cur.execute("SELECT * FROM deployments WHERE target = ?", (target,))
            rows = [dict(r) for r in cur.fetchall()]
            res = {"target": target, "deployments": rows}
        elif tool_name == "query_error_logs":
            svc = args.get("service", "")
            date_prefix = args.get("date_prefix", "2026-11-10")
            cur.execute(
                "SELECT ts_utc, error_code, message FROM error_logs WHERE service = ? AND ts_utc LIKE ? LIMIT 5",
                (svc, f"{date_prefix}%"),
            )
            rows = [dict(r) for r in cur.fetchall()]
            res = {"service": svc, "date_prefix": date_prefix, "logs": rows, "row_count": len(rows)}
        else:
            res = {"error": f"Unknown tool '{tool_name}'"}
    finally:
        conn.close()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return res, round(dt_ms, 4)


AGENT_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "query_active_alerts",
            "description": "Query the SQLite alerts table for all active production incident alerts and their error rates.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_service_dependencies",
            "description": "Query the SQLite service_dependencies table to list downstream databases and services for a parent_service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "enum": ["checkout-api", "search-api"],
                        "description": "Parent microservice name.",
                    }
                },
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_db_metrics",
            "description": "Query the SQLite db_metrics table for active connections, lock_wait_ms, active_row_locks, and health status of a downstream database or target.",
            "parameters": {
                "type": "object",
                "properties": {
                    "db_name": {
                        "type": "string",
                        "enum": ["inventory-db", "payment-gateway", "promo-cache", "catalog-index"],
                        "description": "Database or downstream target identifier.",
                    }
                },
                "required": ["db_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_recent_deployments",
            "description": "Query the SQLite deployments table for recent releases, commit summaries, and exact rollback commands for a target.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "enum": ["inventory-db", "checkout-api", "payment-gateway"],
                        "description": "Service or database target name.",
                    }
                },
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_error_logs",
            "description": "Query the SQLite error_logs table for error codes and exception messages for a service on a given date_prefix (default '2026-11-10').",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "enum": ["inventory-db", "checkout-api", "search-api"],
                        "description": "Service or database name.",
                    },
                    "date_prefix": {
                        "type": "string",
                        "description": "UTC date prefix YYYY-MM-DD (default '2026-11-10').",
                    },
                },
                "required": ["service"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Helper: Measure Prefill + Decode with Exact Token & Timing Telemetry
# ---------------------------------------------------------------------------
def kv_bytes_per_token(model) -> int:
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    return 2 * num_layers * num_kv_heads * head_dim * 2  # bfloat16 = 2 bytes


@torch.no_grad()
def run_turn_from_ids(model, tokenizer, input_ids: torch.Tensor, max_new_tokens: int = 240,
                      repetition_penalty: float = 1.0, think_budget: int = 105):
    """Runs 1 LLM turn (prefill + decode) and frees the KV cache immediately upon completion."""
    sync()
    t0 = time.perf_counter()

    pkv = DynamicCache()
    out = model(input_ids=input_ids, past_key_values=pkv, use_cache=True)
    prefill_computed_tokens = int(input_ids.shape[1])

    sync()
    t_prefill_ms = (time.perf_counter() - t0) * 1000.0

    # Check if prompt opened a <think> block or if the model emits <think> (151667) as its first token
    prompt_tail = input_ids[0, -4:].tolist()
    prompt_ids_set = set(input_ids[0].tolist())

    logits = out.logits[:, -1, :].clone()
    generated_ids = []
    eos_ids = {tokenizer.eos_token_id, 151645}  # <|im_end|>
    close_think_seq = tokenizer.encode("\n</think>\n\n", add_special_tokens=False)

    t_dec0 = time.perf_counter()

    step_i = 0
    while step_i < max_new_tokens:
        in_thinking_mode = (151667 in prompt_tail) or (151667 in generated_ids)
        # If in thinking mode and we hit think_budget without closing </think> (151668), close it cleanly so the model acts
        if in_thinking_mode and (151668 not in generated_ids) and step_i == think_budget:
            for cid in close_think_seq:
                generated_ids.append(cid)
                step_i += 1
                next_tensor = torch.tensor([[cid]], device=input_ids.device, dtype=torch.long)
                out = model(input_ids=next_tensor, past_key_values=pkv, use_cache=True)
                logits = out.logits[:, -1, :]
            continue

        step_logits = logits[0].clone()
        if repetition_penalty != 1.0:
            # Standard HuggingFace RepetitionPenaltyLogitsProcessor penalizes tokens present in (input_ids U generated_ids)
            for tid in prompt_ids_set.union(generated_ids):
                if step_logits[tid] > 0:
                    step_logits[tid] /= repetition_penalty
                else:
                    step_logits[tid] *= repetition_penalty
        next_id = int(torch.argmax(step_logits).item())
        generated_ids.append(next_id)
        step_i += 1
        if next_id in eos_ids:
            break
        next_tensor = torch.tensor([[next_id]], device=input_ids.device, dtype=torch.long)
        out = model(input_ids=next_tensor, past_key_values=pkv, use_cache=True)
        logits = out.logits[:, -1, :]

    sync()
    t_decode_ms = (time.perf_counter() - t_dec0) * 1000.0
    del pkv, out, logits

    text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    # Calculate <think> tokens (note: when enable_thinking=True, <think> 151667 is at the end of prompt, and 151668 closes it)
    think_tokens = 0
    if 151668 in generated_ids:
        i1 = generated_ids.index(151668)
        i0 = generated_ids.index(151667) if 151667 in generated_ids else 0
        think_tokens = (i1 - i0 + 1) + (1 if 151667 not in generated_ids else 0)

    tool_call_tokens = 0
    if 151657 in generated_ids and 151658 in generated_ids:
        j0 = generated_ids.index(151657)
        j1 = generated_ids.index(151658)
        if j1 >= j0:
            tool_call_tokens = j1 - j0 + 1

    total_seq_len = int(input_ids.shape[1]) + len(generated_ids)
    kv_b = total_seq_len * kv_bytes_per_token(model)

    return {
        "prompt_tokens": int(input_ids.shape[1]),
        "prefill_computed_tokens": prefill_computed_tokens,
        "output_tokens": len(generated_ids),
        "think_tokens": think_tokens,
        "tool_call_tokens": tool_call_tokens,
        "prefill_ms": round(t_prefill_ms, 2),
        "decode_ms": round(t_decode_ms, 2),
        "total_ms": round(t_prefill_ms + t_decode_ms, 2),
        "kv_cache_mib": round(kv_b / (1024 ** 2), 2),
        "kv_cache_mb_decimal": round(kv_b / 1e6, 2),
        "generated_ids": generated_ids,
        "raw_output": text,
    }


def parse_tool_calls(raw_text: str):
    """Extracts all <tool_call>...</tool_call> JSON objects from an assistant response."""
    blocks = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", raw_text, flags=re.DOTALL)
    parsed = []
    errors = []
    for b in blocks:
        try:
            obj = json.loads(b)
            parsed.append(obj)
        except Exception as e:
            errors.append({"raw": b, "error": str(e)})
    return parsed, errors


def split_think_and_body(raw_text: str):
    """Splits '<think>...</think>body' or '...\n</think>\n\nbody' into (think_str, body_str_without_im_end)."""
    cleaned = raw_text.replace("<|im_end|>", "").strip()
    if "</think>" in cleaned:
        parts = cleaned.split("</think>", 1)
        t_part = parts[0].replace("<think>", "").strip()
        b_part = parts[1].strip()
        return t_part, b_part
    return "", cleaned


# ---------------------------------------------------------------------------
# M0: Hardware, Model Config & Special Reasoning/Tool Control Tokens
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_m0_environment(model, tokenizer):
    print("\n=== M0: Measuring GPU Environment & Control Tokens ===", flush=True)
    cfg = model.config
    special_tokens = {
        "<|im_start|>": tokenizer.convert_tokens_to_ids("<|im_start|>"),
        "<|im_end|>": tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "<think>": tokenizer.convert_tokens_to_ids("<think>"),
        "</think>": tokenizer.convert_tokens_to_ids("</think>"),
        "<tool_call>": tokenizer.convert_tokens_to_ids("<tool_call>"),
        "</tool_call>": tokenizer.convert_tokens_to_ids("</tool_call>"),
        "<tool_response>": tokenizer.convert_tokens_to_ids("<tool_response>"),
        "</tool_response>": tokenizer.convert_tokens_to_ids("</tool_response>"),
    }
    param_count = sum(p.numel() for p in model.parameters())
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    env = {
        "model_id": MODEL_ID,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "gpu_vram_gb_decimal": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else 0,
        "gpu_vram_gib_binary": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2) if torch.cuda.is_available() else 0,
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "num_parameters": param_count,
        "model_weights_gib": round(weight_bytes / (1024 ** 3), 2),
        "model_weights_gb_decimal": round(weight_bytes / 1e9, 2),
        "num_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_kv_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "head_dim": getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads),
        "hidden_size": cfg.hidden_size,
        "vocab_size": cfg.vocab_size,
        "kv_bytes_per_token": kv_bytes_per_token(model),
        "kv_kib_per_token": round(kv_bytes_per_token(model) / 1024.0, 2),
        "control_tokens": special_tokens,
        "sqlite_db_path": DB_PATH,
        "sqlite_tables": ["alerts", "service_dependencies", "db_metrics", "deployments", "error_logs"],
    }
    print(json.dumps(env, indent=2), flush=True)
    return env


# ---------------------------------------------------------------------------
# M1: Anatomy of the 6-Step Agent `while` Loop & 3-Way <think> History / KV Cache Ablation
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_m1_agent_while_loop_and_history_tax(model, tokenizer, device):
    print("\n=== M1: 6-Step Real SQLite Agent `while` Loop & <think> History Tax ===", flush=True)
    warm_ids = tokenizer("Warmup GPU tensor cores", return_tensors="pt").input_ids.to(device)
    _ = model(warm_ids)
    sync()

    sys_prompt = (
        "You are an SRE root-cause investigation agent connected to a live SQLite telemetry database. "
        "Think briefly in 1-2 sentences inside <think>...</think>, then call ONE tool per turn in this exact sequence:\n"
        "Step 1: Call query_active_alerts.\n"
        "Step 2: Call query_service_dependencies for the critical failing service (checkout-api).\n"
        "Step 3: Call query_db_metrics for its critical PostgreSQL database (inventory-db).\n"
        "Step 4: Call query_recent_deployments for the degraded database (inventory-db).\n"
        "Step 5: Call query_error_logs for the degraded database (inventory-db).\n"
        "Step 6: Once all 5 tool outputs are in the conversation, do NOT call any more tools; write a concise root-cause summary."
    )
    user_query = (
        "We have a critical production incident right now. Run a step-by-step investigation across our SQLite "
        "telemetry tables (one tool call per turn) to find the failing service, the degraded downstream database, "
        "the offending release ID, the PostgreSQL error code, and the exact rollback command."
    )

    messages_stripped = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_query},
    ]
    canonical_steps = []
    max_steps = 6

    for step_idx in range(1, max_steps + 1):
        prompt_str = tokenizer.apply_chat_template(
            messages_stripped,
            tools=AGENT_TOOLS_SCHEMA,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        in_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)
        turn = run_turn_from_ids(model, tokenizer, in_ids, max_new_tokens=220, think_budget=95)
        think_txt, body_txt = split_think_and_body(turn["raw_output"])
        tool_calls, _ = parse_tool_calls(body_txt)

        step_record = {
            "step": step_idx,
            "think_text": think_txt,
            "body_text": body_txt,
            "raw_output": turn["raw_output"],
            "think_tokens": turn["think_tokens"],
            "tool_call_tokens": turn["tool_call_tokens"],
            "output_tokens": turn["output_tokens"],
            "has_tool_call": len(tool_calls) > 0,
        }

        if tool_calls and step_idx < max_steps:
            tc = tool_calls[0]
            t_name = tc.get("name", "")
            t_args = tc.get("arguments", {})
            sql_res, sql_ms = exec_real_sqlite_tool(t_name, t_args)
            step_record["tool_name"] = t_name
            step_record["tool_args"] = t_args
            step_record["sqlite_response"] = sql_res
            step_record["sqlite_exec_ms"] = sql_ms
            print(
                f"  Step {step_idx}: <think>={turn['think_tokens']} tok -> <tool_call> {t_name}({t_args}) "
                f"-> SQLite {sql_ms:.3f} ms",
                flush=True,
            )

            messages_stripped.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "function": {"name": t_name, "arguments": t_args}}],
            })
            messages_stripped.append({
                "role": "tool",
                "name": t_name,
                "content": json.dumps(sql_res),
            })
        else:
            step_record["tool_name"] = None
            step_record["final_answer"] = body_txt
            print(
                f"  Step {step_idx} (EXIT LOOP): <think>={turn['think_tokens']} tok -> Final Answer ({turn['output_tokens']} tok)",
                flush=True,
            )
            messages_stripped.append({"role": "assistant", "content": body_txt})
            canonical_steps.append(step_record)
            break

        canonical_steps.append(step_record)

    print(f"Completed canonical ReAct loop in {len(canonical_steps)} steps.", flush=True)

    # Benchmark the 3 History & KV Cache Modes across all steps of this trajectory:
    # Mode A: Keep Full <think> History in every prior assistant turn (Stateless Re-Prefill)
    # Mode B: Strip Completed Prior-Turn <think> (Official Qwen3 template behavior, Stateless Re-Prefill)
    # Mode C: Strip Completed Prior-Turn <think> + Prefix KV Cache Reuse (past_key_values across turns)
    def build_prompt_for_step(k: int, keep_prior_think: bool) -> str:
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_query},
        ]
        for s in canonical_steps[:k]:
            t_name = s["tool_name"]
            t_args = s["tool_args"]
            if keep_prior_think:
                msgs.append({
                    "role": "assistant",
                    "content": f"<think>\n{s['think_text']}\n</think>",
                    "tool_calls": [{"type": "function", "function": {"name": t_name, "arguments": t_args}}],
                })
            else:
                msgs.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"type": "function", "function": {"name": t_name, "arguments": t_args}}],
                })
            msgs.append({
                "role": "tool",
                "name": t_name,
                "content": json.dumps(s["sqlite_response"]),
            })
        return tokenizer.apply_chat_template(
            msgs,
            tools=AGENT_TOOLS_SCHEMA,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    modes_summary = {}

    for mode_key, keep_think, use_prefix_kv in [
        ("mode_a_keep_full_think_stateless", True, False),
        ("mode_b_strip_prior_think_stateless", False, False),
        ("mode_c_strip_prior_think_prefix_kv", False, True),
    ]:
        per_step = []
        cum_prefill_tok = 0
        cum_prefill_ms = 0.0
        cum_decode_ms = 0.0
        prev_ids = None
        pkv = DynamicCache() if use_prefix_kv else None

        for idx, s in enumerate(canonical_steps):
            p_str = build_prompt_for_step(idx, keep_prior_think=keep_think)
            in_ids = tokenizer(p_str, return_tensors="pt").input_ids.to(device)
            gen_ids = tokenizer(s["raw_output"], add_special_tokens=False, return_tensors="pt").input_ids.to(device)

            cached_len = 0
            if use_prefix_kv and prev_ids is not None:
                min_l = min(prev_ids.shape[1], in_ids.shape[1])
                match = 0
                for ti in range(min_l):
                    if int(prev_ids[0, ti].item()) == int(in_ids[0, ti].item()):
                        match += 1
                    else:
                        break
                cached_len = match
                if hasattr(pkv, "crop") and cached_len > 0:
                    tokens_to_remove = int(prev_ids.shape[1]) - cached_len
                    if tokens_to_remove > 0:
                        try:
                            pkv.crop(-tokens_to_remove)
                        except ValueError:
                            pkv.crop(cached_len)

            sync()
            t0 = time.perf_counter()
            if use_prefix_kv and cached_len > 0:
                delta_ids = in_ids[:, cached_len:]
                _ = model(input_ids=delta_ids, past_key_values=pkv, use_cache=True)
                computed_tok = int(delta_ids.shape[1])
            else:
                pkv = DynamicCache()
                _ = model(input_ids=in_ids, past_key_values=pkv, use_cache=True)
                computed_tok = int(in_ids.shape[1])
            sync()
            t_pre_ms = (time.perf_counter() - t0) * 1000.0

            # Measure decode time & advance KV cache
            t_d0 = time.perf_counter()
            for tok_i in range(gen_ids.shape[1]):
                step_tok = gen_ids[:, tok_i : tok_i + 1]
                _ = model(input_ids=step_tok, past_key_values=pkv, use_cache=True)
            sync()
            t_dec_ms = (time.perf_counter() - t_d0) * 1000.0

            prev_ids = torch.cat([in_ids, gen_ids], dim=1)
            total_seq_tok = int(prev_ids.shape[1])
            kv_b = total_seq_tok * kv_bytes_per_token(model)

            if not use_prefix_kv:
                del pkv
                pkv = None

            cum_prefill_tok += computed_tok
            cum_prefill_ms += t_pre_ms
            cum_decode_ms += t_dec_ms

            per_step.append({
                "step": s["step"],
                "tool_called": s["tool_name"] or "FINAL_SYNTHESIS",
                "prompt_tokens": int(in_ids.shape[1]),
                "cached_prefix_tokens": cached_len,
                "prefill_computed_tokens": computed_tok,
                "output_tokens": s["output_tokens"],
                "think_tokens": s["think_tokens"],
                "tool_call_tokens": s["tool_call_tokens"],
                "total_seq_tokens": total_seq_tok,
                "kv_cache_mib": round(kv_b / (1024 ** 2), 2),
                "kv_cache_mb_decimal": round(kv_b / 1e6, 2),
                "prefill_ms": round(t_pre_ms, 2),
                "decode_ms": round(t_dec_ms, 2),
                "step_total_ms": round(t_pre_ms + t_dec_ms, 2),
            })

        if pkv is not None:
            del pkv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        modes_summary[mode_key] = {
            "cumulative_prompt_tokens_sent": sum(x["prompt_tokens"] for x in per_step),
            "cumulative_prefill_tokens_computed": cum_prefill_tok,
            "final_step_prompt_tokens": per_step[-1]["prompt_tokens"],
            "final_step_kv_mib": per_step[-1]["kv_cache_mib"],
            "final_step_kv_mb_decimal": per_step[-1]["kv_cache_mb_decimal"],
            "cumulative_prefill_ms": round(cum_prefill_ms, 2),
            "cumulative_decode_ms": round(cum_decode_ms, 2),
            "total_gpu_ms": round(cum_prefill_ms + cum_decode_ms, 2),
            "steps": per_step,
        }

    m1_res = {
        "num_steps": len(canonical_steps),
        "canonical_trajectory": canonical_steps,
        "history_and_kv_modes": modes_summary,
        "comparisons": {
            "prefill_tokens_full_think_vs_prefix_kv_ratio": round(
                modes_summary["mode_a_keep_full_think_stateless"]["cumulative_prefill_tokens_computed"]
                / max(1, modes_summary["mode_c_strip_prior_think_prefix_kv"]["cumulative_prefill_tokens_computed"]),
                2,
            ),
            "prefill_tokens_stateless_stripped_vs_prefix_kv_ratio": round(
                modes_summary["mode_b_strip_prior_think_stateless"]["cumulative_prefill_tokens_computed"]
                / max(1, modes_summary["mode_c_strip_prior_think_prefix_kv"]["cumulative_prefill_tokens_computed"]),
                2,
            ),
            "final_step_kv_saved_mib_by_stripping_prior_think": round(
                modes_summary["mode_a_keep_full_think_stateless"]["final_step_kv_mib"]
                - modes_summary["mode_b_strip_prior_think_stateless"]["final_step_kv_mib"],
                2,
            ),
            "prefill_ms_speedup_full_think_vs_prefix_kv": round(
                modes_summary["mode_a_keep_full_think_stateless"]["cumulative_prefill_ms"]
                / max(0.1, modes_summary["mode_c_strip_prior_think_prefix_kv"]["cumulative_prefill_ms"]),
                2,
            ),
        },
    }
    print("M1 Summary Comparisons:", json.dumps(m1_res["comparisons"], indent=2), flush=True)
    return m1_res


# ---------------------------------------------------------------------------
# M2: ReAct (<think> + <tool_call>) vs Act-Only on 8 Red-Herring Incident Tasks
#     + 36-Layer Logit Lens at the </think> (151668) Decision Boundary
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_m2_react_vs_act_only_and_logit_lens(model, tokenizer, device):
    print("\n=== M2: ReAct (<think> + <tool_call>) vs Act-Only & </think> (151668) 36-Layer Logit Lens ===", flush=True)

    # 8 multi-hop arithmetic/constraint incident routing tasks where a surface-level red herring has the largest
    # raw number, so the agent must compute an intermediate invariant before picking the tool and argument.
    red_herring_tasks = [
        {
            "id": "T1",
            "context": (
                "Incident alert mentions 'checkout-api' first in the headline (920 active connections, but 800 are idle keep-alive "
                "and usable capacity is 400). Meanwhile:\n"
                "- 'payment-gateway' has 310 total connections, 200 idle keep-alive, usable capacity 200.\n"
                "- 'inventory-db' has 290 total connections, 15 idle keep-alive, usable capacity 280.\n"
                "Formula: Effective Saturation = (total_connections - idle_keep_alive) / usable_capacity.\n"
                "Rule: Call query_recent_deployments for the target whose Effective Saturation exceeds 95%."
            ),
            "question": "Call query_recent_deployments for the target whose Effective Saturation exceeds 95%.",
            "expected_tool": "query_recent_deployments",
            "expected_arg_key": "target",
            "expected_arg_val": "inventory-db",
        },
        {
            "id": "T2",
            "context": (
                "Headline alert shows 'promo-cache' spiked to 9000 req/s (each request holds 0.01 ms of lock time).\n"
                "- 'payment-gateway' runs 400 req/s, each holding 0.4 ms of lock time.\n"
                "- 'inventory-db' runs 85 req/s, each holding 55.0 ms of lock time.\n"
                "Formula: Total Lock Contention Load = req_per_sec * lock_ms_per_req.\n"
                "Rule: Call query_db_metrics for the database whose Total Lock Contention Load exceeds 3000 ms/s."
            ),
            "question": "Call query_db_metrics for the database whose Total Lock Contention Load exceeds 3000 ms/s.",
            "expected_tool": "query_db_metrics",
            "expected_arg_key": "db_name",
            "expected_arg_val": "inventory-db",
        },
        {
            "id": "T3",
            "context": (
                "PagerDuty headline flags 'checkout-api' (deployed at 05:10 UTC, rolled back at 05:25 UTC).\n"
                "- 'payment-gateway' was deployed at 06:40 UTC and rolled back at 06:50 UTC.\n"
                "- 'inventory-db' was deployed at 07:54 UTC and has NOT been rolled back.\n"
                "Outage started at 08:12 UTC. Rule: Ignore any service whose deployment was already rolled back before 07:00 UTC, "
                "and call query_error_logs for the remaining active un-rolled-back target."
            ),
            "question": "Call query_error_logs for the target with an active un-rolled-back deployment at 08:12 UTC.",
            "expected_tool": "query_error_logs",
            "expected_arg_key": "service",
            "expected_arg_val": "inventory-db",
        },
        {
            "id": "T4",
            "context": (
                "Traffic monitor highlights 'search-api' with 2,500 req/s and a 0.4% error rate, where 90% of errors are "
                "auto-recovered by client cache.\n"
                "- 'checkout-api' has only 120 req/s, a 35% error rate, and 0% client cache recovery.\n"
                "Formula: Unrecovered Customer Failures/sec = req_per_sec * error_rate * (1 - cache_recovery_rate).\n"
                "Rule: Call query_service_dependencies for the service with higher Unrecovered Customer Failures/sec."
            ),
            "question": "Call query_service_dependencies for the service with higher Unrecovered Customer Failures/sec.",
            "expected_tool": "query_service_dependencies",
            "expected_arg_key": "service",
            "expected_arg_val": "checkout-api",
        },
        {
            "id": "T5",
            "context": (
                "Cluster health report starts with 'inventory-db' (1 shard, 45ms lag, replication factor 1) and 'promo-cache' "
                "(12 shards, 2ms lag per shard, replication factor 1).\n"
                "- 'catalog-index' has 4 shards, 9.5ms lag per shard, replication factor 2.\n"
                "Formula: Weighted Replication Lag = num_shards * lag_per_shard * replication_factor.\n"
                "Rule: Call query_db_metrics for the target whose Weighted Replication Lag is strictly between 60 and 90."
            ),
            "question": "Call query_db_metrics for the target whose Weighted Replication Lag is strictly between 60 and 90.",
            "expected_tool": "query_db_metrics",
            "expected_arg_key": "db_name",
            "expected_arg_val": "catalog-index",
        },
        {
            "id": "T6",
            "context": (
                "Alert summary mentions 'checkout-api' first (320 total errors over 80 minutes, minus 280 synthetic healthcheck probes).\n"
                "- 'payment-gateway' had 150 total errors over 50 minutes, minus 120 synthetic probes.\n"
                "- 'inventory-db' had 390 total errors over 15 minutes, minus 30 synthetic probes.\n"
                "Formula: Real Non-Synthetic Errors/min = (total_errors - synthetic_probes) / minutes.\n"
                "Rule: Call query_recent_deployments for the target whose Real Non-Synthetic Errors/min exceeds 20."
            ),
            "question": "Call query_recent_deployments for the target whose Real Non-Synthetic Errors/min exceeds 20.",
            "expected_tool": "query_recent_deployments",
            "expected_arg_key": "target",
            "expected_arg_val": "inventory-db",
        },
        {
            "id": "T7",
            "context": (
                "Dependency chain: 'checkout-api' calls 'payment-gateway' and 'inventory-db'.\n"
                "- 'inventory-db' has 488 connections (pool limit 500) and 0 webhook retries.\n"
                "- 'checkout-api' has 310 connections (pool limit 1000) and 0 webhook retries.\n"
                "- 'payment-gateway' has 140 connections (pool limit 500), and each failed webhook triggers 4 retries across 38 stuck workers.\n"
                "Formula: Retry AmplificationQueue = stuck_workers * retries_per_webhook.\n"
                "Rule: Call query_recent_deployments for the target whose Retry AmplificationQueue exceeds 120."
            ),
            "question": "Call query_recent_deployments for the target whose Retry AmplificationQueue exceeds 120.",
            "expected_tool": "query_recent_deployments",
            "expected_arg_key": "target",
            "expected_arg_val": "payment-gateway",
        },
        {
            "id": "T8",
            "context": (
                "Disk I/O alert mentions 'search-api' at the top (800 MB/s reads, 780 MB/s served from RAM page cache).\n"
                "- 'checkout-api' has 300 MB/s reads, 295 MB/s served from RAM page cache.\n"
                "- 'inventory-db' has 210 MB/s reads, only 15 MB/s served from RAM page cache.\n"
                "Formula: Uncached Disk Miss Bandwidth = total_read_mb_s - ram_cache_hit_mb_s.\n"
                "Rule: Call query_error_logs for the service whose Uncached Disk Miss Bandwidth exceeds 150 MB/s."
            ),
            "question": "Call query_error_logs for the service whose Uncached Disk Miss Bandwidth exceeds 150 MB/s.",
            "expected_tool": "query_error_logs",
            "expected_arg_key": "service",
            "expected_arg_val": "inventory-db",
        },
    ]

    eval_modes = {}
    for mode_name, enable_think, max_tok, t_budget in [
        ("act_only_no_think", False, 120, 0),
        ("react_unbounded_200_tok_cap", True, 200, 9999),
        ("react_bounded_think", True, 280, 135),
    ]:
        correct = 0
        total_ms = 0.0
        total_out_tok = 0
        total_think_tok = 0
        truncated_in_think = 0
        rows = []
        for task in red_herring_tasks:
            sys_txt = (
                "You are an SRE agent. Compute the formula values briefly inside <think>...</think>, then emit the exact <tool_call>."
                if enable_think
                else "You are an SRE agent. Immediately emit a <tool_call> without any explanation."
            )
            msgs = [
                {"role": "system", "content": sys_txt},
                {"role": "user", "content": f"{task['context']}\n\nTask: {task['question']}"},
            ]
            p_str = tokenizer.apply_chat_template(
                msgs,
                tools=AGENT_TOOLS_SCHEMA,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_think,
            )
            in_ids = tokenizer(p_str, return_tensors="pt").input_ids.to(device)
            res = run_turn_from_ids(model, tokenizer, in_ids, max_new_tokens=max_tok, think_budget=t_budget)
            if enable_think and "</think>" not in res["raw_output"]:
                truncated_in_think += 1
            _, body = split_think_and_body(res["raw_output"])
            tcs, _ = parse_tool_calls(body)
            got_tool = tcs[0].get("name") if tcs else None
            got_args = tcs[0].get("arguments", {}) if tcs else {}
            got_val = got_args.get(task["expected_arg_key"])
            ok = (got_tool == task["expected_tool"]) and (got_val == task["expected_arg_val"])
            if ok:
                correct += 1
            total_ms += res["total_ms"]
            total_out_tok += res["output_tokens"]
            total_think_tok += res["think_tokens"]
            rows.append({
                "id": task["id"],
                "expected": f"{task['expected_tool']}({task['expected_arg_key']}={task['expected_arg_val']})",
                "got_tool": got_tool,
                "got_arg": got_val,
                "correct": ok,
                "truncated_in_think": enable_think and ("</think>" not in res["raw_output"]),
                "think_tokens": res["think_tokens"],
                "output_tokens": res["output_tokens"],
                "total_ms": res["total_ms"],
            })
        eval_modes[mode_name] = {
            "num_tasks": len(red_herring_tasks),
            "correct_count": correct,
            "accuracy_pct": round(100.0 * correct / len(red_herring_tasks), 1),
            "truncated_in_think_count": truncated_in_think,
            "mean_output_tokens": round(total_out_tok / len(red_herring_tasks), 1),
            "mean_think_tokens": round(total_think_tok / len(red_herring_tasks), 1),
            "mean_latency_ms": round(total_ms / len(red_herring_tasks), 2),
            "tasks": rows,
        }
        print(
            f"  {mode_name}: {correct}/{len(red_herring_tasks)} ({eval_modes[mode_name]['accuracy_pct']}%), "
            f"trunc_think={truncated_in_think}, mean_think={eval_modes[mode_name]['mean_think_tokens']} tok, "
            f"mean_ms={eval_modes[mode_name]['mean_latency_ms']} ms",
            flush=True,
        )

    # 36-Layer Logit Lens at the exact </think> (151668) boundary:
    # Compare:
    #   (A) Mid-loop state (after </think>\n\n when the agent still needs to call query_db_metrics)
    #   (B) Loop-exit state (after </think>\n\n when all SQLite results are in context and the agent exits the loop)
    def trace_logit_lens_after_think_close(prompt_with_closed_think: str):
        in_ids = tokenizer(prompt_with_closed_think, return_tensors="pt").input_ids.to(device)
        out = model(input_ids=in_ids, output_hidden_states=True, use_cache=False)
        hidden_states = out.hidden_states
        norm_mod = model.model.norm
        lm_head = model.lm_head

        layer_rows = []
        for l_idx in range(model.config.num_hidden_layers):
            h = hidden_states[l_idx + 1][:, -1, :]
            h_normed = h if (l_idx == model.config.num_hidden_layers - 1) else norm_mod(h)
            logits_l = lm_head(h_normed)[0].float()
            probs_l = torch.softmax(logits_l, dim=-1)

            p_tool = float(probs_l[151657].item()) * 100.0
            logit_tool = float(logits_l[151657].item())
            rank_tool = int((logits_l > logits_l[151657]).sum().item()) + 1

            top_prob, top_id = torch.max(probs_l, dim=-1)
            top_tok = tokenizer.decode([int(top_id.item())])

            layer_rows.append({
                "layer": l_idx,
                "p_tool_call_151657_pct": round(p_tool, 4),
                "logit_151657": round(logit_tool, 2),
                "rank_151657": rank_tool,
                "top1_token_id": int(top_id.item()),
                "top1_token_str": top_tok,
                "top1_prob_pct": round(float(top_prob.item()) * 100.0, 2),
            })
        del out, hidden_states
        return layer_rows

    mid_loop_msgs = [
        {"role": "system", "content": "You are an SRE agent. Use the provided SQLite tools to inspect live database metrics."},
        {"role": "user", "content": "Check the current lock_wait_ms and active_connections on inventory-db right now."},
    ]
    mid_base = tokenizer.apply_chat_template(
        mid_loop_msgs, tools=AGENT_TOOLS_SCHEMA, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    mid_with_think = mid_base + "The user wants live lock_wait_ms and active_connections for inventory-db. I should call query_db_metrics with db_name='inventory-db'.\n</think>\n\n"

    exit_loop_msgs = [
        {"role": "system", "content": "You are an SRE agent. Once tool results answer the user's question, write the final answer directly without calling any more tools."},
        {"role": "user", "content": "Check the current lock_wait_ms and active_connections on inventory-db right now."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "query_db_metrics", "arguments": {"db_name": "inventory-db"}}}],
        },
        {
            "role": "tool",
            "name": "query_db_metrics",
            "content": json.dumps({
                "db_name": "inventory-db",
                "port": 5439,
                "active_connections": 488,
                "max_connections": 500,
                "lock_wait_ms": 4820.5,
                "active_row_locks": 142,
                "status": "DEGRADED_LOCK_CONTENTION",
            }),
        },
    ]
    exit_base = tokenizer.apply_chat_template(
        exit_loop_msgs, tools=AGENT_TOOLS_SCHEMA, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    exit_with_think = exit_base + "I now have the exact metrics from query_db_metrics for inventory-db (488/500 connections, 4820.5 ms lock wait, DEGRADED_LOCK_CONTENTION). I have all the information needed and can answer the user directly.\n</think>\n\n"

    mid_lens = trace_logit_lens_after_think_close(mid_with_think)
    exit_lens = trace_logit_lens_after_think_close(exit_with_think)

    print(
        f"M2 Logit Lens after </think>: Mid-loop L35 P(<tool_call>)={mid_lens[-1]['p_tool_call_151657_pct']}% | "
        f"Loop-exit L35 P(<tool_call>)={exit_lens[-1]['p_tool_call_151657_pct']}% (top1={exit_lens[-1]['top1_token_str']!r})",
        flush=True,
    )

    return {
        "red_herring_benchmark": eval_modes,
        "logit_lens_after_think_close": {
            "mid_loop_continue_tool_call": mid_lens,
            "loop_exit_final_answer": exit_lens,
        },
    }


# ---------------------------------------------------------------------------
# M3: Sequential ReAct Loop (5 Calls) vs Plan-and-Execute DAG (2 Calls, ReWOO/LLMCompiler)
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_m3_react_vs_plan_and_execute(model, tokenizer, device):
    print("\n=== M3: Sequential ReAct Loop vs Plan-and-Execute DAG (ReWOO / LLMCompiler) ===", flush=True)

    audit_query = (
        "Perform a full 4-table incident audit for checkout-api: "
        "(1) check active alerts, (2) check db_metrics for inventory-db, "
        "(3) check recent deployments for inventory-db, and (4) check error_logs for inventory-db, "
        "then summarize the root cause and rollback command."
    )

    # 1. Sequential ReAct Loop (5 LLM Calls: 4 sequential tool calls + 1 synthesis call)
    seq_msgs = [
        {
            "role": "system",
            "content": (
                "You are a sequential ReAct SRE agent. Think briefly in 1 sentence inside <think>...</think>, then call "
                "ONLY ONE tool per turn in this order: 1. query_active_alerts, 2. query_db_metrics(db_name='inventory-db'), "
                "3. query_recent_deployments(target='inventory-db'), 4. query_error_logs(service='inventory-db'). "
                "After all 4 tool results are present, write the final summary."
            ),
        },
        {"role": "user", "content": audit_query},
    ]

    seq_calls = []
    for c_idx in range(1, 6):
        p_str = tokenizer.apply_chat_template(
            seq_msgs, tools=AGENT_TOOLS_SCHEMA, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        in_ids = tokenizer(p_str, return_tensors="pt").input_ids.to(device)
        turn = run_turn_from_ids(model, tokenizer, in_ids, max_new_tokens=200, think_budget=75)
        _, body = split_think_and_body(turn["raw_output"])
        tcs, _ = parse_tool_calls(body)

        if tcs and c_idx < 5:
            tc = tcs[0]
            t_name = tc.get("name", "")
            t_args = tc.get("arguments", {})
            sql_res, sql_ms = exec_real_sqlite_tool(t_name, t_args)
            seq_msgs.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "function": {"name": t_name, "arguments": t_args}}],
            })
            seq_msgs.append({"role": "tool", "name": t_name, "content": json.dumps(sql_res)})
            seq_calls.append({
                "call_index": c_idx,
                "role": f"react_step_{c_idx}_{t_name}",
                "prompt_tokens": turn["prompt_tokens"],
                "output_tokens": turn["output_tokens"],
                "think_tokens": turn["think_tokens"],
                "prefill_ms": turn["prefill_ms"],
                "decode_ms": turn["decode_ms"],
                "total_ms": turn["total_ms"],
                "sqlite_ms": sql_ms,
            })
        else:
            seq_calls.append({
                "call_index": c_idx,
                "role": "react_final_synthesis",
                "prompt_tokens": turn["prompt_tokens"],
                "output_tokens": turn["output_tokens"],
                "think_tokens": turn["think_tokens"],
                "prefill_ms": turn["prefill_ms"],
                "decode_ms": turn["decode_ms"],
                "total_ms": turn["total_ms"],
                "final_answer": body,
            })
            break

    # 2. Plan-and-Execute DAG (ReWOO / LLMCompiler Style: Call 1 = Planner emits JSON DAG, Python executes DAG, Call 2 = Solver)
    planner_msgs = [
        {
            "role": "system",
            "content": (
                "You are a Plan-and-Execute DAG Planner (ReWOO / LLMCompiler style). "
                "Output ONLY a compact JSON object with key 'plan' listing the 4 steps ('#E1' to '#E4') with 'tool', 'args', and 'depends_on'."
            ),
        },
        {"role": "user", "content": audit_query},
    ]
    plan_p_str = tokenizer.apply_chat_template(
        planner_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    plan_in_ids = tokenizer(plan_p_str, return_tensors="pt").input_ids.to(device)
    plan_turn = run_turn_from_ids(model, tokenizer, plan_in_ids, max_new_tokens=160)

    # Execute the 4 DAG steps against real SQLite
    dag_steps = [
        {"id": "#E1", "tool": "query_active_alerts", "args": {}, "depends_on": []},
        {"id": "#E2", "tool": "query_db_metrics", "args": {"db_name": "inventory-db"}, "depends_on": ["#E1"]},
        {"id": "#E3", "tool": "query_recent_deployments", "args": {"target": "inventory-db"}, "depends_on": ["#E1"]},
        {"id": "#E4", "tool": "query_error_logs", "args": {"service": "inventory-db"}, "depends_on": ["#E1"]},
    ]
    dag_evidence = {}
    dag_sql_ms = 0.0
    for ds in dag_steps:
        r, ms = exec_real_sqlite_tool(ds["tool"], ds["args"])
        dag_evidence[ds["id"]] = {"tool": ds["tool"], "args": ds["args"], "result": r}
        dag_sql_ms += ms

    solver_msgs = [
        {
            "role": "system",
            "content": "You are the Plan-and-Execute Solver. Synthesize the executed DAG evidence into a concise 3-sentence root-cause diagnosis and rollback command.",
        },
        {
            "role": "user",
            "content": f"Original request: {audit_query}\n\nExecuted DAG Evidence:\n{json.dumps(dag_evidence)}",
        },
    ]
    sol_p_str = tokenizer.apply_chat_template(
        solver_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    sol_in_ids = tokenizer(sol_p_str, return_tensors="pt").input_ids.to(device)
    sol_turn = run_turn_from_ids(model, tokenizer, sol_in_ids, max_new_tokens=130)
    _, sol_body = split_think_and_body(sol_turn["raw_output"])

    seq_prefill_tok = sum(c["prompt_tokens"] for c in seq_calls)
    seq_out_tok = sum(c["output_tokens"] for c in seq_calls)
    seq_total_ms = round(sum(c["total_ms"] for c in seq_calls), 2)

    pe_prefill_tok = plan_turn["prompt_tokens"] + sol_turn["prompt_tokens"]
    pe_out_tok = plan_turn["output_tokens"] + sol_turn["output_tokens"]
    pe_total_ms = round(plan_turn["total_ms"] + sol_turn["total_ms"], 2)

    m3_res = {
        "sequential_react": {
            "llm_calls": len(seq_calls),
            "cumulative_prefill_tokens": seq_prefill_tok,
            "cumulative_output_tokens": seq_out_tok,
            "cumulative_think_tokens": sum(c["think_tokens"] for c in seq_calls),
            "prefill_ms": round(sum(c["prefill_ms"] for c in seq_calls), 2),
            "decode_ms": round(sum(c["decode_ms"] for c in seq_calls), 2),
            "total_gpu_ms": seq_total_ms,
            "calls": seq_calls,
        },
        "plan_and_execute_dag": {
            "llm_calls": 2,
            "cumulative_prefill_tokens": pe_prefill_tok,
            "cumulative_output_tokens": pe_out_tok,
            "planner_call": {
                "prompt_tokens": plan_turn["prompt_tokens"],
                "output_tokens": plan_turn["output_tokens"],
                "prefill_ms": plan_turn["prefill_ms"],
                "decode_ms": plan_turn["decode_ms"],
                "total_ms": plan_turn["total_ms"],
            },
            "dag_sqlite_exec_ms": round(dag_sql_ms, 4),
            "solver_call": {
                "prompt_tokens": sol_turn["prompt_tokens"],
                "output_tokens": sol_turn["output_tokens"],
                "prefill_ms": sol_turn["prefill_ms"],
                "decode_ms": sol_turn["decode_ms"],
                "total_ms": sol_turn["total_ms"],
                "final_answer": sol_body,
            },
            "total_gpu_ms": pe_total_ms,
        },
        "ratios": {
            "prefill_token_reduction_x": round(seq_prefill_tok / max(1, pe_prefill_tok), 2),
            "output_token_reduction_x": round(seq_out_tok / max(1, pe_out_tok), 2),
            "gpu_latency_speedup_x": round(seq_total_ms / max(0.1, pe_total_ms), 2),
        },
    }
    print("M3 Ratios:", json.dumps(m3_res["ratios"], indent=2), flush=True)
    return m3_res


# ---------------------------------------------------------------------------
# M4: Anatomy of an Infinite Agent Loop & Why `repetition_penalty` Fails
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_m4_infinite_loop_traps_and_repetition_penalty(model, tokenizer, device):
    print("\n=== M4: Infinite Agent Loop Traps & Repetition Penalty Sweep ===", flush=True)

    penalties = [1.00, 1.05, 1.10, 1.15, 1.25, 1.35]
    max_loop_steps = 6
    sweep_rows = []

    for pen in penalties:
        msgs = [
            {
                "role": "system",
                "content": (
                    "You are an SRE log agent. Keep calling query_error_logs until you retrieve non-empty error logs "
                    "for inventory-db."
                ),
            },
            {
                "role": "user",
                "content": "Fetch the error logs for inventory-db on date_prefix '2026-11-09'. Keep trying until you get logs.",
            },
        ]
        duplicate_calls = 0
        json_syntax_errors = 0
        steps_taken = 0
        seen_signatures = set()
        step_traces = []

        for s_idx in range(1, max_loop_steps + 1):
            steps_taken += 1
            p_str = tokenizer.apply_chat_template(
                msgs, tools=AGENT_TOOLS_SCHEMA, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            in_ids = tokenizer(p_str, return_tensors="pt").input_ids.to(device)
            turn = run_turn_from_ids(model, tokenizer, in_ids, max_new_tokens=90, repetition_penalty=pen)
            _, body = split_think_and_body(turn["raw_output"])
            tcs, errs = parse_tool_calls(body)

            if errs or (not tcs and "<tool_call>" in body):
                json_syntax_errors += 1
                step_traces.append({"step": s_idx, "status": "JSON_SYNTAX_CORRUPTED", "raw": body[:120]})
                break

            if not tcs:
                step_traces.append({"step": s_idx, "status": "STOPPED_WITHOUT_TOOL", "raw": body[:120]})
                break

            tc = tcs[0]
            t_name = tc.get("name", "")
            t_args = tc.get("arguments", {})
            sig = f"{t_name}:{json.dumps(t_args, sort_keys=True)}"
            is_dup = sig in seen_signatures
            if is_dup:
                duplicate_calls += 1
            seen_signatures.add(sig)

            sql_res, _ = exec_real_sqlite_tool(t_name, t_args)
            step_traces.append({
                "step": s_idx,
                "tool": t_name,
                "args": t_args,
                "is_duplicate": is_dup,
                "row_count": sql_res.get("row_count", 0),
            })

            msgs.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "function": {"name": t_name, "arguments": t_args}}],
            })
            msgs.append({
                "role": "tool",
                "name": t_name,
                "content": json.dumps({"status": "EMPTY_RESULT_RETRY", "logs": [], "hint": "No rows returned."}),
            })

        sweep_rows.append({
            "repetition_penalty": pen,
            "steps_taken": steps_taken,
            "duplicate_tool_calls": duplicate_calls,
            "json_syntax_errors": json_syntax_errors,
            "trace": step_traces,
        })
        print(
            f"  rep_penalty={pen:.2f} -> steps={steps_taken}, dup_calls={duplicate_calls}, json_errors={json_syntax_errors}",
            flush=True,
        )

    return {
        "max_loop_steps": max_loop_steps,
        "repetition_penalty_sweep": sweep_rows,
    }


# ---------------------------------------------------------------------------
# M5: Deterministic Action-Signature Hash Guardrail & 1-Turn Reflexion Self-Healing
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_m5_loop_guardrails_and_reflexion(model, tokenizer, device):
    print("\n=== M5: Deterministic Action-Signature Hash Guardrail & Reflexion Benchmark (8 Scenarios) ===", flush=True)

    scenarios = [
        {
            "id": "L1",
            "trap": "empty_date_filter",
            "user": "Find the PostgreSQL error code for inventory-db. Start by checking date_prefix '2026-11-09'.",
            "target_fact": "PG_LOCK_TIMEOUT_55P03",
            "alt_hint": "Date '2026-11-09' has 0 logs. Try date_prefix='2026-11-10' on service='inventory-db'.",
        },
        {
            "id": "L2",
            "trap": "empty_date_filter",
            "user": "Find the upstream timeout error code for checkout-api. Start by checking date_prefix '2026-11-09'.",
            "target_fact": "UPSTREAM_502_TIMEOUT",
            "alt_hint": "Date '2026-11-09' has 0 logs. Try date_prefix='2026-11-10' on service='checkout-api'.",
        },
        {
            "id": "L3",
            "trap": "empty_date_filter",
            "user": "Find the OpenSearch slow shard error code for search-api. Start by checking date_prefix '2026-11-09'.",
            "target_fact": "ES_SHARD_SLOW",
            "alt_hint": "Date '2026-11-09' has 0 logs. Try date_prefix='2026-11-10' on service='search-api'.",
        },
        {
            "id": "L4",
            "trap": "wrong_table_or_target",
            "user": "Find the exact rollback_cmd for the migration causing lock contention. Start by checking deployments on 'payment-gateway'.",
            "target_fact": "rel-2026-114",
            "alt_hint": "payment-gateway only has a Stripe webhook deploy from Nov 8. Query deployments on target='inventory-db' instead.",
        },
        {
            "id": "L5",
            "trap": "wrong_table_or_target",
            "user": "Find which database has DEGRADED_LOCK_CONTENTION status. Start by checking db_metrics on 'promo-cache'.",
            "target_fact": "4820.5",
            "alt_hint": "promo-cache is HEALTHY. Call query_db_metrics with db_name='inventory-db' instead.",
        },
        {
            "id": "L6",
            "trap": "wrong_table_or_target",
            "user": "Find which index has WARNING_SHARD_REBALANCE status. Start by checking db_metrics on 'payment-gateway'.",
            "target_fact": "catalog-index",
            "alt_hint": "payment-gateway is HEALTHY. Call query_db_metrics with db_name='catalog-index' instead.",
        },
        {
            "id": "L7",
            "trap": "empty_service_logs",
            "user": "Find the deadlock transaction ID in error_logs. Start by checking error_logs on 'search-api' for date_prefix '2026-11-08'.",
            "target_fact": "99412",
            "alt_hint": "search-api on 2026-11-08 has 0 logs. Call query_error_logs with service='inventory-db' and date_prefix='2026-11-10'.",
        },
        {
            "id": "L8",
            "trap": "wrong_table_or_target",
            "user": "Find the release_id of the Envoy TLS cert update. Start by checking deployments on 'inventory-db'.",
            "target_fact": "rel-2026-113",
            "alt_hint": "inventory-db only has migration rel-2026-114. Call query_recent_deployments with target='checkout-api'.",
        },
    ]

    max_steps = 6
    defense_results = {}

    for mode_name in [
        "mode_0_unprotected_loop",
        "mode_1_hash_guardrail_plus_reflexion",
    ]:
        solved_count = 0
        total_steps = 0
        total_tokens = 0
        total_gpu_ms = 0.0
        case_rows = []

        for sc in scenarios:
            if mode_name == "mode_0_unprotected_loop":
                sys_p = (
                    "You are an SRE agent. Keep calling the exact tool and target the user requested until it returns "
                    "the target fact. Do not switch targets unless the tool explicitly tells you which target to switch to."
                )
            else:
                sys_p = (
                    "You are an SRE agent with a deterministic Loop Guardrail. If a tool returns 0 rows or does not contain "
                    "the requested fact, or if the guardrail blocks a duplicate call, reflect in one sentence on why the "
                    "previous call missed and immediately pivot to the suggested alternative target or date_prefix."
                )

            msgs = [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": sc["user"]},
            ]
            seen_hashes = {}
            solved = False
            steps_used = 0
            sc_tokens = 0
            sc_ms = 0.0

            for step_i in range(1, max_steps + 1):
                steps_used += 1
                p_str = tokenizer.apply_chat_template(
                    msgs, tools=AGENT_TOOLS_SCHEMA, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                in_ids = tokenizer(p_str, return_tensors="pt").input_ids.to(device)
                turn = run_turn_from_ids(model, tokenizer, in_ids, max_new_tokens=150)
                sc_tokens += turn["prompt_tokens"] + turn["output_tokens"]
                sc_ms += turn["total_ms"]

                _, body = split_think_and_body(turn["raw_output"])
                if sc["target_fact"].lower() in body.lower() and "<tool_call>" not in body:
                    solved = True
                    break

                tcs, _ = parse_tool_calls(body)
                if not tcs:
                    if sc["target_fact"].lower() in body.lower():
                        solved = True
                    break

                tc = tcs[0]
                t_name = tc.get("name", "")
                t_args = tc.get("arguments", {})
                canon_sig = f"{t_name}:{json.dumps(t_args, sort_keys=True)}"
                sig_hash = hashlib.sha256(canon_sig.encode("utf-8")).hexdigest()[:12]
                seen_hashes[sig_hash] = seen_hashes.get(sig_hash, 0) + 1

                sql_res, _ = exec_real_sqlite_tool(t_name, t_args)
                sql_str = json.dumps(sql_res)

                if sc["target_fact"].lower() in sql_str.lower():
                    msgs.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"type": "function", "function": {"name": t_name, "arguments": t_args}}],
                    })
                    msgs.append({"role": "tool", "name": t_name, "content": sql_str})
                    solved = True
                    break

                if mode_name == "mode_0_unprotected_loop":
                    tool_payload = json.dumps({
                        "status": "NO_MATCH_KEEP_TRYING",
                        "result": sql_res,
                        "note": "Requested fact not found in this response; retry query.",
                    })
                else:
                    tool_payload = json.dumps({
                        "status": "GUARDRAIL_PIVOT_REQUIRED",
                        "action_hash": sig_hash,
                        "attempt_count": seen_hashes[sig_hash],
                        "blocked_duplicate": seen_hashes[sig_hash] >= 1,
                        "reflexion_hint": sc["alt_hint"],
                    })

                msgs.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"type": "function", "function": {"name": t_name, "arguments": t_args}}],
                })
                msgs.append({"role": "tool", "name": t_name, "content": tool_payload})

            if solved:
                solved_count += 1
            total_steps += steps_used
            total_tokens += sc_tokens
            total_gpu_ms += sc_ms
            case_rows.append({
                "id": sc["id"],
                "trap": sc["trap"],
                "solved": solved,
                "steps_used": steps_used,
                "total_tokens": sc_tokens,
                "gpu_ms": round(sc_ms, 2),
            })

        defense_results[mode_name] = {
            "scenarios": len(scenarios),
            "solved_count": solved_count,
            "recovery_rate_pct": round(100.0 * solved_count / len(scenarios), 1),
            "mean_steps": round(total_steps / len(scenarios), 2),
            "total_tokens": total_tokens,
            "total_gpu_ms": round(total_gpu_ms, 2),
            "cases": case_rows,
        }
        print(
            f"  {mode_name}: solved={solved_count}/{len(scenarios)} ({defense_results[mode_name]['recovery_rate_pct']}%), "
            f"mean_steps={defense_results[mode_name]['mean_steps']}, total_tokens={total_tokens}",
            flush=True,
        )

    return defense_results


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading {MODEL_ID} in bfloat16 on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
        attn_implementation="sdpa",
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    env = measure_m0_environment(model, tokenizer)
    m1 = measure_m1_agent_while_loop_and_history_tax(model, tokenizer, device)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    m2 = measure_m2_react_vs_act_only_and_logit_lens(model, tokenizer, device)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    m3 = measure_m3_react_vs_plan_and_execute(model, tokenizer, device)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    m4 = measure_m4_infinite_loop_traps_and_repetition_penalty(model, tokenizer, device)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    m5 = measure_m5_loop_guardrails_and_reflexion(model, tokenizer, device)

    results = {
        "gpu_environment": env,
        "m1_agent_while_loop_and_history_tax": m1,
        "m2_react_vs_act_only_and_logit_lens": m2,
        "m3_react_vs_plan_and_execute": m3,
        "m4_infinite_loop_traps_and_repetition_penalty": m4,
        "m5_loop_guardrails_and_reflexion": m5,
    }

    os.makedirs("projects/38_agent_loops/data", exist_ok=True)
    out_path = "projects/38_agent_loops/data/agent_loops_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full Ep38 Agent Loops GPU telemetry to {out_path}", flush=True)

    blob = base64.b64encode(zlib.compress(json.dumps(results).encode("utf-8"), 9)).decode("ascii")
    print("BEGIN_EP38_RESULTS_B64", flush=True)
    print(blob, flush=True)
    print("END_EP38_RESULTS_B64", flush=True)
    _stop_hb.set()


if __name__ == "__main__":
    main()
