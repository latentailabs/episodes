"""Ep13 rerun (12 Nov 2026): honest M3 / M4b / M5 after the go-live review.
M3v2: planner sees tool schemas, its JSON plan is PARSED and executed (no hard-coded DAG), caps raised, completion checked.
M4b: empty-date trap WITHOUT "keep trying" coercion, real SQLite results returned, rep penalty 1.00 vs 1.35.
M5v2: same system prompt in both modes, real SQLite results, generic guardrail text (no per-scenario answer hints),
      duplicate actions really blocked (not executed), solved = target fact in the model final answer.
Helpers (DB, tools schema, run_turn_from_ids, parsers) are copied unchanged from measure_agent_loops.py.
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
# Rerun helpers
# ---------------------------------------------------------------------------
TOOL_NAMES = {t["function"]["name"] if "function" in t else t["name"] for t in AGENT_TOOLS_SCHEMA}
EOS_IDS = {151645}


def run_turn_v2(model, tokenizer, input_ids, max_new_tokens=512, repetition_penalty=1.0, think_budget=None):
    """Same greedy turn as run_turn_from_ids, with two fixes:
    (1) a think budget is only enforced when the prompt really ends inside an OPEN <think> (no </think> after it),
        so enable_thinking=False prompts are never force-closed;
    (2) reports hit_cap (generation stopped at max_new_tokens instead of <|im_end|>)."""
    sync()
    t0 = time.perf_counter()
    pkv = DynamicCache()
    out = model(input_ids=input_ids, past_key_values=pkv, use_cache=True)
    sync()
    t_prefill_ms = (time.perf_counter() - t0) * 1000.0
    ids = input_ids[0].tolist()
    last_open = max([i for i, t in enumerate(ids) if t == 151667], default=-1)
    last_close = max([i for i, t in enumerate(ids) if t == 151668], default=-1)
    prompt_open_think = last_open > last_close
    prompt_ids_set = set(ids)
    logits = out.logits[:, -1, :].clone()
    gen = []
    close_seq = tokenizer.encode("\n</think>\n\n", add_special_tokens=False)
    t_dec0 = time.perf_counter()
    step_i = 0
    ended = False
    while step_i < max_new_tokens:
        if (think_budget is not None and prompt_open_think and 151668 not in gen and step_i == think_budget):
            for cid in close_seq:
                gen.append(cid)
                step_i += 1
                out = model(input_ids=torch.tensor([[cid]], device=input_ids.device), past_key_values=pkv, use_cache=True)
                logits = out.logits[:, -1, :]
            continue
        sl = logits[0].clone()
        if repetition_penalty != 1.0:
            for tid in prompt_ids_set.union(gen):
                sl[tid] = sl[tid] / repetition_penalty if sl[tid] > 0 else sl[tid] * repetition_penalty
        nid = int(torch.argmax(sl).item())
        gen.append(nid)
        step_i += 1
        if nid in EOS_IDS or nid == tokenizer.eos_token_id:
            ended = True
            break
        out = model(input_ids=torch.tensor([[nid]], device=input_ids.device), past_key_values=pkv, use_cache=True)
        logits = out.logits[:, -1, :]
    sync()
    t_decode_ms = (time.perf_counter() - t_dec0) * 1000.0
    del pkv, out, logits
    think_tokens = 0
    if 151668 in gen:
        think_tokens = gen.index(151668) + 1
    return {
        "prompt_tokens": int(input_ids.shape[1]),
        "output_tokens": len(gen),
        "think_tokens": think_tokens,
        "hit_cap": not ended,
        "prefill_ms": round(t_prefill_ms, 2),
        "decode_ms": round(t_decode_ms, 2),
        "total_ms": round(t_prefill_ms + t_decode_ms, 2),
        "raw_output": tokenizer.decode(gen, skip_special_tokens=False),
    }


def encode(tokenizer, msgs, device, tools=True, thinking=False):
    s = tokenizer.apply_chat_template(
        msgs, tools=AGENT_TOOLS_SCHEMA if tools else None, tokenize=False,
        add_generation_prompt=True, enable_thinking=thinking,
    )
    return tokenizer(s, return_tensors="pt").input_ids.to(device)


def asst_tool_msg(name, args):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}]}


ANSWER_FACTS = {"release": "rel-2026-114", "error_code": "PG_LOCK_TIMEOUT_55P03", "rollback": "pg_migrate rollback"}


def answer_checks(text):
    low = text.lower()
    return {k: (v.lower() in low) for k, v in ANSWER_FACTS.items()}


def is_empty_result(res):
    if not isinstance(res, dict) or "error" in res:
        return True
    if "row_count" in res and res["row_count"] == 0:
        return True
    for k in ("deployments", "dependencies", "alerts", "logs"):
        if k in res and isinstance(res[k], list) and len(res[k]) == 0:
            return True
    return False


# ---------------------------------------------------------------------------
# M3v2: Sequential ReAct vs Plan-and-Execute with a PARSED plan
# ---------------------------------------------------------------------------
AUDIT_QUERY = (
    "Perform a full 4-table incident audit for checkout-api: "
    "(1) check active alerts, (2) check db_metrics for inventory-db, "
    "(3) check recent deployments for inventory-db, and (4) check error_logs for inventory-db, "
    "then summarize the root cause and rollback command."
)


def m3_once(model, tokenizer, device):
    # Sequential ReAct: one tool per turn, stateless re-prefill each call (same as the original M3).
    msgs = [
        {"role": "system", "content": (
            "You are a ReAct SRE agent. Think briefly inside <think>, then call ONE tool per turn. "
            "When every requested check is done, write the final answer: root cause, offending release, "
            "PostgreSQL error code and the exact rollback command.")},
        {"role": "user", "content": AUDIT_QUERY},
    ]
    calls = []
    final = ""
    for c_idx in range(1, 9):
        turn = run_turn_v2(model, tokenizer, encode(tokenizer, msgs, device, thinking=True),
                           max_new_tokens=512, think_budget=75)
        _, body = split_think_and_body(turn["raw_output"])
        tcs, _ = parse_tool_calls(body)
        row = {k: turn[k] for k in ("prompt_tokens", "output_tokens", "think_tokens", "hit_cap",
                                    "prefill_ms", "decode_ms", "total_ms")}
        row["call_index"] = c_idx
        row["tool_calls_emitted"] = len(tcs)
        if tcs:
            tc = tcs[0]
            name, args = tc.get("name", ""), tc.get("arguments", {})
            res, ms = exec_real_sqlite_tool(name, args)
            row.update({"tool": name, "args": args, "sqlite_ms": ms})
            msgs.append(asst_tool_msg(name, args))
            msgs.append({"role": "tool", "name": name, "content": json.dumps(res)})
            calls.append(row)
        else:
            final = body
            row["final_answer"] = body
            calls.append(row)
            break
    seq = {
        "llm_calls": len(calls),
        "tools_called": [c.get("tool") for c in calls if c.get("tool")],
        "cumulative_prefill_tokens": sum(c["prompt_tokens"] for c in calls),
        "cumulative_output_tokens": sum(c["output_tokens"] for c in calls),
        "total_gpu_ms": round(sum(c["total_ms"] for c in calls), 2),
        "any_call_hit_cap": any(c["hit_cap"] for c in calls),
        "final_answer_checks": answer_checks(final),
        "calls": calls,
    }

    # Plan-and-Execute: planner sees the same tool schemas; its JSON plan is parsed and executed as written.
    p_msgs = [
        {"role": "system", "content": (
            "You are the planner in a Plan-and-Execute agent (ReWOO style). Do not call tools and do not answer. "
            "Output ONLY one JSON object: {\"plan\": [{\"id\": \"#E1\", \"tool\": \"<tool name>\", "
            "\"args\": {...}, \"depends_on\": []}, ...]} using the available tools.")},
        {"role": "user", "content": AUDIT_QUERY},
    ]
    plan_turn = run_turn_v2(model, tokenizer, encode(tokenizer, p_msgs, device, thinking=False), max_new_tokens=512)
    _, plan_body = split_think_and_body(plan_turn["raw_output"])
    plan, parse_ok, parse_err = [], False, ""
    try:
        blob = plan_body[plan_body.index("{"): plan_body.rindex("}") + 1]
        obj = json.loads(blob)
        plan = obj.get("plan", obj if isinstance(obj, list) else [])
        parse_ok = isinstance(plan, list) and len(plan) > 0
    except Exception as e:
        parse_err = str(e)[:200]
    evidence, dag_ms, executed = {}, 0.0, []
    for i, st in enumerate(plan if parse_ok else []):
        if not isinstance(st, dict):
            continue
        sid = st.get("id", f"#E{i + 1}")
        name, args = st.get("tool", ""), st.get("args", {}) or {}
        valid = name in TOOL_NAMES
        res, ms = exec_real_sqlite_tool(name, args) if valid else ({"error": f"unknown tool {name}"}, 0.0)
        dag_ms += ms
        evidence[sid] = {"tool": name, "args": args, "result": res}
        executed.append({"id": sid, "tool": name, "args": args, "valid_tool": valid,
                         "depends_on": st.get("depends_on", []), "empty": is_empty_result(res)})
    s_msgs = [
        {"role": "system", "content": (
            "You are the solver in a Plan-and-Execute agent. Using only the evidence, write the root cause, "
            "the offending release, the PostgreSQL error code and the exact rollback command.")},
        {"role": "user", "content": f"Original request: {AUDIT_QUERY}\n\nExecuted plan evidence:\n{json.dumps(evidence)}"},
    ]
    sol_turn = run_turn_v2(model, tokenizer, encode(tokenizer, s_msgs, device, tools=False, thinking=False),
                           max_new_tokens=512)
    _, sol_body = split_think_and_body(sol_turn["raw_output"])
    pe = {
        "llm_calls": 2,
        "plan_parse_ok": parse_ok,
        "plan_parse_error": parse_err,
        "planner_raw": plan_body[:1500],
        "executed_steps": executed,
        "requested_tools_covered": sorted({"query_active_alerts", "query_db_metrics", "query_recent_deployments",
                                           "query_error_logs"} & {e["tool"] for e in executed}),
        "dag_sqlite_exec_ms": round(dag_ms, 4),
        "planner_call": {k: plan_turn[k] for k in ("prompt_tokens", "output_tokens", "hit_cap",
                                                   "prefill_ms", "decode_ms", "total_ms")},
        "solver_call": {k: sol_turn[k] for k in ("prompt_tokens", "output_tokens", "hit_cap",
                                                 "prefill_ms", "decode_ms", "total_ms")},
        "solver_final_answer": sol_body,
        "final_answer_checks": answer_checks(sol_body),
        "cumulative_prefill_tokens": plan_turn["prompt_tokens"] + sol_turn["prompt_tokens"],
        "cumulative_output_tokens": plan_turn["output_tokens"] + sol_turn["output_tokens"],
        "total_gpu_ms": round(plan_turn["total_ms"] + sol_turn["total_ms"], 2),
    }
    return seq, pe


def measure_m3v2(model, tokenizer, device, reps=3):
    print("\n=== M3v2: Sequential ReAct vs parsed Plan-and-Execute ===", flush=True)
    runs = []
    for r in range(reps):
        seq, pe = m3_once(model, tokenizer, device)
        runs.append({"seq": seq, "pe": pe})
        print(f"  rep {r + 1}: seq calls={seq['llm_calls']} prefill={seq['cumulative_prefill_tokens']} "
              f"out={seq['cumulative_output_tokens']} ms={seq['total_gpu_ms']} checks={seq['final_answer_checks']} | "
              f"PE parse={pe['plan_parse_ok']} steps={len(pe['executed_steps'])} prefill={pe['cumulative_prefill_tokens']} "
              f"out={pe['cumulative_output_tokens']} ms={pe['total_gpu_ms']} checks={pe['final_answer_checks']}", flush=True)

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2]

    s_ms = med([r["seq"]["total_gpu_ms"] for r in runs])
    p_ms = med([r["pe"]["total_gpu_ms"] for r in runs])
    s0, p0 = runs[0]["seq"], runs[0]["pe"]
    same_tokens = all(r["seq"]["cumulative_output_tokens"] == s0["cumulative_output_tokens"]
                      and r["pe"]["cumulative_output_tokens"] == p0["cumulative_output_tokens"] for r in runs)
    return {
        "method_notes": [
            "Greedy decoding; both arms stateless (full prompt re-prefilled every call, no prefix KV reuse).",
            "Planner and ReAct calls both receive the same tool schemas.",
            "Planner JSON is parsed and executed exactly as written; nothing hard-coded.",
            "max_new_tokens=512 per call; ReAct think budget 75 tokens; hit_cap recorded per call.",
            f"Timing = median of {reps} repeats; token counts from repeat 1.",
        ],
        "token_counts_identical_across_repeats": same_tokens,
        "sequential_react": s0,
        "plan_and_execute": p0,
        "median_total_gpu_ms": {"sequential_react": s_ms, "plan_and_execute": p_ms},
        "all_repeat_total_gpu_ms": {"sequential_react": [r["seq"]["total_gpu_ms"] for r in runs],
                                    "plan_and_execute": [r["pe"]["total_gpu_ms"] for r in runs]},
        "ratios": {
            "prefill_token_reduction_x": round(s0["cumulative_prefill_tokens"] / max(1, p0["cumulative_prefill_tokens"]), 2),
            "output_token_reduction_x": round(s0["cumulative_output_tokens"] / max(1, p0["cumulative_output_tokens"]), 2),
            "median_gpu_time_speedup_x": round(s_ms / max(0.1, p_ms), 2),
        },
    }


# ---------------------------------------------------------------------------
# M4b: empty-date trap without "keep trying" coercion
# ---------------------------------------------------------------------------
def measure_m4b(model, tokenizer, device):
    print("\n=== M4b: empty-date trap, neutral prompt, real SQLite results ===", flush=True)
    rows = []
    for pen in (1.00, 1.35):
        msgs = [
            {"role": "system", "content": "You are an SRE log agent. Use the tools to answer the user."},
            {"role": "user", "content": "Find the PostgreSQL error code in the inventory-db error logs. "
                                        "Start by checking date_prefix '2026-11-09'."},
        ]
        seen, trace, final = set(), [], ""
        for s in range(1, 7):
            turn = run_turn_v2(model, tokenizer, encode(tokenizer, msgs, device), max_new_tokens=300,
                               repetition_penalty=pen)
            _, body = split_think_and_body(turn["raw_output"])
            tcs, errs = parse_tool_calls(body)
            if errs:
                trace.append({"step": s, "status": "JSON_ERROR", "raw": body[:200]})
                break
            if not tcs:
                final = body
                trace.append({"step": s, "status": "FINAL_ANSWER", "hit_cap": turn["hit_cap"]})
                break
            name, args = tcs[0].get("name", ""), tcs[0].get("arguments", {})
            sig = f"{name}:{json.dumps(args, sort_keys=True)}"
            dup = sig in seen
            seen.add(sig)
            res, _ = exec_real_sqlite_tool(name, args)
            trace.append({"step": s, "tool": name, "args": args, "duplicate": dup,
                          "row_count": res.get("row_count") if isinstance(res, dict) else None})
            msgs.append(asst_tool_msg(name, args))
            msgs.append({"role": "tool", "name": name, "content": json.dumps(res)})
        r = {"repetition_penalty": pen, "steps": len(trace),
             "duplicate_calls": sum(1 for t in trace if t.get("duplicate")),
             "found_error_code_in_answer": "pg_lock_timeout_55p03" in final.lower(),
             "final_answer": final[:600], "trace": trace}
        rows.append(r)
        print(f"  pen={pen}: steps={r['steps']} dups={r['duplicate_calls']} found={r['found_error_code_in_answer']}", flush=True)
    return {"method_notes": ["Greedy; system prompt has no retry instruction; tool returns the real SQLite rows."],
            "runs": rows}


# ---------------------------------------------------------------------------
# M5v2: loop guardrail with generic text, same system prompt for both modes
# ---------------------------------------------------------------------------
SCENARIOS = [
    {"id": "L1", "trap": "empty_date_filter", "target_fact": "PG_LOCK_TIMEOUT_55P03",
     "user": "Find the PostgreSQL error code for inventory-db. Start by checking date_prefix '2026-11-09'."},
    {"id": "L2", "trap": "empty_date_filter", "target_fact": "UPSTREAM_502_TIMEOUT",
     "user": "Find the upstream timeout error code for checkout-api. Start by checking date_prefix '2026-11-09'."},
    {"id": "L3", "trap": "empty_date_filter", "target_fact": "ES_SHARD_SLOW",
     "user": "Find the OpenSearch slow shard error code for search-api. Start by checking date_prefix '2026-11-09'."},
    {"id": "L4", "trap": "wrong_target", "target_fact": "rel-2026-114",
     "user": "Find the exact rollback_cmd for the migration causing lock contention. Start by checking deployments on 'payment-gateway'."},
    {"id": "L5", "trap": "wrong_target", "target_fact": "inventory-db",
     "user": "Find which database has DEGRADED_LOCK_CONTENTION status. Start by checking db_metrics on 'promo-cache'."},
    {"id": "L6", "trap": "wrong_target", "target_fact": "catalog-index",
     "user": "Find which index has WARNING_SHARD_REBALANCE status. Start by checking db_metrics on 'payment-gateway'."},
    {"id": "L7", "trap": "empty_service_logs", "target_fact": "99412",
     "user": "Find the deadlock transaction ID in error_logs. Start by checking error_logs on 'search-api' for date_prefix '2026-11-08'."},
    {"id": "L8", "trap": "wrong_target", "target_fact": "rel-2026-113",
     "user": "Find the release_id of the Envoy TLS cert update. Start by checking deployments on 'inventory-db'."},
]
M5_SYSTEM = ("You are an SRE agent. Use the tools to find the answer to the user's question. "
             "When you have it, reply with the answer in one or two sentences.")
M5_SYSTEM_V3 = ("You are an SRE agent. Use the tools to find the answer to the user's question. Keep using tools "
                "until you have found the answer. Only reply without a tool call when you have the answer, "
                "then give it in one or two sentences.")
GUARD_EMPTY = ("Guardrail: this call returned no matching rows. Do not repeat it. In one sentence, state why "
               "the arguments missed, then change the tool, target/service or date_prefix.")
GUARD_DUP = ("Guardrail: this exact tool call (same name and arguments) was already made, so it was blocked and not "
             "executed. In one sentence, state why it did not answer the question, then change the tool, "
             "target/service or date_prefix.")


def measure_m5v2(model, tokenizer, device, max_steps=6, system_prompt=None, continue_on_prose=False):
    system_prompt = system_prompt or M5_SYSTEM
    print("\n=== M5v2: generic action-hash guardrail vs none ===", flush=True)
    out = {}
    for mode in ("mode_0_no_guardrail", "mode_1_action_hash_guardrail"):
        cases = []
        for sc in SCENARIOS:
            msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": sc["user"]}]
            seen, steps, toks, ms, final = {}, [], 0, 0.0, ""
            fact_in_tools = False
            for s in range(1, max_steps + 1):
                turn = run_turn_v2(model, tokenizer, encode(tokenizer, msgs, device), max_new_tokens=300)
                toks += turn["prompt_tokens"] + turn["output_tokens"]
                ms += turn["total_ms"]
                _, body = split_think_and_body(turn["raw_output"])
                tcs, errs = parse_tool_calls(body)
                if not tcs:
                    final = body
                    if (continue_on_prose and sc["target_fact"].lower() not in body.lower() and s < max_steps):
                        # Real runtimes keep looping while the task is unfinished: append the prose turn and go again.
                        steps.append({"step": s, "status": "PROSE_CONTINUE", "hit_cap": turn["hit_cap"], "text": body[:300]})
                        msgs.append({"role": "assistant", "content": body})
                        continue
                    steps.append({"step": s, "status": "FINAL_ANSWER", "hit_cap": turn["hit_cap"]})
                    break
                name, args = tcs[0].get("name", ""), tcs[0].get("arguments", {})
                canon = json.dumps({"name": name, "arguments": args}, sort_keys=True, separators=(",", ":"))
                h = hashlib.sha256(canon.encode()).hexdigest()[:12]
                seen[h] = seen.get(h, 0) + 1
                dup = seen[h] > 1
                rec = {"step": s, "tool": name, "args": args, "action_hash": h, "duplicate": dup}
                msgs.append(asst_tool_msg(name, args))
                if mode == "mode_1_action_hash_guardrail" and dup:
                    rec["blocked"] = True
                    content = json.dumps({"status": "BLOCKED_DUPLICATE_ACTION", "action_hash": h,
                                          "attempt": seen[h], "guardrail": GUARD_DUP})
                else:
                    res, _ = exec_real_sqlite_tool(name, args)
                    empty = is_empty_result(res)
                    rec["empty"] = empty
                    if sc["target_fact"].lower() in json.dumps(res).lower():
                        fact_in_tools = True
                    if mode == "mode_1_action_hash_guardrail" and empty:
                        rec["guardrail_note"] = True
                        content = json.dumps({"result": res, "action_hash": h, "guardrail": GUARD_EMPTY})
                    else:
                        content = json.dumps(res)
                msgs.append({"role": "tool", "name": name, "content": content})
                steps.append(rec)
            solved = any(sc["target_fact"].lower() in x.lower() for x in [final] + [st.get("text", "") for st in steps])
            cases.append({"id": sc["id"], "trap": sc["trap"], "solved": solved,
                          "fact_seen_in_tool_results": fact_in_tools,
                          "llm_calls": len(steps), "duplicate_calls": sum(1 for x in steps if x.get("duplicate")),
                          "blocked_calls": sum(1 for x in steps if x.get("blocked")),
                          "total_tokens": toks, "gpu_ms": round(ms, 2),
                          "final_answer": final[:500], "steps": steps})
        n = len(cases)
        out[mode] = {
            "scenarios": n,
            "solved_count": sum(c["solved"] for c in cases),
            "recovery_rate_pct": round(100.0 * sum(c["solved"] for c in cases) / n, 1),
            "mean_llm_calls": round(sum(c["llm_calls"] for c in cases) / n, 2),
            "total_duplicate_calls": sum(c["duplicate_calls"] for c in cases),
            "total_blocked_calls": sum(c["blocked_calls"] for c in cases),
            "total_tokens": sum(c["total_tokens"] for c in cases),
            "total_gpu_ms": round(sum(c["gpu_ms"] for c in cases), 2),
            "cases": cases,
        }
        o = out[mode]
        print(f"  {mode}: solved={o['solved_count']}/{n} dups={o['total_duplicate_calls']} "
              f"blocked={o['total_blocked_calls']} calls={o['mean_llm_calls']} tok={o['total_tokens']} "
              f"ms={o['total_gpu_ms']}", flush=True)
        for c in cases:
            print(f"    {c['id']}: solved={c['solved']} calls={c['llm_calls']} dups={c['duplicate_calls']} "
                  f"seq={[ (x.get('tool'), x.get('args')) for x in c['steps'] if x.get('tool')]}", flush=True)
    out["system_prompt"] = system_prompt
    out["method_notes"] = [
        "Same system prompt in both modes; tools return the real SQLite rows.",
        "Guardrail text is generic (no per-scenario target, date or answer).",
        "Duplicate = same SHA-256 of canonical {name, arguments}; in mode 1 duplicates are blocked and not executed.",
        "Solved = the target fact appears in the model's final answer (a turn with no tool call). Greedy, thinking off, 300-token cap per call, max 6 calls.",
    ]
    return out


@torch.no_grad()
def main():
    import transformers
    device = "cuda"
    print(f"Loading {MODEL_ID} bf16...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto",
                                                 attn_implementation="sdpa")
    model.eval()
    env = {"model_id": MODEL_ID, "gpu_name": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "transformers": transformers.__version__, "dtype": "bfloat16", "attn": "sdpa", "decoding": "greedy"}
    print(env, flush=True)
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "m5v4":
        res = {"rerun_date": "2026-11-12", "gpu_environment": env,
               "m5v4_neutral_prompt_continue": measure_m5v2(model, tokenizer, device, continue_on_prose=True),
               "m5v4_persistence_prompt_continue": measure_m5v2(model, tokenizer, device, system_prompt=M5_SYSTEM_V3,
                                                                continue_on_prose=True)}
        blob = base64.b64encode(zlib.compress(json.dumps(res).encode(), 9)).decode()
        print("BEGIN_RERUN_B64", flush=True); print(blob, flush=True); print("END_RERUN_B64", flush=True)
        _stop_hb.set()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "m5v3":
        m5 = measure_m5v2(model, tokenizer, device, system_prompt=M5_SYSTEM_V3)
        res = {"rerun_date": "2026-11-12", "gpu_environment": env, "m5v3_persistence_prompt_guardrail": m5}
        blob = base64.b64encode(zlib.compress(json.dumps(res).encode(), 9)).decode()
        print("BEGIN_RERUN_B64", flush=True); print(blob, flush=True); print("END_RERUN_B64", flush=True)
        _stop_hb.set()
        return
    m3 = measure_m3v2(model, tokenizer, device)
    gc.collect(); torch.cuda.empty_cache()
    m4b = measure_m4b(model, tokenizer, device)
    gc.collect(); torch.cuda.empty_cache()
    m5 = measure_m5v2(model, tokenizer, device)
    results = {"rerun_date": "2026-11-12", "gpu_environment": env,
               "m3v2_react_vs_parsed_plan_and_execute": m3,
               "m4b_empty_date_trap_neutral_prompt": m4b,
               "m5v2_generic_action_hash_guardrail": m5}
    os.makedirs("data", exist_ok=True)
    with open("data/agent_loops_rerun_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    blob = base64.b64encode(zlib.compress(json.dumps(results).encode(), 9)).decode()
    print("BEGIN_RERUN_B64", flush=True)
    print(blob, flush=True)
    print("END_RERUN_B64", flush=True)
    _stop_hb.set()


if __name__ == "__main__":
    main()
