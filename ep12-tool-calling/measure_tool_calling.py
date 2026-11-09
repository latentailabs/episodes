"""End-to-End GPU Benchmark for Episode 12 (AI Agents Series Ep 2):
How LLM Tool Calling Actually Works (projects/37_tool_calling)

Runs 5 real PyTorch experiments on Qwen/Qwen3-8B (bfloat16) on an NVIDIA GPU:
  M1: The 2-Call Handshake Illusion (No Tools vs Call 1 <tool_call> -> Python -> Call 2 <tool_response>)
      + 5-Step Multi-Tool Incident Chain (Stateless Re-Prefill vs Prefix KV Cache Reuse)
  M2: Inside Token ID 151657 (<tool_call>) — 36-Layer Logit Lens Trajectory & Attention Routing
      across 6 Tool-Required Queries vs 6 Plain-Text Queries
  M3: Sequential Tool Calling (5 API Calls, 1 Tool/Turn) vs Parallel Tool Calling (2 API Calls, 4 Tools at Once)
  M4: The Tool-Bloat Cliff — Scaling Catalog Size K in {1, 5, 10, 25, 50} Tools (Prompt Tokens, TTFT, KV Cache MB)
      + Disjoint Typed Schemas vs Overlapping Vague Schemas Accuracy Benchmark
  M5: Unconstrained Tool Argument Errors (T=0.7) vs FSM Grammar-Constrained <tool_call> (-inf Logit Masking)
      + Runtime Tool Error Self-Correction Loop via <tool_response>
"""
import base64
import gc
import json
import math
import os
import re
import threading
import time
import zlib

_t0_hb = time.time()
_stop_hb = threading.Event()


def _hb():
    while not _stop_hb.wait(5.0):
        print(f"[hb] Ep37 Tool Calling GPU benchmark running... {int(time.time() - _t0_hb)}s", flush=True)


threading.Thread(target=_hb, daemon=True).start()

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from transformers.cache_utils import DynamicCache

MODEL_ID = "Qwen/Qwen3-8B"
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Simulated Production Telemetry Backend (Deterministic Python Tools)
# ---------------------------------------------------------------------------
SERVICE_DB = {
    "auth-api": {
        "service": "auth-api",
        "port": 5439,
        "active_connections": 412,
        "max_connections": 500,
        "replica_lag_ms": 1.8,
        "healthy": True,
    },
    "billing-api": {
        "service": "billing-api",
        "port": 5440,
        "active_connections": 489,
        "max_connections": 500,
        "replica_lag_ms": 142.5,
        "healthy": False,
    },
    "search-api": {
        "service": "search-api",
        "port": 5441,
        "active_connections": 128,
        "max_connections": 500,
        "replica_lag_ms": 0.9,
        "healthy": True,
    },
    "analytics-api": {
        "service": "analytics-api",
        "port": 5442,
        "active_connections": 64,
        "max_connections": 500,
        "replica_lag_ms": 3.4,
        "healthy": True,
    },
}

CORE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_postgres_status",
            "description": "Query the live production Postgres database port, active connections, replica lag, and health status for a microservice.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "enum": ["auth-api", "billing-api", "search-api", "analytics-api"],
                        "description": "Exact production microservice identifier.",
                    }
                },
                "required": ["service"],
            },
        },
    }
]


def execute_python_tool(name: str, arguments: dict) -> dict:
    if name == "query_postgres_status":
        svc = arguments.get("service", "")
        if svc in SERVICE_DB:
            return SERVICE_DB[svc]
        return {
            "error": f"KeyError: Unknown service '{svc}'. Valid service names are: {list(SERVICE_DB.keys())}"
        }
    return {"error": f"Unknown tool '{name}'"}


def parse_tool_calls(raw_output: str):
    """Extract all <tool_call>...</tool_call> JSON blocks emitted by Qwen3."""
    matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", raw_output, flags=re.S)
    parsed = []
    for m in matches:
        try:
            obj = json.loads(m)
            parsed.append(obj)
        except Exception as e:
            parsed.append({"_parse_error": str(e), "_raw": m})
    return parsed


def measure_m0_environment(model, tokenizer):
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0.0
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    weight_gb = round(param_bytes / (1024**3), 2)

    special_tokens = {}
    for tok_str in [
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
    ]:
        special_tokens[tok_str] = int(tokenizer.convert_tokens_to_ids(tok_str))

    env = {
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "pytorch_version": torch.__version__,
        "model_id": MODEL_ID,
        "precision": "bfloat16",
        "weight_gb": weight_gb,
        "vocab_size": int(model.config.vocab_size),
        "num_layers": int(model.config.num_hidden_layers),
        "hidden_size": int(model.config.hidden_size),
        "special_token_ids": special_tokens,
    }
    print(f"[M0] Environment & Special Tokens: {json.dumps(env, indent=2)}", flush=True)
    return env


def timed_prefill_and_decode(model, tokenizer, input_ids, max_new_tokens=64):
    """Run an exact timed prefill (TTFT) + greedy decode pass, returning output text and timings."""
    sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        out_prefill = model(input_ids=input_ids, use_cache=True)
    sync()
    prefill_ms = (time.perf_counter() - t0) * 1000.0

    past_kv = out_prefill.past_key_values
    next_tok = torch.argmax(out_prefill.logits[:, -1, :], dim=-1, keepdim=True)
    gen_ids = [int(next_tok.item())]

    eos_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}

    sync()
    t_dec0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            if gen_ids[-1] in eos_ids:
                break
            out_step = model(input_ids=next_tok, past_key_values=past_kv, use_cache=True)
            past_kv = out_step.past_key_values
            next_tok = torch.argmax(out_step.logits[:, -1, :], dim=-1, keepdim=True)
            gen_ids.append(int(next_tok.item()))
            if gen_ids[-1] in eos_ids:
                break
    sync()
    decode_ms = (time.perf_counter() - t_dec0) * 1000.0

    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    clean_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    del out_prefill, past_kv
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "prompt_tokens": int(input_ids.shape[1]),
        "generated_tokens": len(gen_ids),
        "generated_token_ids": gen_ids,
        "prefill_ms": round(prefill_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "total_ms": round(prefill_ms + decode_ms, 2),
        "raw_output": raw_text,
        "clean_output": clean_text,
    }


def measure_m1_two_call_handshake(model, tokenizer, device):
    """M1: Prove that 1 tool question is physically 2 separate LLM API calls + measure Prefix KV Cache across a 5-step tool chain."""
    print("\n[M1] Running 2-Call Handshake Proof & 5-Step Tool Chain Prefix KV Benchmark...", flush=True)

    user_query = "What port is our production auth-api Postgres database running on right now, and how many active connections does it have?"
    msgs_user_only = [{"role": "user", "content": user_query}]

    # Warmup pass
    warm_ids = tokenizer("Warmup GPU pass", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        _ = model.generate(warm_ids, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    sync()

    # Mode 0: Without tools registered
    prompt_no_tools = tokenizer.apply_chat_template(
        msgs_user_only, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids_no_tools = tokenizer(prompt_no_tools, return_tensors="pt").input_ids.to(device)
    res_no_tools = timed_prefill_and_decode(model, tokenizer, ids_no_tools, max_new_tokens=64)

    # Mode 1: Call 1 with CORE_TOOLS registered
    prompt_call1 = tokenizer.apply_chat_template(
        msgs_user_only, tools=CORE_TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids_call1 = tokenizer(prompt_call1, return_tensors="pt").input_ids.to(device)
    res_call1 = timed_prefill_and_decode(model, tokenizer, ids_call1, max_new_tokens=64)

    parsed_calls = parse_tool_calls(res_call1["raw_output"])
    tc = parsed_calls[0] if parsed_calls else {"name": "query_postgres_status", "arguments": {"service": "auth-api"}}

    # Local Python execution
    t_py0 = time.perf_counter()
    tool_result = execute_python_tool(tc["name"], tc.get("arguments", {}))
    py_exec_ms = round((time.perf_counter() - t_py0) * 1000.0, 4)

    # Call 2: Append assistant tool_calls + role="tool" response
    msgs_call2 = [
        {"role": "user", "content": user_query},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}],
        },
        {"role": "tool", "name": tc["name"], "content": json.dumps(tool_result)},
    ]
    prompt_call2 = tokenizer.apply_chat_template(
        msgs_call2, tools=CORE_TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids_call2 = tokenizer(prompt_call2, return_tensors="pt").input_ids.to(device)
    res_call2_stateless = timed_prefill_and_decode(model, tokenizer, ids_call2, max_new_tokens=64)

    # Inspect the exact suffix added in Call 2 after Call 1's prompt + generated tokens
    # Call 1 full sequence in ChatML is prompt_call1 + res_call1["raw_output"] + "\n"
    call1_full_prefix_str = prompt_call1 + res_call1["raw_output"].strip() + "\n"
    shared_prefix_ids = tokenizer(call1_full_prefix_str, return_tensors="pt").input_ids.to(device)
    shared_len = 0
    max_cmp = min(shared_prefix_ids.shape[1], ids_call2.shape[1])
    for i in range(max_cmp):
        if int(shared_prefix_ids[0, i]) == int(ids_call2[0, i]):
            shared_len += 1
        else:
            break

    # Measure Call 2 with Prefix KV Cache (pre-populating KV cache with shared_len tokens from Call 1)
    with torch.no_grad():
        prefix_out = model(input_ids=ids_call2[:, :shared_len], use_cache=True)
    cached_kv = prefix_out.past_key_values
    delta_ids = ids_call2[:, shared_len:]

    sync()
    t_c2_cache = time.perf_counter()
    with torch.no_grad():
        _ = model(input_ids=delta_ids, past_key_values=cached_kv, use_cache=True)
    sync()
    call2_cached_prefill_ms = round((time.perf_counter() - t_c2_cache) * 1000.0, 2)

    # Part B: 5-Step Multi-Tool Incident Investigation Chain (Stateless Re-Prefill vs Prefix KV Cache Reuse)
    chain_tools = [
        CORE_TOOLS[0],
        {
            "type": "function",
            "function": {
                "name": "get_recent_error_logs",
                "description": "Fetch the last N error log entries from a microservice.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["service"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_k8s_pod_restarts",
                "description": "Check Kubernetes pod restart count and OOMKilled status for a service.",
                "parameters": {
                    "type": "object",
                    "properties": {"service": {"type": "string"}},
                    "required": ["service"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "verify_pgbouncer_pool",
                "description": "Inspect PgBouncer connection pool saturation and queued clients.",
                "parameters": {
                    "type": "object",
                    "properties": {"service": {"type": "string"}},
                    "required": ["service"],
                },
            },
        },
    ]

    chain_history = [
        {
            "role": "user",
            "content": "Investigate why billing-api is timing out: check auth-api DB first, then billing-api DB, then billing-api error logs, then k8s restarts, and finally pgbouncer pool.",
        }
    ]
    scripted_steps = [
        ("query_postgres_status", {"service": "auth-api"}, SERVICE_DB["auth-api"]),
        ("query_postgres_status", {"service": "billing-api"}, SERVICE_DB["billing-api"]),
        (
            "get_recent_error_logs",
            {"service": "billing-api", "limit": 5},
            {"service": "billing-api", "errors": ["FATAL: remaining connection slots are reserved", "timeout after 5000ms"]},
        ),
        (
            "check_k8s_pod_restarts",
            {"service": "billing-api"},
            {"service": "billing-api", "restarts_last_1h": 7, "last_termination_reason": "OOMKilled"},
        ),
        (
            "verify_pgbouncer_pool",
            {"service": "billing-api"},
            {"service": "billing-api", "sv_active": 489, "cl_waiting": 142, "pool_mode": "session"},
        ),
    ]

    chain_rows = []
    prev_ids = None
    for step_idx in range(len(scripted_steps) + 1):
        prompt_str = tokenizer.apply_chat_template(
            chain_history, tools=chain_tools, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        cur_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)
        total_toks = int(cur_ids.shape[1])

        # Stateless full prefill timing (median of 3 runs)
        stateless_times = []
        for _ in range(3):
            sync()
            t_s = time.perf_counter()
            with torch.no_grad():
                _ = model(input_ids=cur_ids, use_cache=True)
            sync()
            stateless_times.append((time.perf_counter() - t_s) * 1000.0)
        stateless_ms = round(sorted(stateless_times)[1], 2)

        # Prefix KV Cache reuse from previous API call in the chain
        if prev_ids is None:
            cached_toks = 0
            delta_toks = total_toks
            cached_ms = stateless_ms
        else:
            common = 0
            limit = min(prev_ids.shape[1], cur_ids.shape[1])
            for k in range(limit):
                if int(prev_ids[0, k]) == int(cur_ids[0, k]):
                    common += 1
                else:
                    break
            cached_toks = common
            delta_toks = total_toks - common
            cached_times = []
            for _ in range(3):
                with torch.no_grad():
                    pkv = model(input_ids=cur_ids[:, :common], use_cache=True).past_key_values
                sync()
                t_c = time.perf_counter()
                with torch.no_grad():
                    _ = model(input_ids=cur_ids[:, common:], past_key_values=pkv, use_cache=True)
                sync()
                cached_times.append((time.perf_counter() - t_c) * 1000.0)
            cached_ms = round(sorted(cached_times)[1], 2)

        chain_rows.append({
            "api_call_num": step_idx + 1,
            "stage": f"Tool Call {step_idx + 1}" if step_idx < len(scripted_steps) else "Final Synthesis Call",
            "total_prompt_tokens": total_toks,
            "cached_prefix_tokens": cached_toks,
            "new_delta_tokens": delta_toks,
            "stateless_prefill_ms": stateless_ms,
            "prefix_cached_prefill_ms": cached_ms,
            "speedup_x": round(stateless_ms / max(cached_ms, 0.01), 2),
        })
        prev_ids = cur_ids

        if step_idx < len(scripted_steps):
            fn_name, fn_args, fn_out = scripted_steps[step_idx]
            chain_history.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "function": {"name": fn_name, "arguments": fn_args}}],
            })
            chain_history.append({
                "role": "tool",
                "name": fn_name,
                "content": json.dumps(fn_out),
            })

    sum_stateless_toks = sum(r["total_prompt_tokens"] for r in chain_rows)
    sum_cached_delta_toks = sum(r["new_delta_tokens"] for r in chain_rows)
    sum_stateless_ms = round(sum(r["stateless_prefill_ms"] for r in chain_rows), 2)
    sum_cached_ms = round(sum(r["prefix_cached_prefill_ms"] for r in chain_rows), 2)

    print(
        f"  [M1] Call 1 ({res_call1['prompt_tokens']} -> {res_call1['generated_tokens']} toks, {res_call1['total_ms']} ms) "
        f"| Call 2 ({res_call2_stateless['prompt_tokens']} -> {res_call2_stateless['generated_tokens']} toks, {res_call2_stateless['total_ms']} ms)",
        flush=True,
    )
    print(
        f"  [M1] 6-Call Chain: Stateless prefilled {sum_stateless_toks} toks ({sum_stateless_ms} ms) vs Prefix Cache {sum_cached_delta_toks} delta toks ({sum_cached_ms} ms)",
        flush=True,
    )

    return {
        "user_query": user_query,
        "user_query_raw_tokens": len(tokenizer.encode(user_query)),
        "mode0_no_tools": res_no_tools,
        "mode1_call1_emit_tool_call": res_call1,
        "mode1_parsed_tool_call": tc,
        "mode1_python_exec_ms": py_exec_ms,
        "mode1_tool_output": tool_result,
        "mode1_call2_compiled_prompt": prompt_call2,
        "mode1_call2_stateless": res_call2_stateless,
        "mode1_call2_shared_prefix_tokens": shared_len,
        "mode1_call2_new_delta_tokens": int(ids_call2.shape[1]) - shared_len,
        "mode1_call2_prefix_cached_prefill_ms": call2_cached_prefill_ms,
        "mode1_two_call_total_ms": round(res_call1["total_ms"] + py_exec_ms + res_call2_stateless["total_ms"], 2),
        "multi_step_tool_chain": {
            "calls": chain_rows,
            "cumulative_stateless_prefill_tokens": sum_stateless_toks,
            "cumulative_prefix_cache_delta_tokens": sum_cached_delta_toks,
            "token_waste_ratio_x": round(sum_stateless_toks / max(sum_cached_delta_toks, 1), 2),
            "cumulative_stateless_prefill_ms": sum_stateless_ms,
            "cumulative_prefix_cached_prefill_ms": sum_cached_ms,
            "prefill_speedup_x": round(sum_stateless_ms / max(sum_cached_ms, 0.01), 2),
        },
    }


def measure_m2_token_151657_logit_lens_and_attention(model, tokenizer, device):
    """M2: 36-Layer Logit Lens and Attention Routing for Token ID 151657 (<tool_call>) vs Plain-Text Answers."""
    print("\n[M2] Running 36-Layer Logit Lens & Attention Routing on Token ID 151657 (<tool_call>)...", flush=True)

    tool_call_id = tokenizer.convert_tokens_to_ids("<tool_call>")

    tool_queries = [
        "What port is our production auth-api Postgres database running on right now?",
        "How many active connections does billing-api currently have on Postgres?",
        "Is the search-api Postgres database healthy right now and what is its replica lag?",
        "Check the current Postgres port and connection count for analytics-api.",
        "Query the live database health status of billing-api.",
        "What is the current replica lag in milliseconds on auth-api?",
    ]
    plain_queries = [
        "What is the standard default TCP port for PostgreSQL in official documentation?",
        "What does ACID stand for in relational database systems?",
        "Explain the difference between a primary key and a foreign key in one sentence.",
        "What HTTP status code means Internal Server Error?",
        "How many bits are in an IPv4 address?",
        "What SQL keyword is used to filter rows after a GROUP BY clause?",
    ]

    decision_eval = []
    for label, q_list, expect_tool in [("tool_required", tool_queries, True), ("plain_text", plain_queries, False)]:
        correct = 0
        samples = []
        for q in q_list:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                tools=CORE_TOOLS,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=40, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            gen_ids = out[0, ids.shape[1]:].tolist()
            raw_out = tokenizer.decode(gen_ids, skip_special_tokens=False)
            triggered_tool = (len(gen_ids) > 0 and gen_ids[0] == tool_call_id) or ("<tool_call>" in raw_out)
            is_ok = (triggered_tool == expect_tool)
            if is_ok:
                correct += 1
            samples.append({
                "query": q,
                "first_generated_token_id": int(gen_ids[0]) if gen_ids else -1,
                "first_generated_token_str": tokenizer.decode([gen_ids[0]]) if gen_ids else "",
                "triggered_tool": bool(triggered_tool),
                "correct_decision": bool(is_ok),
                "raw_output_preview": raw_out[:120],
            })
        decision_eval.append({
            "category": label,
            "total": len(q_list),
            "correct": correct,
            "accuracy_pct": round(100.0 * correct / len(q_list), 1),
            "samples": samples,
        })

    # Detailed 36-Layer Logit Lens & Attention Routing on Pair 0:
    # Tool prompt vs Plain prompt
    pair_traces = {}
    for mode_key, query_text in [
        ("tool_trigger_prompt", tool_queries[0]),
        ("plain_answer_prompt", plain_queries[0]),
    ]:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": query_text}],
            tools=CORE_TOOLS,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        seq_ids = ids[0].tolist()
        pieces = [tokenizer.decode([tid]) for tid in seq_ids]

        # Partition token positions into 3 semantic regions:
        # 1) control_tokens (<|im_start|>, <|im_end|>, <think>, </think>, role names)
        # 2) tools_schema_region (everything inside the system # Tools block)
        # 3) user_query_region (the user's question tokens)
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        think_id = tokenizer.convert_tokens_to_ids("<think>")
        end_think_id = tokenizer.convert_tokens_to_ids("</think>")

        # Find where the user turn starts (second occurrence of <|im_start|>)
        im_starts = [i for i, tid in enumerate(seq_ids) if tid == im_start_id]
        user_turn_start = im_starts[1] if len(im_starts) > 1 else 0
        assistant_turn_start = im_starts[2] if len(im_starts) > 2 else len(seq_ids) - 1

        control_positions = set()
        schema_positions = set()
        user_query_positions = set()

        for pos, tid in enumerate(seq_ids):
            if tid in {im_start_id, im_end_id, think_id, end_think_id} or pos in {
                0, 1, 2, user_turn_start, user_turn_start + 1, user_turn_start + 2
            } or pos >= assistant_turn_start:
                control_positions.add(pos)
            elif pos < user_turn_start:
                schema_positions.add(pos)
            else:
                user_query_positions.add(pos)

        with torch.no_grad():
            out = model(
                input_ids=ids,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=False,
            )

        num_layers = len(out.attentions)
        logit_lens_layers = []
        attention_layers = []

        for l_idx in range(num_layers):
            # Hidden state after layer l_idx is out.hidden_states[l_idx + 1]
            h = out.hidden_states[l_idx + 1][:, -1, :]
            # Apply final RMSNorm + lm_head for Logit Lens (final layer already normed in out.logits)
            if l_idx == num_layers - 1:
                logits_l = out.logits[0, -1, :].float()
            else:
                normed_h = model.model.norm(h)
                logits_l = model.lm_head(normed_h)[0].float()

            probs_l = torch.softmax(logits_l, dim=-1)
            tc_prob = float(probs_l[tool_call_id].item()) * 100.0
            tc_logit = float(logits_l[tool_call_id].item())
            tc_rank = int((logits_l > logits_l[tool_call_id]).sum().item()) + 1

            top_vals, top_idxs = torch.topk(probs_l, k=3)
            top1_id = int(top_idxs[0].item())
            top1_piece = tokenizer.decode([top1_id])
            top1_prob = float(top_vals[0].item()) * 100.0

            logit_lens_layers.append({
                "layer": l_idx,
                "tool_call_151657_prob_pct": round(tc_prob, 2),
                "tool_call_151657_logit": round(tc_logit, 2),
                "tool_call_151657_rank": tc_rank,
                "top1_token_id": top1_id,
                "top1_token_str": top1_piece,
                "top1_prob_pct": round(top1_prob, 2),
            })

            # Attention routing from final position (-1) across heads in layer l_idx
            attn_l = out.attentions[l_idx][0, :, -1, :].float().mean(dim=0)  # [seq_len]
            ctrl_mass = sum(float(attn_l[p].item()) for p in control_positions) * 100.0
            schema_mass = sum(float(attn_l[p].item()) for p in schema_positions) * 100.0
            user_mass = sum(float(attn_l[p].item()) for p in user_query_positions) * 100.0

            attention_layers.append({
                "layer": l_idx,
                "control_tokens_attn_pct": round(ctrl_mass, 2),
                "tools_schema_attn_pct": round(schema_mass, 2),
                "user_query_attn_pct": round(user_mass, 2),
            })

        mean_ctrl = round(sum(r["control_tokens_attn_pct"] for r in attention_layers) / num_layers, 2)
        mean_schema = round(sum(r["tools_schema_attn_pct"] for r in attention_layers) / num_layers, 2)
        mean_user = round(sum(r["user_query_attn_pct"] for r in attention_layers) / num_layers, 2)

        pair_traces[mode_key] = {
            "query": query_text,
            "prompt_tokens": len(seq_ids),
            "region_token_counts": {
                "control_tokens": len(control_positions),
                "tools_schema_tokens": len(schema_positions),
                "user_query_tokens": len(user_query_positions),
            },
            "mean_attention_routing_pct": {
                "control_tokens": mean_ctrl,
                "tools_schema": mean_schema,
                "user_query": mean_user,
            },
            "logit_lens_by_layer": logit_lens_layers,
            "attention_by_layer": attention_layers,
        }
        print(
            f"  [M2 {mode_key}] Final Layer <tool_call> prob={logit_lens_layers[-1]['tool_call_151657_prob_pct']}% "
            f"(rank #{logit_lens_layers[-1]['tool_call_151657_rank']}), top1={logit_lens_layers[-1]['top1_token_str']!r} "
            f"| Mean Attn -> Schema={mean_schema}%, User={mean_user}%, Control={mean_ctrl}%",
            flush=True,
        )

    return {
        "decision_accuracy_eval": decision_eval,
        "traces": pair_traces,
    }


def measure_m3_sequential_vs_parallel(model, tokenizer, device):
    """M3: Compare Sequential 1-Tool-Per-Turn Loop (5 API Calls) vs Parallel Multi-Tool Calling (2 API Calls)."""
    print("\n[M3] Running Sequential (5 API Calls) vs Parallel (2 API Calls) Multi-Tool Benchmark...", flush=True)

    services = ["auth-api", "billing-api", "search-api", "analytics-api"]

    # Mode A: Sequential 1-Tool-Per-Turn Loop (4 tool calls one per turn + 1 final synthesis call = 5 API calls)
    seq_messages = [
        {
            "role": "system",
            "content": "You are a production SRE assistant. Call only ONE tool per turn, waiting for its result before checking the next service.",
        },
        {
            "role": "user",
            "content": "Check the live Postgres database status one by one for auth-api, billing-api, search-api, and analytics-api, then report which service is unhealthy.",
        },
    ]

    seq_call_logs = []
    for step_idx in range(5):
        prompt_str = tokenizer.apply_chat_template(
            seq_messages, tools=CORE_TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)
        res = timed_prefill_and_decode(model, tokenizer, ids, max_new_tokens=96)

        parsed = parse_tool_calls(res["raw_output"])
        # If the model emitted multiple tool calls despite instructions, take the first one for strict sequential turn-by-turn comparison
        if step_idx < 4:
            svc = services[step_idx]
            chosen_call = parsed[0] if parsed else {"name": "query_postgres_status", "arguments": {"service": svc}}
            chosen_call["arguments"] = {"service": svc}
            tool_out = execute_python_tool("query_postgres_status", {"service": svc})
            seq_messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "function": chosen_call}],
            })
            seq_messages.append({
                "role": "tool",
                "name": "query_postgres_status",
                "content": json.dumps(tool_out),
            })
            seq_call_logs.append({
                "call_num": step_idx + 1,
                "role": f"Sequential Tool Call #{step_idx + 1} ({svc})",
                "prompt_tokens": res["prompt_tokens"],
                "generated_tokens": res["generated_tokens"],
                "prefill_ms": res["prefill_ms"],
                "decode_ms": res["decode_ms"],
                "total_ms": res["total_ms"],
                "raw_output": res["raw_output"],
            })
        else:
            seq_call_logs.append({
                "call_num": 5,
                "role": "Sequential Final Synthesis Call",
                "prompt_tokens": res["prompt_tokens"],
                "generated_tokens": res["generated_tokens"],
                "prefill_ms": res["prefill_ms"],
                "decode_ms": res["decode_ms"],
                "total_ms": res["total_ms"],
                "raw_output": res["raw_output"],
                "clean_output": res["clean_output"],
            })

    # Mode B: Native Parallel Tool Calling (Call 1 emits all 4 <tool_call> blocks at once; Call 2 synthesizes final answer = 2 API calls)
    par_messages = [
        {
            "role": "user",
            "content": "Check the live Postgres database status for all four services (auth-api, billing-api, search-api, and analytics-api) in parallel right now, then report which service is unhealthy.",
        }
    ]
    par_prompt1 = tokenizer.apply_chat_template(
        par_messages, tools=CORE_TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    par_ids1 = tokenizer(par_prompt1, return_tensors="pt").input_ids.to(device)
    par_res1 = timed_prefill_and_decode(model, tokenizer, par_ids1, max_new_tokens=140)
    par_calls = parse_tool_calls(par_res1["raw_output"])

    # Execute all emitted tool calls in parallel and append to par_messages
    par_messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
            for c in par_calls
        ],
    })
    for c in par_calls:
        out_obj = execute_python_tool(c["name"], c.get("arguments", {}))
        par_messages.append({
            "role": "tool",
            "name": c["name"],
            "content": json.dumps(out_obj),
        })

    par_prompt2 = tokenizer.apply_chat_template(
        par_messages, tools=CORE_TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    par_ids2 = tokenizer(par_prompt2, return_tensors="pt").input_ids.to(device)
    par_res2 = timed_prefill_and_decode(model, tokenizer, par_ids2, max_new_tokens=96)

    seq_total_ms = round(sum(c["total_ms"] for c in seq_call_logs), 2)
    seq_prefill_toks = sum(c["prompt_tokens"] for c in seq_call_logs)
    seq_decode_toks = sum(c["generated_tokens"] for c in seq_call_logs)

    par_total_ms = round(par_res1["total_ms"] + par_res2["total_ms"], 2)
    par_prefill_toks = par_res1["prompt_tokens"] + par_res2["prompt_tokens"]
    par_decode_toks = par_res1["generated_tokens"] + par_res2["generated_tokens"]

    print(
        f"  [M3] Sequential (5 API calls): {seq_total_ms} ms ({seq_prefill_toks} prefill toks, {seq_decode_toks} decode toks) "
        f"| Parallel (2 API calls, {len(par_calls)} tools emitted in Call 1): {par_total_ms} ms "
        f"({par_prefill_toks} prefill toks, {par_decode_toks} decode toks) -> {round(seq_total_ms / max(par_total_ms, 0.01), 2)}x faster",
        flush=True,
    )

    return {
        "sequential_5_calls": {
            "api_calls_count": 5,
            "total_prefill_tokens": seq_prefill_toks,
            "total_decode_tokens": seq_decode_toks,
            "total_latency_ms": seq_total_ms,
            "calls": seq_call_logs,
        },
        "parallel_2_calls": {
            "api_calls_count": 2,
            "tools_emitted_in_call1": len(par_calls),
            "parsed_parallel_tool_calls": par_calls,
            "total_prefill_tokens": par_prefill_toks,
            "total_decode_tokens": par_decode_toks,
            "total_latency_ms": par_total_ms,
            "call1": par_res1,
            "call2": par_res2,
        },
        "latency_speedup_x": round(seq_total_ms / max(par_total_ms, 0.01), 2),
        "prefill_token_reduction_x": round(seq_prefill_toks / max(par_prefill_toks, 1), 2),
    }


def build_synthetic_tool_catalog(num_tools: int):
    """Build a realistic catalog of `num_tools` DevOps/Cloud tools for M4 scaling & schema bloat tests."""
    catalog = list(CORE_TOOLS)
    domains = [
        ("redis_cache_stats", "Inspect Redis cluster memory fragmentation, hit rate, and evicted keys."),
        ("kafka_consumer_lag", "Fetch Apache Kafka consumer group offset lag by topic and partition."),
        ("elasticsearch_shard_health", "Check Elasticsearch cluster status, unassigned shards, and JVM heap."),
        ("s3_bucket_Replication_metrics", "Get AWS S3 cross-region replication pending bytes and operation latency."),
        ("cloudflare_waf_drops", "Query Cloudflare WAF blocked request counts and top firewall rule IDs."),
        ("prometheus_alert_rules", "List currently firing Prometheus alert rules by severity level."),
        ("jaeger_trace_latency", "Retrieve p95 and p99 distributed trace latency for a microservice."),
        ("vault_lease_ttl", "Check HashiCorp Vault dynamic secret lease TTL and renewal status."),
        ("celery_queue_depth", "Measure Celery worker queue backlog and active task count."),
        ("nginx_upstream_5xx", "Count NGINX upstream 502/504 gateway errors over the last window."),
    ]
    idx = 1
    while len(catalog) < num_tools:
        base_name, base_desc = domains[(idx - 1) % len(domains)]
        suffix = f"_v{(idx - 1) // len(domains) + 1}" if idx > len(domains) else ""
        fn_name = f"{base_name}{suffix}"
        catalog.append({
            "type": "function",
            "function": {
                "name": fn_name,
                "description": f"{base_desc} (Module #{idx})",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service": {
                            "type": "string",
                            "enum": ["auth-api", "billing-api", "search-api", "analytics-api"],
                            "description": "Target microservice name.",
                        },
                        "environment": {
                            "type": "string",
                            "enum": ["prod-us-east", "prod-eu-west", "staging"],
                            "description": "Deployment region identifier.",
                        },
                        "window_minutes": {
                            "type": "integer",
                            "description": "Lookback telemetry window in minutes (5 to 1440).",
                        },
                    },
                    "required": ["service", "environment"],
                },
            },
        })
        idx += 1
    return catalog[:num_tools]


def measure_m4_tool_bloat_and_schema_quality(model, tokenizer, device):
    """M4: Sweep tool count K in {1, 5, 10, 25, 50} and compare Disjoint Typed Schemas vs Vague Overlapping Schemas."""
    print("\n[M4] Running Tool-Bloat Scaling (1..50 Tools) & Schema Disambiguation Benchmark...", flush=True)

    user_msg = [{"role": "user", "content": "What port is our production auth-api Postgres database running on right now?"}]
    kv_bytes_per_token = (
        2
        * model.config.num_hidden_layers
        * model.config.num_key_value_heads
        * (model.config.hidden_size // model.config.num_attention_heads)
        * 2  # bfloat16 = 2 bytes
    )

    scaling_rows = []
    for k in [1, 5, 10, 20, 30, 50]:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tools_k = build_synthetic_tool_catalog(k)
        prompt_k = tokenizer.apply_chat_template(
            user_msg, tools=tools_k, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids_k = tokenizer(prompt_k, return_tensors="pt").input_ids.to(device)
        n_toks = int(ids_k.shape[1])

        # For K=50 (~8,600 tokens), eager fp32 [1, 32, N, N] softmax requires 8.87 GiB per layer and OOMs 40GB A100;
        # switch to SDPA if eager OOMs so we record both the eager OOM cliff and the SDPA measurement.
        attn_mode_used = "eager"
        ttfts = []
        try:
            for _ in range(3):
                sync()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out_tmp = model(input_ids=ids_k, use_cache=True, logits_to_keep=1)
                sync()
                ttfts.append((time.perf_counter() - t0) * 1000.0)
                del out_tmp
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            attn_mode_used = "sdpa_fallback_after_eager_oom"
            model.config._attn_implementation = "sdpa"
            ttfts = []
            for _ in range(3):
                sync()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out_tmp = model(input_ids=ids_k, use_cache=True, logits_to_keep=1)
                sync()
                ttfts.append((time.perf_counter() - t0) * 1000.0)
                del out_tmp
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        med_ttft = round(sorted(ttfts)[len(ttfts) // 2], 2)
        kv_mb = round((n_toks * kv_bytes_per_token) / (1024**2), 2)

        # Verify if the model still picks query_postgres_status accurately
        with torch.no_grad():
            out_k = model.generate(ids_k, max_new_tokens=40, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        raw_k = tokenizer.decode(out_k[0, n_toks:], skip_special_tokens=False)
        del out_k
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.config._attn_implementation = "eager"

        parsed_k = parse_tool_calls(raw_k)
        picked = parsed_k[0].get("name", "") if parsed_k else "NONE"

        scaling_rows.append({
            "num_tools": k,
            "prompt_tokens": n_toks,
            "tools_schema_overhead_tokens": n_toks - 25,
            "prefill_ttft_ms": med_ttft,
            "attn_mode": attn_mode_used,
            "kv_cache_mb": kv_mb,
            "selected_tool": picked,
            "correct_tool_selected": picked == "query_postgres_status",
        })
        print(
            f"  [M4 Scaling K={k:2d} tools] {n_toks:4d} tokens | TTFT={med_ttft:6.2f} ms ({attn_mode_used}) | KV={kv_mb:5.2f} MB | picked={picked}",
            flush=True,
        )

    # Part B: Disjoint Well-Typed Schemas vs Overlapping Vague Schemas across 8 SRE Queries
    clean_tools = [
        {
            "type": "function",
            "function": {
                "name": "query_postgres_status",
                "description": "Use ONLY for relational PostgreSQL database port, active SQL connections, and replica lag.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string", "enum": ["auth-api", "billing-api", "search-api", "analytics-api"]}
                    },
                    "required": ["service"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_redis_cache",
                "description": "Use ONLY for in-memory Redis cache hit ratio, evicted keys, and memory fragmentation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string", "enum": ["auth-api", "billing-api", "search-api", "analytics-api"]}
                    },
                    "required": ["service"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_k8s_pod_health",
                "description": "Use ONLY for Kubernetes container restarts, OOMKilled events, and CPU throttling.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string", "enum": ["auth-api", "billing-api", "search-api", "analytics-api"]}
                    },
                    "required": ["service"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_kafka_lag",
                "description": "Use ONLY for Apache Kafka event stream consumer lag and partition offsets.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string", "enum": ["auth-api", "billing-api", "search-api", "analytics-api"]}
                    },
                    "required": ["service"],
                },
            },
        },
    ]

    # Vague overlapping tools (a real anti-pattern where multiple functions have generic names/descriptions and untyped args)
    vague_tools = [
        {
            "type": "function",
            "function": {
                "name": "check_service_status",
                "description": "Check status and metrics for a backend service.",
                "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_backend_info",
                "description": "Get live backend health, connections, or data store info for a service.",
                "parameters": {"type": "object", "properties": {"svc": {"type": "string"}}, "required": ["svc"]},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_system_metrics",
                "description": "Fetch system telemetry, database, cache, or pod status for a service.",
                "parameters": {"type": "object", "properties": {"target": {"type": "string"}}, "required": ["target"]},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_postgres_status",
                "description": "Query service data.",
                "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]},
            },
        },
    ]

    eval_cases = [
        ("What port is our auth-api Postgres database running on?", "query_postgres_status", "auth-api"),
        ("Check the Postgres replica lag and active SQL connections for billing-api.", "query_postgres_status", "billing-api"),
        ("What is the Redis cache hit ratio and evicted key count for search-api?", "query_redis_cache", "search-api"),
        ("How many times has the billing-api Kubernetes pod restarted due to OOMKilled?", "query_k8s_pod_health", "billing-api"),
        ("What is the Kafka consumer offset lag for analytics-api?", "query_kafka_lag", "analytics-api"),
        ("Is our search-api PostgreSQL instance healthy and what port is it on?", "query_postgres_status", "search-api"),
        ("Check Redis memory fragmentation for auth-api.", "query_redis_cache", "auth-api"),
        ("Check Kubernetes container restarts for analytics-api.", "query_k8s_pod_health", "analytics-api"),
    ]

    schema_comparison = {}
    for mode_name, tool_set in [("disjoint_typed_schemas", clean_tools), ("overlapping_vague_schemas", vague_tools)]:
        exact_matches = 0
        rows = []
        for q_text, expected_fn, expected_svc in eval_cases:
            p_str = tokenizer.apply_chat_template(
                [{"role": "user", "content": q_text}],
                tools=tool_set,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            q_ids = tokenizer(p_str, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                q_out = model.generate(q_ids, max_new_tokens=45, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            raw_gen = tokenizer.decode(q_out[0, q_ids.shape[1]:], skip_special_tokens=False)
            p_calls = parse_tool_calls(raw_gen)
            got_fn = p_calls[0].get("name", "") if p_calls else "NONE"
            got_args = p_calls[0].get("arguments", {}) if p_calls else {}
            got_svc = got_args.get("service", "")

            is_exact = (got_fn == expected_fn) and (got_svc == expected_svc)
            if is_exact:
                exact_matches += 1
            rows.append({
                "query": q_text,
                "expected_tool": expected_fn,
                "expected_service": expected_svc,
                "got_tool": got_fn,
                "got_arguments": got_args,
                "exact_match": is_exact,
            })
        acc = round(100.0 * exact_matches / len(eval_cases), 1)
        schema_comparison[mode_name] = {
            "trials": len(eval_cases),
            "exact_matches": exact_matches,
            "accuracy_pct": acc,
            "details": rows,
        }
        print(f"  [M4 {mode_name}] Exact Tool + Arg Match: {exact_matches}/{len(eval_cases)} ({acc}%)", flush=True)

    return {
        "catalog_scaling": scaling_rows,
        "schema_quality_comparison": schema_comparison,
    }


class PrefixConstraintLogitsProcessor(LogitsProcessor):
    """FSM-style prefix constraint processor that masks any token not continuing a valid <tool_call> string to -inf."""

    def __init__(self, prompt_len: int, id_to_piece: dict, valid_completions: list, candidate_items: list, step_trace_sink: list = None):
        self.prompt_len = prompt_len
        self.id_to_piece = id_to_piece
        self.valid_completions = valid_completions
        self.candidate_items = candidate_items
        self.step_trace_sink = step_trace_sink

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        gen_ids = input_ids[0, self.prompt_len:].tolist()
        prefix = "".join(self.id_to_piece.get(int(i), "") for i in gen_ids)

        mask = torch.full_like(scores, float("-inf"))
        allowed_ids = []
        for tok_id, piece in self.candidate_items:
            cand = prefix + piece
            if any(target.startswith(cand) or cand.startswith(target) for target in self.valid_completions):
                mask[0, tok_id] = 0.0
                allowed_ids.append(tok_id)

        if not allowed_ids:
            return scores

        masked_scores = scores + mask
        if self.step_trace_sink is not None and len(self.step_trace_sink) < 28:
            raw_probs = torch.softmax(scores[0].float(), dim=-1)
            post_probs = torch.softmax(masked_scores[0].float(), dim=-1)
            top_raw_v, top_raw_i = torch.topk(scores[0].float(), k=5)
            top_post_v, top_post_i = torch.topk(masked_scores[0].float(), k=min(5, len(allowed_ids)))
            self.step_trace_sink.append({
                "step": len(gen_ids),
                "prefix_so_far": prefix,
                "allowed_vocab_count": len(allowed_ids),
                "masked_to_neg_inf_count": int(scores.shape[-1]) - len(allowed_ids),
                "top5_raw_before_mask": [
                    {
                        "token_id": int(idx.item()),
                        "token_str": self.id_to_piece.get(int(idx.item()), ""),
                        "raw_logit": round(float(val.item()), 2),
                        "raw_prob_pct": round(float(raw_probs[idx].item()) * 100.0, 2),
                    }
                    for val, idx in zip(top_raw_v, top_raw_i)
                ],
                "top_after_mask": [
                    {
                        "token_id": int(idx.item()),
                        "token_str": self.id_to_piece.get(int(idx.item()), ""),
                        "post_logit": round(float(val.item()), 2),
                        "post_prob_pct": round(float(post_probs[idx].item()) * 100.0, 2),
                    }
                    for val, idx in zip(top_post_v, top_post_i)
                    if not math.isinf(float(val.item()))
                ],
            })
        return masked_scores


def measure_m5_constrained_tools_and_self_correction(model, tokenizer, device):
    """M5: Unconstrained vs Grammar-Constrained Tool Calls + <tool_response> Error Self-Correction Loop."""
    print("\n[M5] Running Constrained Tool Decoding & Error Self-Correction Benchmark...", flush=True)

    # Build token piece lookup table for fast prefix checking
    vocab_size = model.config.vocab_size
    id_to_piece = {}
    for tid in range(tokenizer.vocab_size):
        id_to_piece[tid] = tokenizer.decode([tid])
    for tid in [151644, 151645, 151657, 151658, 151665, 151666, 151667, 151668]:
        id_to_piece[tid] = tokenizer.decode([tid])

    # Part A: Ambiguous user phrasings where unconstrained tool calls often hallucinate service names
    # e.g. user says "auth backend" or "billing microservice" or "search engine" while enum requires exact slug
    ambiguous_prompts = [
        ("Check the Postgres database status for our authentication service right now.", "auth-api"),
        ("Query the Postgres database for the billing and payments service.", "billing-api"),
        ("Check the live Postgres database status on our search service.", "search-api"),
        ("Inspect Postgres database connections on the analytics service.", "analytics-api"),
    ]

    # Compare 1) Loose schema without enum (Unconstrained) vs 2) Strict FSM Grammar Mask (-inf)
    loose_tool = [
        {
            "type": "function",
            "function": {
                "name": "query_postgres_status",
                "description": "Query the live production Postgres database status for a microservice.",
                "parameters": {
                    "type": "object",
                    "properties": {"service": {"type": "string", "description": "Service name"}},
                    "required": ["service"],
                },
            },
        }
    ]

    valid_completions = [
        f'<tool_call>\n{{"name": "query_postgres_status", "arguments": {{"service": "{svc}"}}}}\n</tool_call><|im_end|>'
        for svc in ["auth-api", "billing-api", "search-api", "analytics-api"]
    ]
    candidate_items = [
        (tid, piece)
        for tid, piece in id_to_piece.items()
        if piece and any((piece in target) or (target in piece) for target in valid_completions)
    ]

    unconstrained_results = []
    constrained_results = []
    captured_fsm_trace = []

    # Run 20 trials (5 per prompt at T=0.7) for Unconstrained Loose Schema vs Constrained FSM
    for p_idx, (q_text, target_svc) in enumerate(ambiguous_prompts):
        p_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": q_text}],
            tools=loose_tool,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        q_ids = tokenizer(p_str, return_tensors="pt").input_ids.to(device)

        for trial in range(5):
            torch.manual_seed(1000 + p_idx * 10 + trial)
            with torch.no_grad():
                u_out = model.generate(
                    q_ids,
                    max_new_tokens=45,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                )
            u_raw = tokenizer.decode(u_out[0, q_ids.shape[1]:], skip_special_tokens=False)
            u_calls = parse_tool_calls(u_raw)
            u_svc = u_calls[0].get("arguments", {}).get("service", "") if u_calls else "PARSE_ERROR"
            u_valid = u_svc == target_svc
            unconstrained_results.append({
                "query": q_text,
                "expected_service": target_svc,
                "got_service": u_svc,
                "valid": u_valid,
                "raw_output": u_raw,
            })

            # Constrained FSM run
            trace_sink = captured_fsm_trace if (p_idx == 0 and trial == 0) else None
            proc = PrefixConstraintLogitsProcessor(
                prompt_len=int(q_ids.shape[1]),
                id_to_piece=id_to_piece,
                valid_completions=valid_completions,
                candidate_items=candidate_items,
                step_trace_sink=trace_sink,
            )
            torch.manual_seed(1000 + p_idx * 10 + trial)
            with torch.no_grad():
                c_out = model.generate(
                    q_ids,
                    max_new_tokens=45,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    logits_processor=LogitsProcessorList([proc]),
                    pad_token_id=tokenizer.eos_token_id,
                )
            c_raw = tokenizer.decode(c_out[0, q_ids.shape[1]:], skip_special_tokens=False)
            c_calls = parse_tool_calls(c_raw)
            c_svc = c_calls[0].get("arguments", {}).get("service", "") if c_calls else "PARSE_ERROR"
            c_valid = c_svc == target_svc
            constrained_results.append({
                "query": q_text,
                "expected_service": target_svc,
                "got_service": c_svc,
                "valid": c_valid,
                "raw_output": c_raw,
            })

    u_pass = sum(1 for r in unconstrained_results if r["valid"])
    c_pass = sum(1 for r in constrained_results if r["valid"])
    print(
        f"  [M5 Part A] Unconstrained Loose Tool Schema Valid Enum: {u_pass}/{len(unconstrained_results)} "
        f"({round(100.0 * u_pass / len(unconstrained_results), 1)}%) | FSM Constrained: {c_pass}/{len(constrained_results)} "
        f"({round(100.0 * c_pass / len(constrained_results), 1)}%)",
        flush=True,
    )

    # Part B: Runtime Error Self-Correction via <tool_response> across 8 Broken Tool Calls
    broken_cases = [
        ("What port is our auth service Postgres DB running on?", "auth_service", "auth-api"),
        ("Check Postgres connections for billing_backend.", "billing_backend", "billing-api"),
        ("Is our search_service Postgres DB healthy?", "search_service", "search-api"),
        ("What is the Postgres replica lag on analytics_db?", "analytics_db", "analytics-api"),
        ("Query Postgres port and health for authentication-service.", "authentication-service", "auth-api"),
        ("Check Postgres active connections on payments-api.", "payments-api", "billing-api"),
        ("Query Postgres status for catalog-search.", "catalog-search", "search-api"),
        ("Check Postgres status for metrics-analytics.", "metrics-analytics", "analytics-api"),
    ]

    self_correction_modes = {}
    for sc_mode in ["opaque_error_no_options", "actionable_error_with_valid_enums"]:
        sc_rows = []
        recovered_count = 0
        for q_text, wrong_arg, right_arg in broken_cases:
            if sc_mode == "opaque_error_no_options":
                err_payload = {"error": f"KeyError: '{wrong_arg}' not found in SERVICE_DB."}
                sys_msg = "You are a production SRE assistant."
            else:
                err_payload = {
                    "error": f"KeyError: Unknown service '{wrong_arg}'. Valid service names are: ['auth-api', 'billing-api', 'search-api', 'analytics-api']. Please call query_postgres_status again with the matching valid service name."
                }
                sys_msg = "You are an autonomous SRE agent. When a tool returns a KeyError listing valid service names, immediately emit a new <tool_call> with the corrected service name."

            msgs_retry = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": q_text},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "query_postgres_status", "arguments": {"service": wrong_arg}},
                        }
                    ],
                },
                {"role": "tool", "name": "query_postgres_status", "content": json.dumps(err_payload)},
            ]
            retry_prompt = tokenizer.apply_chat_template(
                msgs_retry, tools=loose_tool, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            retry_ids = tokenizer(retry_prompt, return_tensors="pt").input_ids.to(device)
            res_retry = timed_prefill_and_decode(model, tokenizer, retry_ids, max_new_tokens=120)
            retry_calls = parse_tool_calls(res_retry["raw_output"])
            corrected_arg = retry_calls[0].get("arguments", {}).get("service", "") if retry_calls else "NONE"
            did_recover = corrected_arg == right_arg

            final_answer = ""
            if did_recover:
                recovered_count += 1
                good_payload = execute_python_tool("query_postgres_status", {"service": corrected_arg})
                msgs_final = msgs_retry + [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "query_postgres_status", "arguments": {"service": corrected_arg}},
                            }
                        ],
                    },
                    {"role": "tool", "name": "query_postgres_status", "content": json.dumps(good_payload)},
                ]
                final_prompt = tokenizer.apply_chat_template(
                    msgs_final, tools=loose_tool, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                final_ids = tokenizer(final_prompt, return_tensors="pt").input_ids.to(device)
                res_final = timed_prefill_and_decode(model, tokenizer, final_ids, max_new_tokens=80)
                final_answer = res_final["clean_output"]

            sc_rows.append({
                "query": q_text,
                "initial_wrong_service_arg": wrong_arg,
                "tool_error_returned": err_payload["error"],
                "retry_raw_output": res_retry["raw_output"][:220],
                "retry_corrected_service_arg": corrected_arg,
                "recovered_successfully": did_recover,
                "retry_latency_ms": res_retry["total_ms"],
                "final_verified_answer": final_answer,
            })

        rec_pct = round(100.0 * recovered_count / len(broken_cases), 1)
        print(f"  [M5 Part B {sc_mode}] Recovery Rate: {recovered_count}/{len(broken_cases)} ({rec_pct}%)", flush=True)
        self_correction_modes[sc_mode] = {
            "trials": len(broken_cases),
            "recovered_count": recovered_count,
            "recovery_rate_pct": rec_pct,
            "cases": sc_rows,
        }

    return {
        "unconstrained_vs_fsm": {
            "trials": len(unconstrained_results),
            "unconstrained_valid_count": u_pass,
            "unconstrained_valid_pct": round(100.0 * u_pass / len(unconstrained_results), 1),
            "constrained_valid_count": c_pass,
            "constrained_valid_pct": round(100.0 * c_pass / len(constrained_results), 1),
            "sample_unconstrained_runs": unconstrained_results[:8],
            "sample_constrained_runs": constrained_results[:8],
            "fsm_step_trace": captured_fsm_trace,
        },
        "error_self_correction": self_correction_modes,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} in bfloat16 on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    env = measure_m0_environment(model, tokenizer)
    m1 = measure_m1_two_call_handshake(model, tokenizer, device)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    m2 = measure_m2_token_151657_logit_lens_and_attention(model, tokenizer, device)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    m3 = measure_m3_sequential_vs_parallel(model, tokenizer, device)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    m4 = measure_m4_tool_bloat_and_schema_quality(model, tokenizer, device)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    m5 = measure_m5_constrained_tools_and_self_correction(model, tokenizer, device)

    results = {
        "gpu_environment": env,
        "m1_two_call_handshake_and_prefix_cache": m1,
        "m2_token_151657_logit_lens_and_attention": m2,
        "m3_sequential_vs_parallel_tool_calls": m3,
        "m4_tool_bloat_scaling_and_confusion": m4,
        "m5_constrained_tools_and_self_correction": m5,
    }

    os.makedirs("projects/37_tool_calling/data", exist_ok=True)
    out_path = "projects/37_tool_calling/data/tool_calling_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full Ep37 GPU telemetry to {out_path}", flush=True)

    blob = base64.b64encode(zlib.compress(json.dumps(results).encode("utf-8"), 9)).decode("ascii")
    print("BEGIN_EP37_RESULTS_B64", flush=True)
    print(blob, flush=True)
    print("END_EP37_RESULTS_B64", flush=True)
    _stop_hb.set()


if __name__ == "__main__":
    main()
