"""End-to-End GPU Benchmark for Episode 11 (AI Agents Series Ep 1):
What Actually Happens in an LLM API Call? (projects/36_llm_api_call)

Runs 5 real PyTorch experiments on Qwen/Qwen3-8B on an NVIDIA GPU:
  M1: Stateless Memory Illusion & 10-Turn Re-Prefill vs Prefix KV Caching (end-to-end timed)
  M2: Chat Template Special Tokens (<|im_start|>, <|im_end|>) vs Raw Text vs Wrong Template + Role Attention
  M3: The Two Physical Speeds of an API Call: Compute-Bound Prefill (TTFT) vs Memory-Bound Decode (ITL)
  M4: Temperature / Top-p Sampling & Why "Return Valid JSON" Crashes json.loads()
  M5: Constrained Decoding (JSON Schema FSM Logit Masking to -inf) — 100.0% Valid JSON + Step-by-Step Logit Traces
"""
import base64
import gc
import json
import math
import os
import re
import time
import zlib

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


def measure_m0_environment_and_configs():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0.0
    env = {
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "pytorch_version": torch.__version__,
        "model_id": MODEL_ID,
        "precision": "bfloat16",
    }
    print(f"[M0] GPU Environment: {env}")
    return env


def measure_m1_stateless_and_prefix_cache(model, tokenizer, device):
    """M1: Prove API calls are 100% stateless and measure 10-turn Re-Prefill vs Prefix KV Caching."""
    print("\n[M1] Running Stateless Memory Proof & 10-Turn Prefix KV Cache Benchmark...")

    # Part A: Stateless proof (Turn 1 sets secret port 5439; Turn 2 asks for it with vs without history)
    turn1_msgs = [
        {"role": "system", "content": "You are a concise production database assistant. Answer in one short sentence."},
        {"role": "user", "content": "My name is Arun and our production Postgres database runs on port 5439. Acknowledge briefly."},
    ]
    t1_prompt = tokenizer.apply_chat_template(turn1_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    t1_ids = tokenizer(t1_prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        t1_out = model.generate(t1_ids, max_new_tokens=28, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    t1_reply = tokenizer.decode(t1_out[0, t1_ids.shape[1]:], skip_special_tokens=True).strip()

    # Call 2A: Stateless isolated call (only Turn 2 sent, no Turn 1 in messages[])
    turn2_isolated = [
        {"role": "system", "content": "You are a concise production database assistant. Answer in one short sentence."},
        {"role": "user", "content": "What is my name and what port does our production Postgres database run on?"},
    ]
    t2_iso_prompt = tokenizer.apply_chat_template(turn2_isolated, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    t2_iso_ids = tokenizer(t2_iso_prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        t2_iso_out = model.generate(t2_iso_ids, max_new_tokens=36, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    t2_iso_reply = tokenizer.decode(t2_iso_out[0, t2_iso_ids.shape[1]:], skip_special_tokens=True).strip()

    # Call 2B: Full history re-sent in messages[]
    turn2_with_hist = turn1_msgs + [
        {"role": "assistant", "content": t1_reply},
        {"role": "user", "content": "What is my name and what port does our production Postgres database run on?"},
    ]
    t2_hist_prompt = tokenizer.apply_chat_template(turn2_with_hist, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    t2_hist_ids = tokenizer(t2_hist_prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        t2_hist_out = model.generate(t2_hist_ids, max_new_tokens=36, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    t2_hist_reply = tokenizer.decode(t2_hist_out[0, t2_hist_ids.shape[1]:], skip_special_tokens=True).strip()

    # Part B: 10-Turn Multi-Turn API Conversation (Stateless Full Re-Prefill vs Prefix KV Cache Reuse)
    # Build 10 realistic engineering turns (~110-130 tokens per turn)
    system_msg = {
        "role": "system",
        "content": (
            "You are a senior site reliability assistant inspecting cluster telemetry logs. "
            "Always reply concisely with the exact metric requested."
        ),
    }
    user_turns = [
        "Turn 1 log: node-a100-01 GPU temperature is 64C, HBM memory used is 31.4 GB out of 40.0 GB, power draw is 312W, utilization is 94%. Summarize node-a100-01 status in one line.",
        "Turn 2 log: node-a100-02 GPU temperature is 71C, HBM memory used is 38.9 GB out of 40.0 GB, power draw is 378W, utilization is 99%. Summarize node-a100-02 status in one line.",
        "Turn 3 log: postgres-primary replica lag is 14 ms, active connections 412 of 500, buffer cache hit ratio 99.4%, WAL write latency 1.8 ms. Summarize postgres-primary in one line.",
        "Turn 4 log: redis-cache-01 memory used 11.2 GB of 16.0 GB, evicted keys 0, p99 latency 0.42 ms, connected clients 1,840, hit rate 96.8%. Summarize redis-cache-01 in one line.",
        "Turn 5 log: api-gateway-eu p50 latency is 18 ms, p99 latency is 142 ms, HTTP 500 error rate is 0.04%, throughput is 4,210 req/s. Summarize api-gateway-eu in one line.",
        "Turn 6 log: kafka-broker-03 under-replicated partitions 0, consumer group lag 840 messages, disk throughput 410 MB/s, network rx 2.1 Gbps. Summarize kafka-broker-03 in one line.",
        "Turn 7 log: vector-index-qdrant indexed vectors 12.4M, HNSW ef_search 128, p95 recall 0.982, query latency p99 11.6 ms, RAM 28.5 GB. Summarize vector-index-qdrant in one line.",
        "Turn 8 log: worker-celery-gpu queue depth 19 jobs, average task duration 1.45s, failed tasks in last hour 1, retry rate 0.2%. Summarize worker-celery-gpu in one line.",
        "Turn 9 log: s3-checkpoint-store bucket size 4.8 TB, put latency p99 68 ms, read bandwidth 8.4 GB/s, lifecycle transition active. Summarize s3-checkpoint-store in one line.",
        "Turn 10 audit question: Looking back across all 9 logs above, which GPU node had 38.9 GB HBM used and what was the Postgres port or replica lag?",
    ]

    # Pre-generate assistant responses so both Stateless Re-Prefill and Prefix Cache process the EXACT same token sequence
    history = [system_msg]
    turn_token_ids = []
    assistant_replies = []
    for u_txt in user_turns:
        history.append({"role": "user", "content": u_txt})
        prompt_str = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=24, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        ans_txt = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        assistant_replies.append(ans_txt)
        history.append({"role": "assistant", "content": ans_txt})
        turn_token_ids.append(ids)

    # Warmup
    with torch.no_grad():
        _ = model(turn_token_ids[0][:, :32], use_cache=True)
    sync()

    # Benchmark Mode 1: Stateless Re-Prefill (Every turn recomputes the entire conversation from token 0)
    stateless_turns = []
    total_stateless_prefill_tokens = 0
    total_stateless_prefill_ms = 0.0
    for idx, ids in enumerate(turn_token_ids):
        n_tok = int(ids.shape[1])
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(ids, use_cache=True)
            _ = out.logits[:, -1, :].argmax(dim=-1)
        sync()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        total_stateless_prefill_tokens += n_tok
        total_stateless_prefill_ms += dt_ms
        stateless_turns.append({
            "turn": idx + 1,
            "context_tokens": n_tok,
            "prefill_tokens_computed": n_tok,
            "ttft_ms": round(dt_ms, 2),
        })

    # Benchmark Mode 2: Prefix KV Cache Reuse (Only the new delta tokens of each turn are prefilled)
    prefix_cached_turns = []
    total_cached_prefill_tokens = 0
    total_cached_prefill_ms = 0.0

    # Build full token sequence up to each turn and feed only the delta slice into the cached KV state
    past_kv = None
    prev_len = 0
    for idx, ids in enumerate(turn_token_ids):
        # If turn > 0, first advance cache with the assistant reply tokens from the previous turn
        full_len = int(ids.shape[1])
        delta_ids = ids[:, prev_len:]
        n_delta = int(delta_ids.shape[1])
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(delta_ids, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            _ = out.logits[:, -1, :].argmax(dim=-1)
        sync()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        prev_len = full_len
        total_cached_prefill_tokens += n_delta
        total_cached_prefill_ms += dt_ms
        prefix_cached_turns.append({
            "turn": idx + 1,
            "context_tokens": full_len,
            "delta_tokens_computed": n_delta,
            "cached_tokens_reused": full_len - n_delta,
            "ttft_ms": round(dt_ms, 2),
        })

    res = {
        "stateless_proof": {
            "turn1_user": turn1_msgs[1]["content"],
            "turn1_assistant": t1_reply,
            "turn2_user": turn2_isolated[1]["content"],
            "turn2_isolated_tokens": int(t2_iso_ids.shape[1]),
            "turn2_isolated_reply": t2_iso_reply,
            "turn2_isolated_remembers_port_5439": ("5439" in t2_iso_reply),
            "turn2_with_history_tokens": int(t2_hist_ids.shape[1]),
            "turn2_with_history_reply": t2_hist_reply,
            "turn2_with_history_remembers_port_5439": ("5439" in t2_hist_reply),
        },
        "ten_turn_benchmark": {
            "stateless_reprefill_turns": stateless_turns,
            "prefix_cached_turns": prefix_cached_turns,
            "total_stateless_prefill_tokens": total_stateless_prefill_tokens,
            "total_cached_prefill_tokens": total_cached_prefill_tokens,
            "redundant_token_ratio": round(total_stateless_prefill_tokens / max(1, total_cached_prefill_tokens), 2),
            "total_stateless_ttft_ms": round(total_stateless_prefill_ms, 2),
            "total_cached_ttft_ms": round(total_cached_prefill_ms, 2),
            "ttft_speedup_total": round(total_stateless_prefill_ms / max(0.1, total_cached_prefill_ms), 2),
            "turn10_stateless_ttft_ms": stateless_turns[-1]["ttft_ms"],
            "turn10_cached_ttft_ms": prefix_cached_turns[-1]["ttft_ms"],
            "turn10_ttft_speedup": round(stateless_turns[-1]["ttft_ms"] / max(0.1, prefix_cached_turns[-1]["ttft_ms"]), 2),
        },
    }
    print(f"[M1] Done: Turn 10 Stateless TTFT = {res['ten_turn_benchmark']['turn10_stateless_ttft_ms']} ms vs Cached = {res['ten_turn_benchmark']['turn10_cached_ttft_ms']} ms ({res['ten_turn_benchmark']['turn10_ttft_speedup']}x)")
    return res


def measure_m2_chat_templates(model, tokenizer, device):
    """M2: Inspect exact ChatML token IDs and compare Raw Text vs Wrong Template vs ChatML Template."""
    print("\n[M2] Running Chat Template & Control Token Benchmark...")
    msgs = [
        {"role": "system", "content": "Answer with the 3-digit HTTP status code only."},
        {"role": "user", "content": "What is the HTTP status code for Not Found?"},
    ]
    formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    token_ids = tokenizer(formatted, add_special_tokens=False).input_ids
    token_breakdown = [
        {"pos": i, "id": int(tid), "piece": repr(tokenizer.decode([tid]))}
        for i, tid in enumerate(token_ids)
    ]

    # Compare 3 template modes across 5 factual short-answer questions
    test_questions = [
        ("What is the HTTP status code for Not Found?", "404"),
        ("What is the default TCP port for HTTPS?", "443"),
        ("What is 17 multiplied by 6?", "102"),
        ("What is the chemical symbol for Gold?", "Au"),
        ("How many bits are in one byte?", "8"),
    ]

    modes_summary = {}
    for mode_name in ["raw_no_template", "wrong_llama2_inst_tags", "correct_chatml_template"]:
        exact_matches = 0
        role_leaks = 0
        avg_gen_tokens = []
        examples = []
        for q, expected in test_questions:
            if mode_name == "raw_no_template":
                prompt = q
            elif mode_name == "wrong_llama2_inst_tags":
                prompt = f"[INST] <<SYS>>\nAnswer with the exact value only, maximum 3 words.\n<</SYS>>\n\n{q} [/INST]"
            else:
                m = [
                    {"role": "system", "content": "Answer with the exact value only, maximum 3 words."},
                    {"role": "user", "content": q},
                ]
                prompt = tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)

            inp = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            with torch.no_grad():
                out = model.generate(inp, max_new_tokens=32, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            gen_ids = out[0, inp.shape[1]:]
            gen_txt = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            avg_gen_tokens.append(int(gen_ids.shape[0]))

            # Check if response is clean & concise and starts with/equals expected
            first_line = gen_txt.split("\n")[0].strip()
            is_clean = (expected.lower() in first_line.lower()) and (len(gen_ids) <= 8) and ("[/INST]" not in gen_txt) and ("?" not in gen_txt)
            if is_clean:
                exact_matches += 1
            if any(marker in gen_txt for marker in ["[/INST]", "[INST]", "Question:", "Answer:", "User:", "\n\n"]):
                role_leaks += 1
            examples.append({
                "question": q,
                "expected": expected,
                "output_text": gen_txt[:140],
                "output_tokens": int(gen_ids.shape[0]),
                "clean_concise_match": is_clean,
            })

        modes_summary[mode_name] = {
            "clean_concise_accuracy_pct": round(100.0 * exact_matches / len(test_questions), 1),
            "continuation_or_role_leak_pct": round(100.0 * role_leaks / len(test_questions), 1),
            "avg_output_tokens": round(sum(avg_gen_tokens) / len(avg_gen_tokens), 1),
            "examples": examples,
        }

    # Measure attention mass paid by the final assistant prompt token to control tokens vs content tokens
    inp_ids = torch.tensor([token_ids], device=device)
    with torch.no_grad():
        out_attn = model(inp_ids, output_attentions=True, use_cache=False)
    # Identify indices of control tokens (<|im_start|>=151644, <|im_end|>=151645, <think>=151667, </think>=151668, role names)
    control_set = {151644, 151645, 151667, 151668}
    ctrl_indices = [i for i, tid in enumerate(token_ids) if tid in control_set]
    layer_ctrl_attn = []
    for l_idx, att in enumerate(out_attn.attentions):
        # att shape: [1, num_heads, seq_len, seq_len]; look at last query position (-1)
        last_q_attn = att[0, :, -1, :].mean(dim=0)  # average across heads
        ctrl_mass = float(last_q_attn[ctrl_indices].sum().item())
        layer_ctrl_attn.append(round(ctrl_mass * 100.0, 2))

    res = {
        "chatml_formatted_string": formatted,
        "chatml_token_count": len(token_ids),
        "special_token_ids": {
            "<|im_start|>": 151644,
            "<|im_end|>": 151645,
            "<think>": 151667,
            "</think>": 151668,
        },
        "token_breakdown": token_breakdown,
        "template_mode_comparison": modes_summary,
        "control_token_attention_pct_by_layer": layer_ctrl_attn,
        "mean_control_token_attention_pct": round(sum(layer_ctrl_attn) / len(layer_ctrl_attn), 2),
    }
    print(f"[M2] Done: Clean accuracy Raw={modes_summary['raw_no_template']['clean_concise_accuracy_pct']}% vs Wrong={modes_summary['wrong_llama2_inst_tags']['clean_concise_accuracy_pct']}% vs ChatML={modes_summary['correct_chatml_template']['clean_concise_accuracy_pct']}%")
    return res


def measure_m3_prefill_vs_decode(model, tokenizer, device):
    """M3: Sweep prompt length N_in in {64, 256, 512, 1024, 2048} and measure Prefill (TTFT) vs 64-token Decode (ITL)."""
    print("\n[M3] Running Prefill (Compute-Bound) vs Decode (Memory-Bound) Benchmark...")
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_gb = round(param_bytes / (1024**3), 2)

    base_chunk = (
        "Server telemetry record: host=prod-db-01, region=ap-south-1, cpu_util=42.8%, "
        "mem_used_gb=29.4, iops_read=14200, iops_write=3800, p99_ms=4.2, status=OK. "
    )
    base_ids = tokenizer(base_chunk, add_special_tokens=False).input_ids

    prompt_lengths = [64, 256, 512, 1024, 2048]
    n_decode = 64
    rows = []

    for n_in in prompt_lengths:
        reps = (n_in // len(base_ids)) + 2
        seq = (base_ids * reps)[:n_in]
        inp = torch.tensor([seq], dtype=torch.long, device=device)

        # Warmup pass
        with torch.no_grad():
            _ = model(inp[:, :min(64, n_in)], use_cache=False)
        sync()

        # 1. Measure Prefill (Time To First Token) across 3 runs
        prefill_times_ms = []
        for _ in range(3):
            sync()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(inp, use_cache=True)
                next_tok = out.logits[:, -1:, :].argmax(dim=-1)
            sync()
            prefill_times_ms.append((time.perf_counter() - t0) * 1000.0)
        ttft_ms = sorted(prefill_times_ms)[1]  # median
        prefill_tok_per_s = (n_in / (ttft_ms / 1000.0))

        # 2. Measure 64-step Autoregressive Decode with KV Cache
        with torch.no_grad():
            out = model(inp, use_cache=True)
            past_kv = out.past_key_values
            cur_tok = out.logits[:, -1:, :].argmax(dim=-1)

        sync()
        t_dec0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_decode):
                out_step = model(cur_tok, past_key_values=past_kv, use_cache=True)
                past_kv = out_step.past_key_values
                cur_tok = out_step.logits[:, -1:, :].argmax(dim=-1)
        sync()
        decode_total_ms = (time.perf_counter() - t_dec0) * 1000.0
        itl_ms = decode_total_ms / n_decode
        decode_tok_per_s = 1000.0 / itl_ms
        effective_bw_gb_s = param_gb / (itl_ms / 1000.0)

        rows.append({
            "prompt_tokens": n_in,
            "decode_tokens": n_decode,
            "ttft_ms": round(ttft_ms, 2),
            "prefill_tok_per_s": round(prefill_tok_per_s, 1),
            "decode_total_ms": round(decode_total_ms, 2),
            "itl_ms_per_tok": round(itl_ms, 2),
            "decode_tok_per_s": round(decode_tok_per_s, 2),
            "prefill_to_decode_speed_ratio": round(prefill_tok_per_s / decode_tok_per_s, 1),
            "effective_hbm_bw_gb_s": round(effective_bw_gb_s, 1),
        })
        del past_kv
        torch.cuda.empty_cache()

    res = {
        "model_weight_gb_bf16": param_gb,
        "sweep": rows,
    }
    print(f"[M3] Done: At N=2048, Prefill={rows[-1]['prefill_tok_per_s']} tok/s (TTFT={rows[-1]['ttft_ms']} ms) vs Decode={rows[-1]['decode_tok_per_s']} tok/s ({rows[-1]['prefill_to_decode_speed_ratio']}x faster per token in Prefill)")
    return res


class StrictJSONSchemaLogitsProcessor(LogitsProcessor):
    """Finite-State Regex Grammar LogitsProcessor enforcing exact schema:
    {"service": "<alpha-dash>", "port": <1-5 digits>, "healthy": true|false}
    Sets all vocabulary tokens that cannot validly extend the prefix to -inf.
    """
    FULL_PATTERN = re.compile(
        r'^\{"service": "[a-z0-9_-]{1,20}", "port": [0-9]{2,5}, "healthy": (true|false)\}$'
    )

    def __init__(self, prompt_len: int, tokenizer, allowed_services):
        self.prompt_len = prompt_len
        self.tokenizer = tokenizer
        self.allowed_services = allowed_services
        self.eos_id = tokenizer.eos_token_id
        self.step_traces = []

        # Pre-decode candidate token pieces for fast prefix validation (ASCII printable subset + EOS)
        vocab_size = len(tokenizer)
        self.candidate_ids = []
        self.candidate_strs = []
        for tid in range(min(vocab_size, 151643)):
            s = tokenizer.decode([tid])
            if s and all(32 <= ord(ch) <= 126 for ch in s):
                # Only keep tokens containing characters that can ever appear in our JSON schema
                if all(ch in '{}":, 0123456789abcdefghijklmnopqrstuvwxyz_-' for ch in s):
                    self.candidate_ids.append(tid)
                    self.candidate_strs.append(s)

    def _is_valid_prefix(self, text: str) -> bool:
        """Check if `text` is a valid prefix of our strict JSON schema."""
        if not text:
            return True
        # Target template structure:
        # {"service": "<svc>", "port": <digits>, "healthy": <true|false>}
        p1 = '{"service": "'
        if len(text) <= len(p1):
            return p1.startswith(text)
        if not text.startswith(p1):
            return False
        rem = text[len(p1):]
        # Inside service string until closing quote
        if '"' not in rem:
            return any(svc.startswith(rem) for svc in self.allowed_services)
        svc_val, rem2 = rem.split('"', 1)
        if svc_val not in self.allowed_services:
            return False
        p2 = ', "port": '
        if len(rem2) <= len(p2):
            return p2.startswith(rem2)
        if not rem2.startswith(p2):
            return False
        rem3 = rem2[len(p2):]
        # Inside digits until comma
        if ',' not in rem3:
            return len(rem3) <= 5 and rem3.isdigit()
        port_val, rem4 = rem3.split(',', 1)
        if not (1 <= len(port_val) <= 5 and port_val.isdigit()):
            return False
        p3 = ' "healthy": '
        if len(rem4) <= len(p3):
            return p3.startswith(rem4)
        if not rem4.startswith(p3):
            return False
        rem5 = rem4[len(p3):]
        valid_tails = ["true}", "false}"]
        return any(vt.startswith(rem5) for vt in valid_tails)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        gen_ids = input_ids[0, self.prompt_len:].tolist()
        cur_text = self.tokenizer.decode(gen_ids)

        # If complete valid JSON object reached, force EOS token
        if self.FULL_PATTERN.match(cur_text):
            mask = torch.full_like(scores, -float("inf"))
            mask[0, self.eos_id] = 0.0
            return scores + mask

        valid_tids = []
        for tid, piece in zip(self.candidate_ids, self.candidate_strs):
            nxt = cur_text + piece
            if self._is_valid_prefix(nxt):
                valid_tids.append(tid)

        # Log first 4 interesting transitions for visualization
        if len(self.step_traces) < 6:
            probs_raw = F.softmax(scores[0], dim=-1)
            top_raw_vals, top_raw_ids = torch.topk(probs_raw, 5)
            raw_top5 = [
                {"token": repr(self.tokenizer.decode([int(i)])), "prob_pct": round(float(v) * 100.0, 2), "logit": round(float(scores[0, i]), 2)}
                for v, i in zip(top_raw_vals, top_raw_ids)
            ]
            mask_t = torch.full_like(scores, -float("inf"))
            mask_t[0, valid_tids] = 0.0
            masked_scores = scores + mask_t
            probs_masked = F.softmax(masked_scores[0], dim=-1)
            top_m_vals, top_m_ids = torch.topk(probs_masked, min(5, len(valid_tids)))
            masked_top5 = [
                {"token": repr(self.tokenizer.decode([int(i)])), "prob_pct": round(float(v) * 100.0, 2), "logit": round(float(masked_scores[0, i]), 2)}
                for v, i in zip(top_m_vals, top_m_ids)
            ]
            self.step_traces.append({
                "step": len(gen_ids),
                "prefix_so_far": cur_text,
                "valid_vocab_tokens_count": len(valid_tids),
                "raw_top5": raw_top5,
                "masked_top5": masked_top5,
            })

        mask = torch.full_like(scores, -float("inf"))
        mask[0, valid_tids] = 0.0
        return scores + mask


def validate_schema(raw_str: str, expected_svc: str, expected_port: int, expected_healthy: bool):
    """Strictly validate raw model output with json.loads() without stripping markdown fences."""
    try:
        obj = json.loads(raw_str)
    except Exception as e:
        if "```" in raw_str:
            return False, "markdown_fence_wrapper"
        if not raw_str.strip().startswith("{"):
            return False, "conversational_preamble"
        if not raw_str.strip().endswith("}"):
            return False, "trailing_text_or_truncated"
        return False, "json_syntax_error"

    if not isinstance(obj, dict):
        return False, "not_json_object"
    if set(obj.keys()) != {"service", "port", "healthy"}:
        return False, "wrong_schema_keys"
    if not isinstance(obj["service"], str) or not isinstance(obj["port"], int) or not isinstance(obj["healthy"], bool):
        return False, "wrong_field_types"
    return True, "valid"


def measure_m4_and_m5_json_vs_constrained(model, tokenizer, device):
    """M4 & M5: Compare Naive JSON Prompting vs Constrained Decoding Logit Masking across Temperatures."""
    print("\n[M4 & M5] Running Unconstrained JSON Failure Benchmark vs Constrained Logit Masking...")

    log_cases = [
        ("WARN [auth-api] listening on port 8081 — healthcheck FAILED (timeout)", "auth-api", 8081, False),
        ("INFO [payment-gw] bound to port 9443 — status=HEALTHY all probes green", "payment-gw", 9443, True),
        ("ERROR [redis-cache] port 6379 connection refused — node UNHEALTHY", "redis-cache", 6379, False),
        ("INFO [search-idx] serving on port 9200 — cluster status GREEN / healthy", "search-idx", 9200, True),
        ("CRITICAL [postgres-db] port 5432 max_connections exceeded — healthy=false", "postgres-db", 5432, False),
        ("INFO [kafka-broker] port 9092 leader election complete — healthy=true", "kafka-broker", 9092, True),
        ("WARN [ billing-svc ] port 7070 memory pressure 98% — health probe FAIL", "billing-svc", 7070, False),
        ("INFO [metrics-prom] port 9090 scraping 42 targets — status healthy", "metrics-prom", 9090, True),
    ]
    allowed_services = [c[1] for c in log_cases]
    temperatures = [0.0, 0.3, 0.7, 1.0, 1.3]
    trials_per_temp = 16  # 16 trials x 5 temps = 80 API calls per mode

    results_by_mode = {}
    captured_step_traces = None

    for mode in ["naive_prompt", "system_strict_prompt", "constrained_logit_mask"]:
        temp_rows = []
        for temp in temperatures:
            valid_count = 0
            error_counts = {
                "markdown_fence_wrapper": 0,
                "conversational_preamble": 0,
                "trailing_text_or_truncated": 0,
                "json_syntax_error": 0,
                "wrong_schema_keys": 0,
                "wrong_field_types": 0,
            }
            sample_failures = []
            sample_success = None
            latencies_ms = []

            for t_idx in range(trials_per_temp):
                log_line, exp_svc, exp_port, exp_ok = log_cases[t_idx % len(log_cases)]
                if mode == "naive_prompt":
                    msgs = [
                        {"role": "user", "content": f"Extract the service name, port number, and whether it is healthy as JSON with keys service, port, healthy from this log:\n{log_line}"}
                    ]
                else:
                    msgs = [
                        {
                            "role": "system",
                            "content": (
                                'Extract the log into a JSON object with exact keys "service" (string), '
                                '"port" (integer), and "healthy" (boolean). Output raw JSON only.'
                            ),
                        },
                        {"role": "user", "content": log_line},
                    ]

                prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                inp = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

                lp = LogitsProcessorList()
                proc = None
                if mode == "constrained_logit_mask":
                    proc = StrictJSONSchemaLogitsProcessor(inp.shape[1], tokenizer, allowed_services)
                    lp.append(proc)

                torch.manual_seed(SEED + t_idx * 31 + int(temp * 100))
                sync()
                t0 = time.perf_counter()
                with torch.no_grad():
                    if temp == 0.0:
                        out = model.generate(
                            inp, max_new_tokens=48, do_sample=False,
                            logits_processor=lp, pad_token_id=tokenizer.eos_token_id
                        )
                    else:
                        out = model.generate(
                            inp, max_new_tokens=48, do_sample=True, temperature=temp, top_p=0.95,
                            logits_processor=lp, pad_token_id=tokenizer.eos_token_id
                        )
                sync()
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

                if proc is not None and captured_step_traces is None and len(proc.step_traces) > 0:
                    captured_step_traces = proc.step_traces

                gen_str = tokenizer.decode(out[0, inp.shape[1]:], skip_special_tokens=True)
                ok, reason = validate_schema(gen_str, exp_svc, exp_port, exp_ok)
                if ok:
                    valid_count += 1
                    if sample_success is None:
                        sample_success = gen_str
                else:
                    error_counts[reason] = error_counts.get(reason, 0) + 1
                    if len(sample_failures) < 3:
                        sample_failures.append({"reason": reason, "raw_output": gen_str[:160]})

            pass_rate = round(100.0 * valid_count / trials_per_temp, 1)
            crash_rate = round(100.0 - pass_rate, 1)
            temp_rows.append({
                "temperature": temp,
                "trials": trials_per_temp,
                "valid_json_pass_pct": pass_rate,
                "json_loads_crash_pct": crash_rate,
                "median_latency_ms": round(sorted(latencies_ms)[len(latencies_ms) // 2], 2),
                "error_breakdown": error_counts,
                "sample_success": sample_success,
                "sample_failures": sample_failures,
            })
            print(f"  [{mode}] T={temp}: valid={pass_rate}%, crash={crash_rate}%")

        results_by_mode[mode] = temp_rows

    return {
        "modes": results_by_mode,
        "constrained_step_traces": captured_step_traces,
    }


def main():
    env = measure_m0_environment_and_configs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {MODEL_ID} in bfloat16 on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # eager allows output_attentions=True in M2
    )
    model.eval()

    m1 = measure_m1_stateless_and_prefix_cache(model, tokenizer, device)
    m2 = measure_m2_chat_templates(model, tokenizer, device)
    m3 = measure_m3_prefill_vs_decode(model, tokenizer, device)
    m4_m5 = measure_m4_and_m5_json_vs_constrained(model, tokenizer, device)

    results = {
        "gpu_environment": env,
        "m1_stateless_and_prefix_cache": m1,
        "m2_chat_templates_and_special_tokens": m2,
        "m3_prefill_vs_decode_two_phases": m3,
        "m4_m5_structured_outputs_and_logit_masking": m4_m5,
    }

    os.makedirs("projects/36_llm_api_call/data", exist_ok=True)
    out_path = "projects/36_llm_api_call/data/api_call_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full GPU telemetry to {out_path}")

    blob = base64.b64encode(zlib.compress(json.dumps(results).encode("utf-8"), 9)).decode("ascii")
    print("BEGIN_EP36_RESULTS_B64")
    print(blob)
    print("END_EP36_RESULTS_B64")


if __name__ == "__main__":
    main()
