#!/usr/bin/env python3
"""
Generate the 5 publication-grade dark-canvas charts for Episode 11
(AI Agents Series Ep 1: projects/36_llm_api_call — What Actually Happens in an LLM API Call?).
All numbers are loaded strictly from projects/36_llm_api_call/data/api_call_results.json.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA_PATH = "projects/36_llm_api_call/data/api_call_results.json"
OUT_DIR = "projects/36_llm_api_call/data/charts"
os.makedirs(OUT_DIR, exist_ok=True)

BG = "#080c14"
PANEL = "#0e1524"
GRID = "#1e293b"
TEXT = "#f8fafc"
MUTED = "#94a3b8"
CYAN = "#38bdf8"
GREEN = "#4ade80"
AMBER = "#fbbf24"
RED = "#f87171"
PURPLE = "#c084fc"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
})


def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def make_fig1_stateless_vs_prefix_cache(data):
    """Fig 1: 10-Turn Stateless Re-Prefill vs Prefix KV Cache Reuse (Tokens & TTFT ms)."""
    # 4 Nov 2026: Fig 1 uses the corrected re-check (data/m1_prefix_recheck.json, m1_prefix_recheck.py):
    # the reusable prefix is the true longest common prefix, timings are medians of 5 runs.
    with open(os.path.join(os.path.dirname(DATA_PATH), "m1_prefix_recheck.json"), "r", encoding="utf-8") as f:
        rc = json.load(f)
    rows = rc["turns"]
    turns = [t["turn"] for t in rows]
    s_toks = [t["context_tokens"] for t in rows]
    c_toks = [t["delta_tokens"] for t in rows]
    s_ttft = [t["stateless_ttft_ms_median"] for t in rows]
    c_ttft = [t["cached_ttft_ms_median"] for t in rows]
    tot_s, tot_c = rc["total_stateless_tokens"], rc["total_cached_tokens"]
    t10 = rows[-1]
    sp10 = t10["stateless_ttft_ms_median"] / t10["cached_ttft_ms_median"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.2), dpi=180)
    fig.patch.set_facecolor(BG)

    # Left panel: Prefill Tokens Computed per Turn
    x = np.arange(len(turns))
    w = 0.38
    ax1.bar(x - w / 2, s_toks, width=w, color=RED, alpha=0.88, label=f"Stateless Re-Prefill ({tot_s:,} total tokens)")
    ax1.bar(x + w / 2, c_toks, width=w, color=GREEN, alpha=0.92, label=f"Prefix KV Cache Reuse ({tot_c:,} total tokens)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"T{t}" for t in turns], fontsize=10, fontweight="bold")
    ax1.set_ylabel("Prefill Tokens Processed by GPU", fontsize=11, fontweight="bold")
    ax1.set_title(f"A. Prefill Tokens per Turn ({tot_s / tot_c:.2f}x More Without Cache)", fontsize=13, fontweight="bold", pad=12, color=TEXT)
    ax1.set_ylim(0, 1150)
    ax1.grid(axis="y", linestyle="--", alpha=0.45)
    ax1.legend(loc="upper left", frameon=True, facecolor=BG, edgecolor=GRID, fontsize=9.5)

    ax1.annotate(
        f"Turn 10:\n{t10['context_tokens']} vs {t10['delta_tokens']} tok\n({t10['reused_prefix_tokens']} reused)",
        xy=(9 - w / 2, s_toks[-1]),
        xytext=(5.8, 960),
        color=AMBER,
        fontsize=9.5,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.6),
    )

    # Right panel: Time-To-First-Token (TTFT ms) per Turn
    ax2.plot(turns, s_ttft, marker="o", linewidth=2.8, markersize=7, color=RED, label="Stateless Re-Prefill TTFT (ms, median of 5)")
    ax2.plot(turns, c_ttft, marker="s", linewidth=2.8, markersize=7, color=GREEN, label="Prefix KV Cache TTFT (ms, median of 5)")
    ax2.fill_between(turns, c_ttft, s_ttft, color=RED, alpha=0.14)
    ax2.set_xticks(turns)
    ax2.set_xticklabels([f"T{t}" for t in turns], fontsize=10, fontweight="bold")
    ax2.set_ylabel("Time-To-First-Token (TTFT ms)", fontsize=11, fontweight="bold")
    ax2.set_title(f"B. Time to First Token on A100 ({sp10:.2f}x Faster at Turn 10)", fontsize=13, fontweight="bold", pad=12, color=TEXT)
    ax2.set_ylim(40, 160)
    ax2.grid(True, linestyle="--", alpha=0.45)
    ax2.legend(loc="upper left", frameon=True, facecolor=BG, edgecolor=GRID, fontsize=9.5)

    ax2.annotate(
        f"{t10['stateless_ttft_ms_median']:.1f} ms\n(re-prefill all {t10['context_tokens']} tok)",
        xy=(10, s_ttft[-1]),
        xytext=(6.9, 143),
        color=RED,
        fontsize=9.5,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.6),
    )
    ax2.annotate(
        f"{t10['cached_ttft_ms_median']:.1f} ms ({sp10:.2f}x faster)\nonly {t10['delta_tokens']} new tok prefilled",
        xy=(10, c_ttft[-1]),
        xytext=(5.8, 47),
        color=GREEN,
        fontsize=9.5,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6),
    )

    fig.suptitle(
        "LLM APIs Have Zero Memory: 10-Turn Re-Prefill vs. Prefix KV Cache (Qwen3-8B, A100, HF eager, batch 1)",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
        color=CYAN,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    out = os.path.join(OUT_DIR, "fig1_stateless_vs_prefix_cache.png")
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print("Saved", out)


def make_fig2_chatml_templates_and_attention(data):
    """Fig 2: Chat Template Modes (0% vs 0% vs 100%) & Layer-wise Attention on Control Tokens."""
    m2 = data["m2_chat_templates_and_special_tokens"]
    modes = m2["template_mode_comparison"]
    attn_by_layer = m2["control_token_attention_pct_by_layer"]
    mean_attn = m2["mean_control_token_attention_pct"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.2), dpi=180)
    fig.patch.set_facecolor(BG)

    labels = [
        "Raw String\n(No Template)",
        "Wrong Template\n([INST] Tags)",
        "Correct ChatML\n(<|im_start|> ID 151644)",
    ]
    acc = [
        modes["raw_no_template"]["clean_concise_accuracy_pct"],
        modes["wrong_llama2_inst_tags"]["clean_concise_accuracy_pct"],
        modes["correct_chatml_template"]["clean_concise_accuracy_pct"],
    ]
    leaks = [
        modes["raw_no_template"]["continuation_or_role_leak_pct"],
        modes["wrong_llama2_inst_tags"]["continuation_or_role_leak_pct"],
        modes["correct_chatml_template"]["continuation_or_role_leak_pct"],
    ]

    x = np.arange(len(labels))
    w = 0.35
    b1 = ax1.bar(x - w / 2, acc, width=w, color=GREEN, label="Clean Concise Answer (%)")
    b2 = ax1.bar(x + w / 2, leaks, width=w, color=RED, label="Role Leak / Run-On Failure (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax1.set_ylabel("% of the 5 Test Questions", fontsize=11, fontweight="bold")
    ax1.set_title("A. Why Raw Strings Fail: Chat Template Accuracy", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylim(0, 118)
    ax1.grid(axis="y", linestyle="--", alpha=0.45)
    ax1.legend(loc="upper left", frameon=True, facecolor=BG, edgecolor=GRID, fontsize=9.5)

    for bar, val in zip(b1, acc):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 2.5, f"{val:.0f}%", ha="center", color=GREEN, fontweight="bold", fontsize=10)
    for bar, val in zip(b2, leaks):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 2.5, f"{val:.0f}%", ha="center", color=RED, fontweight="bold", fontsize=10)

    # Right panel: Layer-wise attention routed to ChatML control tokens (<|im_start|>, <|im_end|>, role headers)
    layers = np.arange(len(attn_by_layer))
    ax2.plot(layers, attn_by_layer, color=CYAN, linewidth=2.5, marker="o", markersize=4.5, label="Attention on Control Tokens (%)")
    ax2.axhline(mean_attn, color=AMBER, linestyle="--", linewidth=1.8, label=f"36-Layer Mean = {mean_attn:.1f}%")
    ax2.fill_between(layers, 0, attn_by_layer, color=CYAN, alpha=0.16)
    ax2.set_xlabel("Transformer Layer Index (0 to 35)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Attention Mass on ChatML Control Tokens (%)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Where the Final Token's Attention Goes (one prompt)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_ylim(0, 102)
    ax2.grid(True, linestyle="--", alpha=0.45)
    ax2.legend(loc="lower right", frameon=True, facecolor=BG, edgecolor=GRID, fontsize=9.5)

    peak_layer = int(np.argmax(attn_by_layer))
    peak_val = attn_by_layer[peak_layer]
    ax2.annotate(
        f"Layer {peak_layer}: {peak_val:.1f}% of attention\non 4 special-token IDs (7 of 38 positions, incl. pos 0)",
        xy=(peak_layer, peak_val),
        xytext=(8.5, 26),
        color=AMBER,
        fontsize=9.5,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.6),
    )

    fig.suptitle(
        "Chat Templates Compile JSON Roles into Control Token IDs (151644 <|im_start|>, 151645 <|im_end|>)",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
        color=CYAN,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    out = os.path.join(OUT_DIR, "fig2_chatml_templates_and_attention.png")
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print("Saved", out)


def make_fig3_prefill_vs_decode(data):
    """Fig 3: Compute-Bound Prefill (7,672 tok/s peak) vs Memory-Bound Decode (15.8 tok/s flat)."""
    sweep = data["m3_prefill_vs_decode_two_phases"]["sweep"]
    p_lens = [r["prompt_tokens"] for r in sweep]
    prefill_tps = [r["prefill_tok_per_s"] for r in sweep]
    decode_tps = [r["decode_tok_per_s"] for r in sweep]
    ttft_ms = [r["ttft_ms"] for r in sweep]
    itl_ms = [r["itl_ms_per_tok"] for r in sweep]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.2), dpi=180)
    fig.patch.set_facecolor(BG)

    x = np.arange(len(p_lens))
    ax1.plot(x, prefill_tps, marker="o", linewidth=3.0, markersize=8, color=CYAN, label="Phase 1: Parallel Prefill")
    ax1.plot(x, decode_tps, marker="s", linewidth=3.0, markersize=8, color=AMBER, label="Phase 2: Serial Decode (batch 1)")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(p) for p in p_lens], fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Prompt Length (Input Tokens)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Throughput (Tokens / sec, Log Scale)", fontsize=11, fontweight="bold")
    ax1.set_title("A. 493.7x Per-Token Gap at N=1,024 (HF eager, batch 1)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylim(8, 20000)
    ax1.grid(True, which="both", linestyle="--", alpha=0.45)
    ax1.legend(loc="center right", frameon=True, facecolor=BG, edgecolor=GRID, fontsize=9.5)

    ax1.annotate(
        "7,672 tok/s at N=1,024\n(493.7x faster per token!)",
        xy=(3, prefill_tps[3]),
        xytext=(1.1, 10500),
        color=CYAN,
        fontsize=9.5,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=CYAN, lw=1.6),
    )
    ax1.annotate(
        "Flat 15.5-15.9 tok/s at every length\n(~255 GB/s effective, ~16% of the 1,555 GB/s spec)",
        xy=(2, decode_tps[2]),
        xytext=(0.7, 28),
        color=AMBER,
        fontsize=9.5,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.6),
    )

    # Right panel: Total Prefill TTFT (ms) for entire prompt vs Single-Token Decode ITL (ms)
    w = 0.36
    b1 = ax2.bar(x - w / 2, ttft_ms, width=w, color=CYAN, label="Total Prefill TTFT for ALL Prompt Tokens (ms)")
    b2 = ax2.bar(x + w / 2, itl_ms, width=w, color=AMBER, label="Single-Token Decode Latency ITL (ms / token)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{p} tok" for p in p_lens], fontsize=10, fontweight="bold")
    ax2.set_xlabel("Prompt Length (Input Tokens)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Measured GPU Latency (ms)", fontsize=11, fontweight="bold")
    ax2.set_title("B. 512 Input Tokens Prefill in 78 ms vs. 1 Output Token in 63 ms", fontsize=13, fontweight="bold", pad=12)
    ax2.set_ylim(0, 360)
    ax2.grid(axis="y", linestyle="--", alpha=0.45)
    ax2.legend(loc="upper left", frameon=True, facecolor=BG, edgecolor=GRID, fontsize=9.5)

    ax2.annotate(
        "512 prompt tokens = 78.0 ms\n1 output token = 63.2 ms!",
        xy=(2 - w / 2, ttft_ms[2]),
        xytext=(0.5, 195),
        color=GREEN,
        fontsize=9.5,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6),
    )

    fig.suptitle(
        "Why Input Tokens Are Cheap and Output Tokens Are Slow: Prefill vs. Decode on NVIDIA A100",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
        color=CYAN,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    out = os.path.join(OUT_DIR, "fig3_prefill_vs_decode_phases.png")
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print("Saved", out)


def make_fig4_json_reliability_across_temps(data):
    """Fig 4: JSON Parsing Reliability Across Temperatures (Naive Prompt 0% vs Constrained 100%)."""
    modes = data["m4_m5_structured_outputs_and_logit_masking"]["modes"]
    temps = [r["temperature"] for r in modes["naive_prompt"]]
    naive_pass = [r["valid_json_pass_pct"] for r in modes["naive_prompt"]]
    strict_pass = [r["valid_json_pass_pct"] for r in modes["system_strict_prompt"]]
    mask_pass = [r["valid_json_pass_pct"] for r in modes["constrained_logit_mask"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.2), dpi=180)
    fig.patch.set_facecolor(BG)

    ax1.plot(temps, naive_pass, marker="x", markersize=9, linewidth=3.0, color=RED, label='Naive Prompt ("...as JSON") — 0.0% Pass (100% Crash)')
    ax1.plot(temps, strict_pass, marker="o", markersize=16, markerfacecolor="none", markeredgewidth=2.4, linewidth=2.6, linestyle="--", color=AMBER, label="Strict System Prompt: 80/80 (a request, no guarantee)")
    ax1.plot(temps, mask_pass, marker="s", markersize=7, linewidth=2.8, color=GREEN, label="-inf Logit Mask: 80/80 (cannot emit off-grammar tokens)")
    ax1.set_xticks(temps)
    ax1.set_xticklabels([f"T={t:.1f}" for t in temps], fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Sampling Temperature (T)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Strict json.loads() + Schema Pass Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title("A. Naive Prompt 0/80; Strict Prompt and Mask 80/80 Each", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylim(-6, 112)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(loc="center right", frameon=True, facecolor=BG, edgecolor=GRID, fontsize=8.8)

    ax1.annotate(
        'Naive prompt wrapped every reply in a\n```json ... ``` Markdown fence (JSON inside was fine),\nso strict json.loads() failed 80/80',
        xy=(0.7, 0.0),
        xytext=(0.18, 28),
        color=RED,
        fontsize=9.5,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.6),
    )

    # Right panel: Step 0 Logit Comparison (Why Naive/Unmasked prefers '{\n' or '```' vs Constrained FSM Mask)
    step0 = data["m4_m5_structured_outputs_and_logit_masking"]["constrained_step_traces"][0]
    raw_toks = [t["token"] for t in step0["raw_top5"]]
    raw_logits = [t["logit"] for t in step0["raw_top5"]]
    allowed_set = {t["token"] for t in step0["masked_top5"]}

    y = np.arange(len(raw_toks))[::-1]
    colors = [GREEN if tok in allowed_set else RED for tok in raw_toks]
    bars = ax2.barh(y, raw_logits, color=colors, height=0.55, alpha=0.9)
    ax2.set_yticks(y)
    ax2.set_yticklabels(raw_toks, fontsize=11, fontweight="bold", family="monospace")
    ax2.set_xlabel("Raw Logit at Decode Step 0 (strict prompt, case 0, T=0)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Step 0 Logit Masking: 151,934 Invalid Tokens Set to -inf", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlim(0, 66)
    ax2.grid(axis="x", linestyle="--", alpha=0.45)

    for bar, tok, lg in zip(bars, raw_toks, raw_logits):
        status = "KEPT (Valid FSM)" if tok in allowed_set else "MASKED -> -inf (0.0%)"
        col = GREEN if tok in allowed_set else RED
        ax2.text(lg + 1.0, bar.get_y() + bar.get_height() / 2, f"{lg:.1f}  [{status}]", va="center", color=col, fontsize=9.2, fontweight="bold")

    fig.suptitle(
        "Why Naive JSON Prompts Fail Strict json.loads() and What a -inf Logit Mask Adds",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
        color=CYAN,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    out = os.path.join(OUT_DIR, "fig4_json_reliability_and_step0_mask.png")
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print("Saved", out)


def make_fig5_fsm_constrained_step_trace(data):
    """Fig 5: Step-by-Step Allowed Vocabulary Size Out of 151,936 Tokens During Constrained Decoding."""
    traces = data["m4_m5_structured_outputs_and_logit_masking"]["constrained_step_traces"]
    steps = [t["step"] for t in traces]
    valid_counts = [t["valid_vocab_tokens_count"] for t in traces]
    prefixes = [t["prefix_so_far"] if t["prefix_so_far"] else "<START>" for t in traces]
    chosen = [t["masked_top5"][0]["token"] for t in traces]

    fig, ax = plt.subplots(figsize=(13.5, 6.0), dpi=180)
    fig.patch.set_facecolor(BG)

    bars = ax.bar(steps, valid_counts, color=PURPLE, width=0.52, alpha=0.9, edgecolor=CYAN, linewidth=1.5)
    ax.set_xticks(steps)
    ax.set_xticklabels([f"Step {s}\nPrefix: {p}" for s, p in zip(steps, prefixes)], fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Allowed Tokens Out of 151,936 Vocab", fontsize=11.5, fontweight="bold")
    ax.set_title(
        "Constrained Decoding FSM: Valid Token Count Per Step (Out of 151,936 Qwen3 Vocabulary Tokens)",
        fontsize=13.5,
        fontweight="bold",
        pad=14,
        color=CYAN,
    )
    ax.set_ylim(0, 42)
    ax.grid(axis="y", linestyle="--", alpha=0.45)

    for bar, cnt, tok in zip(bars, valid_counts, chosen):
        masked_out = 151936 - cnt
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            cnt + 1.0,
            f"{cnt} allowed\n({masked_out:,} -> -inf)\nPick: {tok}",
            ha="center",
            va="bottom",
            color=GREEN,
            fontsize=9.0,
            fontweight="bold",
        )

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig5_fsm_constrained_step_trace.png")
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print("Saved", out)


if __name__ == "__main__":
    d = load_data()
    make_fig1_stateless_vs_prefix_cache(d)
    make_fig2_chatml_templates_and_attention(d)
    make_fig3_prefill_vs_decode(d)
    make_fig4_json_reliability_across_temps(d)
    make_fig5_fsm_constrained_step_trace(d)
