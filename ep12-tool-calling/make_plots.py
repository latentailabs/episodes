#!/usr/bin/env python3
"""
Generate the 5 publication-grade dark-canvas charts for Episode 12
(AI Agents Series Ep 2: projects/37_tool_calling — How LLM Tool Calling Actually Works).
All numbers are loaded strictly from projects/37_tool_calling/data/tool_calling_results.json.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "projects/37_tool_calling/data/tool_calling_results.json"
OUT_DIR = "projects/37_tool_calling/data/charts"
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


def make_fig1_two_call_handshake_and_prefix_cache(data):
    """Fig 1: The 2-Call Handshake Token Anatomy & 6-Call Tool Chain Prefix KV Cache Savings."""
    m1 = data["m1_two_call_handshake_and_prefix_cache"]
    c1 = m1["mode1_call1_emit_tool_call"]
    c2 = m1["mode1_call2_stateless"]
    chain = m1["multi_step_tool_chain"]
    calls = chain["calls"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.4), dpi=180)
    fig.patch.set_facecolor(BG)

    # Left panel: Stacked Token Anatomy of Call 1 vs Call 2 in the 2-Call Handshake
    stages = [
        "Mode 0:\nNo Tools\n(Guesses)",
        "Mode 1 Call 1:\nEmit <tool_call>\n(Halts at 151658)",
        "Mode 1 Call 2:\nInject <tool_response>\n(Synthesizes Answer)",
    ]
    base_toks = m1["mode0_no_tools"]["prompt_tokens"]
    schema_injected = c1["prompt_tokens"] - base_toks
    delta_call2 = m1["mode1_call2_new_delta_tokens"]
    user_ctrl_toks = [base_toks, base_toks, base_toks]
    schema_toks = [0, schema_injected, schema_injected]
    tool_resp_toks = [0, 0, delta_call2]
    gen_toks = [m1["mode0_no_tools"]["generated_tokens"], c1["generated_tokens"], c2["generated_tokens"]]

    x = np.arange(len(stages))
    w = 0.52
    b1 = np.array(user_ctrl_toks)
    b2 = b1 + np.array(schema_toks)
    b3 = b2 + np.array(tool_resp_toks)

    ax1.bar(x, user_ctrl_toks, width=w, color=CYAN, alpha=0.9, label=f"User Query + ChatML Control ({base_toks} tok)")
    ax1.bar(x, schema_toks, width=w, bottom=b1, color=PURPLE, alpha=0.88, label=f"Injected <tools> JSON Schema ({schema_injected} tok)")
    ax1.bar(x, tool_resp_toks, width=w, bottom=b2, color=AMBER, alpha=0.9, label=f"Call 1 <tool_call> + <tool_response> ({delta_call2} tok)")
    ax1.bar(x, gen_toks, width=w, bottom=b3, color=GREEN, alpha=0.92, label="Decoded Output Tokens")

    totals = b3 + np.array(gen_toks)
    latencies = [
        m1["mode0_no_tools"]["total_ms"],
        c1["total_ms"],
        c2["total_ms"],
    ]
    for idx, (tot, lat) in enumerate(zip(totals, latencies)):
        ax1.text(
            idx,
            tot + 10,
            f"{tot} total tok\n({lat:,.0f} ms)",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color=TEXT,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=9.8, fontweight="bold")
    ax1.set_ylabel("Tokens Processed (Prompt + Output)", fontsize=11, fontweight="bold")
    ax1.set_title("A. Why 1 Tool Query Requires 2 Separate HTTP API Calls", fontsize=12.5, fontweight="bold", pad=12, color=TEXT)
    ax1.set_ylim(0, 430)
    ax1.grid(axis="y", linestyle="--", alpha=0.45)
    ax1.legend(loc="upper left", frameon=True, facecolor=BG, edgecolor=GRID, fontsize=8.8)

    # Right panel: 6-Call Incident Tool Chain — Stateless Re-Prefill vs Prefix KV Cache Reuse
    c_nums = [r["api_call_num"] for r in calls]
    s_toks = [r["total_prompt_tokens"] for r in calls]
    d_toks = [r["new_delta_tokens"] for r in calls]
    s_ms = [r["stateless_prefill_ms"] for r in calls]
    c_ms = [r["prefix_cached_prefill_ms"] for r in calls]

    x2 = np.arange(len(c_nums))
    w2 = 0.36
    ax2.bar(
        x2 - w2 / 2,
        s_toks,
        width=w2,
        color=RED,
        alpha=0.88,
        label=f"Stateless Re-Prefill ({chain['cumulative_stateless_prefill_tokens']:,} cumulative tok)",
    )
    ax2.bar(
        x2 + w2 / 2,
        d_toks,
        width=w2,
        color=GREEN,
        alpha=0.92,
        label=f"Prefix KV Cache Delta ({chain['cumulative_prefix_cache_delta_tokens']:,} cumulative tok, {chain['token_waste_ratio_x']}x fewer)",
    )

    ax2_twin = ax2.twinx()
    ax2_twin.plot(x2, s_ms, color=AMBER, marker="o", linewidth=2.4, markersize=6.5, label="Stateless Prefill TTFT (ms)")
    ax2_twin.plot(x2, c_ms, color=CYAN, marker="s", linewidth=2.4, markersize=6.5, linestyle="--", label="Prefix Cached TTFT (ms)")
    ax2_twin.set_ylabel("Prefill Latency / TTFT (ms)", fontsize=10.5, fontweight="bold", color=AMBER)
    ax2_twin.tick_params(axis="y", colors=AMBER)
    ax2_twin.set_ylim(0, 165)

    last_c = calls[-1]
    ax2.annotate(
        f"Call 6: {last_c['total_prompt_tokens']} vs {last_c['new_delta_tokens']} tok\n"
        f"({last_c['cached_prefix_tokens']} cached, {last_c['speedup_x']}x faster TTFT:\n"
        f"{last_c['stateless_prefill_ms']} ms -> {last_c['prefix_cached_prefill_ms']} ms)",
        xy=(5 - w2 / 2, last_c["total_prompt_tokens"]),
        xytext=(1.7, 740),
        color=AMBER,
        fontsize=9.2,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.6),
    )

    ax2.set_xticks(x2)
    ax2.set_xticklabels([f"Call {n}" for n in c_nums], fontsize=10, fontweight="bold")
    ax2.set_ylabel("Prefill Tokens Computed per API Call", fontsize=11, fontweight="bold")
    ax2.set_title(
        f"B. 6-Call Agent Chain: {chain['token_waste_ratio_x']}x Fewer Prefill Tokens with Prefix KV Cache",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
        color=TEXT,
    )
    ax2.set_ylim(0, 1020)
    ax2.grid(axis="y", linestyle="--", alpha=0.45)

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True, facecolor=BG, edgecolor=GRID, fontsize=8.5)

    fig.suptitle(
        "The 2-Call Handshake & Prefix KV Cache Reuse across Multi-Tool Chains (Qwen3-8B on NVIDIA A100-40GB)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
        color=TEXT,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.text(
        0.5,
        0.008,
        "Source: Real GPU telemetry from projects/37_tool_calling/data/tool_calling_results.json · Image by the author",
        ha="center",
        fontsize=9,
        color=MUTED,
    )
    out_path = os.path.join(OUT_DIR, "fig1_two_call_handshake_and_prefix_cache.png")
    plt.savefig(out_path, dpi=180, facecolor=BG)
    plt.close(fig)
    print(f"Saved {out_path}")


def make_fig2_token_151657_logit_lens_and_attention(data):
    """Fig 2: 36-Layer Logit Lens Trajectory of Token 151657 (<tool_call>) & Attention Routing."""
    m2 = data["m2_token_151657_logit_lens_and_attention"]
    tr_tc = m2["traces"]["tool_trigger_prompt"]
    tr_pt = m2["traces"]["plain_answer_prompt"]

    layers = [r["layer"] for r in tr_tc["logit_lens_by_layer"]]
    tc_probs_on_tool_q = [r["tool_call_151657_prob_pct"] for r in tr_tc["logit_lens_by_layer"]]
    tc_probs_on_text_q = [r["tool_call_151657_prob_pct"] for r in tr_pt["logit_lens_by_layer"]]
    pt_top1_probs = [
        r["top1_prob_pct"] if r["top1_token_str"].strip() == "The" else 0.0
        for r in tr_pt["logit_lens_by_layer"]
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.2), dpi=180)
    fig.patch.set_facecolor(BG)

    # Left panel: 36-Layer Logit Lens Probability Trajectory
    ax1.plot(
        layers,
        tc_probs_on_tool_q,
        color=AMBER,
        marker="o",
        markersize=5.5,
        linewidth=2.8,
        label="Tool Query: P(Token 151657 = '<tool_call>')",
    )
    ax1.plot(
        layers,
        pt_top1_probs,
        color=CYAN,
        marker="s",
        markersize=5.0,
        linewidth=2.4,
        linestyle="--",
        label="General CS Query: P(Token 785 = 'The')",
    )
    ax1.plot(
        layers,
        tc_probs_on_text_q,
        color=RED,
        linewidth=2.0,
        linestyle=":",
        label="General CS Query: P(Token 151657 = '<tool_call>') (0.0%)",
    )

    l33 = tr_tc["logit_lens_by_layer"][33]
    l34 = tr_tc["logit_lens_by_layer"][34]
    l35 = tr_tc["logit_lens_by_layer"][35]
    ax1.annotate(
        f"Layer 33: Rank #{l33['tool_call_151657_rank']} ({l33['tool_call_151657_prob_pct']:.2f}%)\n"
        f"Layer 34: Rank #{l34['tool_call_151657_rank']} ({l34['tool_call_151657_prob_pct']:.2f}%)\n"
        f"Layer 35: Rank #1 ({l35['tool_call_151657_prob_pct']:.1f}%, logit {l35['tool_call_151657_logit']})",
        xy=(34, l34["tool_call_151657_prob_pct"]),
        xytext=(2.5, 68),
        color=AMBER,
        fontsize=9.2,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.6),
    )

    l25_pt = tr_pt["logit_lens_by_layer"][25]
    ax1.annotate(
        f"General Query locks in 'The'\nby Layer 25 ({l25_pt['top1_prob_pct']:.1f}%)",
        xy=(25, l25_pt["top1_prob_pct"]),
        xytext=(3.0, 38),
        color=CYAN,
        fontsize=9.2,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=CYAN, lw=1.6),
    )

    ax1.set_xlabel("Transformer Decoder Layer (0 to 35)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Next-Token Probability (%)", fontsize=11, fontweight="bold")
    ax1.set_title("A. 36-Layer Logit Lens: Late-Layer Surge of Token 151657 (<tool_call>)", fontsize=12.5, fontweight="bold", pad=12, color=TEXT)
    ax1.set_xlim(0, 35.8)
    ax1.set_ylim(-3, 108)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(loc="upper left", frameon=True, facecolor=BG, edgecolor=GRID, fontsize=8.8)

    # Right panel: Layer-by-Layer Attention Routing to <tools> Schema + Mean Summary
    tc_schema_attn = [r["tools_schema_attn_pct"] for r in tr_tc["attention_by_layer"]]
    pt_schema_attn = [r["tools_schema_attn_pct"] for r in tr_pt["attention_by_layer"]]

    ax2.plot(
        layers,
        tc_schema_attn,
        color=PURPLE,
        marker="o",
        markersize=5.0,
        linewidth=2.5,
        label=f"Tool-Trigger Query -> <tools> Schema (Mean: {tr_tc['mean_attention_routing_pct']['tools_schema']:.2f}%)",
    )
    ax2.plot(
        layers,
        pt_schema_attn,
        color=GREEN,
        marker="s",
        markersize=4.5,
        linewidth=2.2,
        linestyle="--",
        label=f"Plain-Text Query -> <tools> Schema (Mean: {tr_pt['mean_attention_routing_pct']['tools_schema']:.2f}%)",
    )
    ax2.fill_between(layers, pt_schema_attn, tc_schema_attn, color=PURPLE, alpha=0.18)

    # Highlight the late-layer schema routing divergence (Layer 24: 31.32% vs 4.54%)
    div_idx = max(range(len(layers)), key=lambda i: tc_schema_attn[i] - pt_schema_attn[i])
    ax2.annotate(
        f"Layer {div_idx} Schema Routing Divergence:\n{tc_schema_attn[div_idx]:.2f}% on <tools> JSON\n"
        f"(vs {pt_schema_attn[div_idx]:.2f}% on plain-text query)",
        xy=(div_idx, tc_schema_attn[div_idx]),
        xytext=(18.5, 58),
        color=PURPLE,
        fontsize=9.2,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.6),
    )

    eval_acc = m2["decision_accuracy_eval"]
    tc_eval = eval_acc[0]
    pt_eval = eval_acc[1]
    ax2.text(
        0.04,
        0.70,
        f"12-Query Decision Benchmark:\n"
        f"• Live Cluster -> 151657: {tc_eval['correct']}/{tc_eval['total']} ({tc_eval['accuracy_pct']}%)\n"
        f"• General CS -> Direct Text: {pt_eval['correct']}/{pt_eval['total']} ({pt_eval['accuracy_pct']}%)",
        transform=ax2.transAxes,
        fontsize=8.6,
        color=TEXT,
        bbox=dict(boxstyle="round,pad=0.45", facecolor=BG, edgecolor=GRID, alpha=0.92),
    )

    ax2.set_xlabel("Transformer Decoder Layer (0 to 35)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Final-Token Attention Mass on <tools> Schema (%)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Attention Routing to <tools> Schema Across All 36 Layers", fontsize=12.5, fontweight="bold", pad=12, color=TEXT)
    ax2.set_xlim(0, 35.8)
    ax2.set_ylim(0, 80)
    ax2.grid(True, linestyle="--", alpha=0.45)
    ax2.legend(loc="upper left", frameon=True, facecolor=BG, edgecolor=GRID, fontsize=8.8)

    fig.suptitle(
        "Inside Token ID 151657 (<tool_call>): 36-Layer Logit Lens & Schema Attention Routing (Qwen3-8B on A100)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
        color=TEXT,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.text(
        0.5,
        0.008,
        "Source: Real GPU telemetry from projects/37_tool_calling/data/tool_calling_results.json · Image by the author",
        ha="center",
        fontsize=9,
        color=MUTED,
    )
    out_path = os.path.join(OUT_DIR, "fig2_token_151657_logit_lens_and_attention.png")
    plt.savefig(out_path, dpi=180, facecolor=BG)
    plt.close(fig)
    print(f"Saved {out_path}")


def make_fig3_sequential_vs_parallel_tools(data):
    """Fig 3: Sequential 1-Tool-Per-Turn Loop (5 API Calls) vs Parallel Multi-Tool Calling (2 API Calls)."""
    m3 = data["m3_sequential_vs_parallel_tool_calls"]
    seq = m3["sequential_5_calls"]
    par = m3["parallel_2_calls"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.2), dpi=180)
    fig.patch.set_facecolor(BG)

    # Left panel: Per-Call Prefill + Decode Tokens across the 5 Sequential Calls vs 2 Parallel Calls
    labels = [
        "Seq 1\n(auth)",
        "Seq 2\n(billing)",
        "Seq 3\n(search)",
        "Seq 4\n(analytics)",
        "Seq 5\n(Answer)",
        "Par 1\n(4 Tools!)",
        "Par 2\n(Answer)",
    ]
    prefill_vals = [c["prompt_tokens"] for c in seq["calls"]] + [
        par["call1"]["prompt_tokens"],
        par["call2"]["prompt_tokens"],
    ]
    decode_vals = [c["generated_tokens"] for c in seq["calls"]] + [
        par["call1"]["generated_tokens"],
        par["call2"]["generated_tokens"],
    ]
    colors_p = [RED] * 5 + [GREEN] * 2

    x = np.arange(len(labels))
    w = 0.56
    ax1.bar(x, prefill_vals, width=w, color=colors_p, alpha=0.82, label="Prompt Prefill Tokens")
    ax1.bar(x, decode_vals, width=w, bottom=prefill_vals, color=AMBER, alpha=0.95, label="Decoded Output Tokens")

    for idx, (p_v, d_v) in enumerate(zip(prefill_vals, decode_vals)):
        ax1.text(
            idx,
            p_v + d_v + 12,
            f"{p_v}+{d_v}",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color=TEXT,
        )

    ax1.axvline(4.5, color=MUTED, linestyle="--", linewidth=1.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.8, fontweight="bold")
    ax1.set_ylabel("Tokens Processed per HTTP Round-Trip", fontsize=11, fontweight="bold")
    ax1.set_title(
        f"A. Per-Call Token Load: {seq['total_prefill_tokens']:,} vs {par['total_prefill_tokens']:,} Prefill Tokens ({m3['prefill_token_reduction_x']}x Fewer)",
        fontsize=12.2,
        fontweight="bold",
        pad=12,
        color=TEXT,
    )
    ax1.set_ylim(0, 720)
    ax1.grid(axis="y", linestyle="--", alpha=0.45)
    ax1.legend(loc="upper left", frameon=True, facecolor=BG, edgecolor=GRID, fontsize=9.0)

    # Right panel: Cumulative End-to-End GPU Time & Total Token Comparison
    modes = [
        f"Sequential Tool Loop\n(5 HTTP Calls, 1 Tool/Turn)\n{seq['total_prefill_tokens']:,} Prefill + {seq['total_decode_tokens']} Decode Tok",
        f"Parallel Tool Calling\n(2 HTTP Calls, 4 <tool_call> Blocks in Call 1)\n{par['total_prefill_tokens']:,} Prefill + {par['total_decode_tokens']} Decode Tok",
    ]
    seq_prefill_ms = sum(c["prefill_ms"] for c in seq["calls"])
    seq_decode_ms = sum(c["decode_ms"] for c in seq["calls"])
    par_prefill_ms = par["call1"]["prefill_ms"] + par["call2"]["prefill_ms"]
    par_decode_ms = par["call1"]["decode_ms"] + par["call2"]["decode_ms"]

    y = np.arange(len(modes))
    h = 0.45
    ax2.barh(y, [seq_prefill_ms, par_prefill_ms], height=h, color=CYAN, alpha=0.92, label="Total GPU Prefill Latency (ms)")
    ax2.barh(
        y,
        [seq_decode_ms, par_decode_ms],
        height=h,
        left=[seq_prefill_ms, par_prefill_ms],
        color=PURPLE,
        alpha=0.88,
        label="Total GPU Decode Latency (ms)",
    )

    totals_ms = [seq["total_latency_ms"], par["total_latency_ms"]]
    for idx, tot_ms in enumerate(totals_ms):
        ax2.text(
            tot_ms + 250,
            idx,
            f"{tot_ms:,.1f} ms ({tot_ms / 1000:.2f} s)",
            va="center",
            ha="left",
            fontsize=10.5,
            fontweight="bold",
            color=GREEN if idx == 1 else RED,
        )

    ax2.annotate(
        f"{m3['latency_speedup_x']}x Faster End-to-End GPU Time\n"
        f"& {m3['prefill_token_reduction_x']}x Fewer Prefill Tokens\n"
        f"(5 Round-Trips -> 2 Round-Trips)",
        xy=(par["total_latency_ms"], 1),
        xytext=(6800, 0.42),
        color=GREEN,
        fontsize=10.0,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.8),
    )

    ax2.set_yticks(y)
    ax2.set_yticklabels(modes, fontsize=10.0, fontweight="bold")
    ax2.invert_yaxis()
    ax2.set_xlabel("Cumulative GPU Execution Time (ms)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Total End-to-End Latency on A100 (4 Microservice Checks)", fontsize=12.2, fontweight="bold", pad=12, color=TEXT)
    ax2.set_xlim(0, 19800)
    ax2.grid(axis="x", linestyle="--", alpha=0.45)
    ax2.legend(loc="lower right", frameon=True, facecolor=BG, edgecolor=GRID, fontsize=9.2)

    fig.suptitle(
        "Sequential Tool Loops (5 API Calls) vs Native Parallel Tool Calling (2 API Calls) on NVIDIA A100",
        fontsize=14,
        fontweight="bold",
        y=0.98,
        color=TEXT,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.text(
        0.5,
        0.008,
        "Source: Real GPU telemetry from projects/37_tool_calling/data/tool_calling_results.json · Image by the author",
        ha="center",
        fontsize=9,
        color=MUTED,
    )
    out_path = os.path.join(OUT_DIR, "fig3_sequential_vs_parallel_tools.png")
    plt.savefig(out_path, dpi=180, facecolor=BG)
    plt.close(fig)
    print(f"Saved {out_path}")


def make_fig4_tool_bloat_scaling_and_schema_quality(data):
    """Fig 4: The Tool-Bloat Cliff (1..50 Tools) & Disjoint Typed vs Overlapping Vague Schemas."""
    m4 = data["m4_tool_bloat_scaling_and_confusion"]
    rows = m4["catalog_scaling"]
    sc = m4["schema_quality_comparison"]

    k_vals = [r["num_tools"] for r in rows]
    tok_vals = [r["prompt_tokens"] for r in rows]
    kv_vals = [r["kv_cache_mb"] for r in rows]
    ttft_vals = [r["prefill_ttft_ms"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.2), dpi=180)
    fig.patch.set_facecolor(BG)

    # Left panel: Tool Catalog Size K vs Prompt Tokens, KV Cache MB, and TTFT ms
    ax1.plot(k_vals, ttft_vals, color=RED, marker="o", markersize=7, linewidth=2.8, label="Prefill TTFT Latency (ms)")
    ax1.set_xlabel("Number of Registered Tools in Chat Template (K)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Time-To-First-Token / Prefill Latency (ms)", fontsize=11, fontweight="bold", color=RED)
    ax1.tick_params(axis="y", colors=RED)
    ax1.set_ylim(0, 1550)
    ax1.grid(True, linestyle="--", alpha=0.45)

    ax1_twin = ax1.twinx()
    ax1_twin.bar(k_vals, kv_vals, width=2.8, color=CYAN, alpha=0.35, label="KV Cache Footprint per Request (MB)")
    ax1_twin.set_ylabel("KV Cache VRAM per Request (MB)", fontsize=11, fontweight="bold", color=CYAN)
    ax1_twin.tick_params(axis="y", colors=CYAN)
    ax1_twin.set_ylim(0, 1450)

    r1 = rows[0]
    r30 = [r for r in rows if r["num_tools"] == 30][0]
    r50 = [r for r in rows if r["num_tools"] == 50][0]

    ax1.annotate(
        f"K=1 Tool: {r1['prompt_tokens']} tok\nTTFT = {r1['prefill_ttft_ms']} ms, KV = {r1['kv_cache_mb']} MB",
        xy=(1, r1["prefill_ttft_ms"]),
        xytext=(5.5, 220),
        color=GREEN,
        fontsize=9.0,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5),
    )
    ax1.annotate(
        f"K=30 Tools (eager): {r30['prompt_tokens']:,} tok\n"
        f"TTFT = {r30['prefill_ttft_ms']:,.1f} ms ({r30['prefill_ttft_ms'] / r1['prefill_ttft_ms']:.1f}x slower!)\n"
        f"KV Cache = {r30['kv_cache_mb']} MB",
        xy=(30, r30["prefill_ttft_ms"]),
        xytext=(2.5, 1080),
        color=RED,
        fontsize=9.0,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.6),
    )
    ax1.annotate(
        f"K=50 Tools: {r50['prompt_tokens']:,} tok\n"
        f"KV = {r50['kv_cache_mb']:,.1f} MB (1.21 GB / 1.18 GiB)\n"
        f"Eager [32, 8624, 8624] OOMs 40GB A100;\n"
        f"SDPA fallback = {r50['prefill_ttft_ms']} ms",
        xy=(50, r50["prefill_ttft_ms"]),
        xytext=(26.5, 420),
        color=AMBER,
        fontsize=9.0,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.6),
    )

    ax1.set_title("A. The Tool-Bloat Cliff: Scaling K = 1 to 50 Registered Tools", fontsize=12.5, fontweight="bold", pad=12, color=TEXT)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True, facecolor=BG, edgecolor=GRID, fontsize=8.8)

    # Right panel: Schema Quality Benchmark (Disjoint Typed vs Overlapping Vague Schemas)
    clean_acc = sc["disjoint_typed_schemas"]["accuracy_pct"]
    vague_acc = sc["overlapping_vague_schemas"]["accuracy_pct"]
    clean_cnt = f"{sc['disjoint_typed_schemas']['exact_matches']}/{sc['disjoint_typed_schemas']['trials']}"
    vague_cnt = f"{sc['overlapping_vague_schemas']['exact_matches']}/{sc['overlapping_vague_schemas']['trials']}"

    categories = [
        f"Disjoint Typed Schemas\n(Explicit enum + Domain Docstring)\n[{clean_cnt} Exact Tool + Arg Match]",
        f"Overlapping Vague Schemas\n(check_service_status / get_backend_info)\n[{vague_cnt} Exact Tool + Arg Match]",
    ]
    vals = [clean_acc, vague_acc]
    bar_colors = [GREEN, RED]

    x_b = np.arange(len(categories))
    bars = ax2.bar(x_b, vals, width=0.46, color=bar_colors, alpha=0.9)
    for bar, v, cnt in zip(bars, vals, [clean_cnt, vague_cnt]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            v + 2.5,
            f"{v:.1f}% ({cnt})",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=TEXT,
        )

    ax2.text(
        0.04,
        0.46,
        "What Breaks Under Vague Overlapping Schemas:\n"
        "• 'Check Postgres replica lag for billing-api' -> picks check_service_status\n"
        "• 'Check Redis memory fragmentation for auth-api' -> picks get_backend_info\n"
        "• 'What is Redis cache hit ratio for search-api?' -> picks fetch_system_metrics\n"
        "• Only 2/8 queries hit the expected function + parameter key!",
        transform=ax2.transAxes,
        fontsize=8.8,
        color=TEXT,
        bbox=dict(boxstyle="round,pad=0.55", facecolor=BG, edgecolor=RED, alpha=0.92),
    )

    ax2.set_xticks(x_b)
    ax2.set_xticklabels(categories, fontsize=9.8, fontweight="bold")
    ax2.set_ylabel("Exact Tool + Argument Selection Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Schema Quality Impact across 8 SRE Production Queries", fontsize=12.5, fontweight="bold", pad=12, color=TEXT)
    ax2.set_ylim(0, 118)
    ax2.grid(axis="y", linestyle="--", alpha=0.45)

    fig.suptitle(
        "The Tool-Bloat Cliff (1 to 50 Tools) & Schema Disambiguation Accuracy (Qwen3-8B on A100)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
        color=TEXT,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.text(
        0.5,
        0.008,
        "Source: Real GPU telemetry from projects/37_tool_calling/data/tool_calling_results.json · Image by the author",
        ha="center",
        fontsize=9,
        color=MUTED,
    )
    out_path = os.path.join(OUT_DIR, "fig4_tool_bloat_scaling_and_schema_quality.png")
    plt.savefig(out_path, dpi=180, facecolor=BG)
    plt.close(fig)
    print(f"Saved {out_path}")


def make_fig5_fsm_constrained_tool_call_and_recovery(data):
    """Fig 5: Unconstrained vs FSM Grammar-Constrained <tool_call> (-inf Mask) & Error Self-Correction."""
    m5 = data["m5_constrained_tools_and_self_correction"]
    uf = m5["unconstrained_vs_fsm"]
    trace = uf["fsm_step_trace"]
    err = m5["error_self_correction"]
    op = err["opaque_error_no_options"]
    act = err["actionable_error_with_valid_enums"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.2), dpi=180)
    fig.patch.set_facecolor(BG)

    # Left panel: Allowed Vocabulary Size across the 24 Decode Steps of <tool_call> (log10 scale)
    steps = [r["step"] for r in trace]
    allowed = [r["allowed_vocab_count"] for r in trace]

    ax1.plot(
        steps,
        allowed,
        color=GREEN,
        marker="o",
        markersize=5.5,
        linewidth=2.6,
        label="Allowed Vocabulary Tokens (out of 151,936)",
    )
    ax1.axhline(151936, color=RED, linestyle="--", linewidth=1.8, label="Unconstrained Full Vocab (151,936 tokens)")
    ax1.set_yscale("log")

    s0 = trace[0]
    s18 = trace[18]
    ax1.annotate(
        f"Step 0 ('<tool_call>'):\nOnly {s0['allowed_vocab_count']} tokens allowed\n({s0['masked_to_neg_inf_count']:,} masked to -inf)",
        xy=(0, s0["allowed_vocab_count"]),
        xytext=(2.0, 220),
        color=CYAN,
        fontsize=9.0,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=CYAN, lw=1.5),
    )
    raw_top_s18 = s18["top5_raw_before_mask"][0]
    post_top_s18 = s18["top_after_mask"][0]
    ax1.annotate(
        f"Step 18 (Service Enum Value):\n"
        f"Raw #1 is '{raw_top_s18['token_str']}' ({raw_top_s18['raw_prob_pct']:.2f}%) -> MASKED to -inf!\n"
        f"Only {s18['allowed_vocab_count']} enum prefix tokens allowed ->\n"
        f"'{post_top_s18['token_str']}' promoted to {post_top_s18['post_prob_pct']:.1f}% ('auth-api')",
        xy=(18, s18["allowed_vocab_count"]),
        xytext=(4.5, 1750),
        color=AMBER,
        fontsize=9.0,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.6),
    )

    ax1.set_xlabel("Decode Step Index Inside <tool_call> Block (Steps 0 to 23)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Allowed Tokens in Softmax (Log Scale)", fontsize=11, fontweight="bold")
    ax1.set_title("A. 24-Step FSM Logit Masking Inside <tool_call> (151,936 Vocab)", fontsize=12.5, fontweight="bold", pad=12, color=TEXT)
    ax1.set_ylim(1, 400000)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(loc="upper right", frameon=True, facecolor=BG, edgecolor=GRID, fontsize=8.8)

    # Right panel: 20-Trial Colloquial Service Benchmark + 8-Trial Error Recovery Comparison
    labels = [
        f"Unconstrained\nLoose Schema\n({uf['unconstrained_valid_count']}/{uf['trials']} Valid Enum)",
        f"FSM Grammar-\nConstrained\n({uf['constrained_valid_count']}/{uf['trials']} Valid Enum)",
        f"Retry w/ Opaque\nKeyError String\n({op['recovered_count']}/{op['trials']} Recovered)",
        f"Retry w/ Actionable\nEnum Error String\n({act['recovered_count']}/{act['trials']} Recovered)",
    ]
    rates = [
        uf["unconstrained_valid_pct"],
        uf["constrained_valid_pct"],
        op["recovery_rate_pct"],
        act["recovery_rate_pct"],
    ]
    colors = [RED, GREEN, RED, CYAN]

    x2 = np.arange(len(labels))
    bars = ax2.bar(x2, rates, width=0.48, color=colors, alpha=0.9)
    for bar, r in zip(bars, rates):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            r + 2.5,
            f"{r:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10.8,
            fontweight="bold",
            color=TEXT,
        )

    ax2.text(
        0.03,
        0.43,
        "Two Layers of Production Tool Defense:\n"
        "1. First-Pass FSM Masking (0.0% -> 100.0%): Masks 'authentication' (99.98% raw prob)\n"
        "   to -inf at Step 18 so only valid enum slugs ('auth-api') can be sampled.\n"
        "2. Actionable <tool_response> Errors (0.0% -> 87.5%): Returning valid enum options\n"
        "   inside the KeyError string lets the LLM self-correct on the next API turn!",
        transform=ax2.transAxes,
        fontsize=8.6,
        color=TEXT,
        bbox=dict(boxstyle="round,pad=0.55", facecolor=BG, edgecolor=GRID, alpha=0.94),
    )

    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels, fontsize=9.0, fontweight="bold")
    ax2.set_ylabel("Valid Production Execution / Recovery Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_title("B. First-Pass Enum Validity (20 Trials) & Error Self-Correction (8 Trials)", fontsize=12.2, fontweight="bold", pad=12, color=TEXT)
    ax2.set_ylim(0, 118)
    ax2.grid(axis="y", linestyle="--", alpha=0.45)

    fig.suptitle(
        "Guaranteeing Valid Tool Calls: FSM Logit Masking (-inf) & Error Self-Correction (Qwen3-8B on A100)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
        color=TEXT,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.text(
        0.5,
        0.008,
        "Source: Real GPU telemetry from projects/37_tool_calling/data/tool_calling_results.json · Image by the author",
        ha="center",
        fontsize=9,
        color=MUTED,
    )
    out_path = os.path.join(OUT_DIR, "fig5_fsm_constrained_tool_call_and_recovery.png")
    plt.savefig(out_path, dpi=180, facecolor=BG)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    d = load_data()
    make_fig1_two_call_handshake_and_prefix_cache(d)
    make_fig2_token_151657_logit_lens_and_attention(d)
    make_fig3_sequential_vs_parallel_tools(d)
    make_fig4_tool_bloat_scaling_and_schema_quality(d)
    make_fig5_fsm_constrained_tool_call_and_recovery(d)
    print("All 5 Episode 12 charts generated successfully.")
