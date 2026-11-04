"""M1 re-check (4 Nov 2026): correct prefix-KV reuse.
Original M1 fed ids[:, prev_len:] although each turn's generation prompt ends with an empty
<think></think> block that the chat template drops once the turn becomes history, so the cached
tokens were not a true prefix. Here the reusable prefix = longest common prefix (LCP) of the previous
turn's prompt and this turn's prompt; the cache is cropped to LCP and only the rest is prefilled.
Each timing = median of 5 runs.
Note: the published m1_prefix_recheck.json was produced before the system message was added to c1 (it printed 36);
with the system message, as in measure_api_call.py, Call 1 is 55 tokens. Same model/dtype/attention as measure_api_call.py.
"""
import json, time, copy, statistics, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
MODEL_ID="Qwen/Qwen3-8B"; dev="cuda"; R=5
tok=AutoTokenizer.from_pretrained(MODEL_ID)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.bfloat16,device_map="auto",attn_implementation="eager").eval()
def sync(): torch.cuda.synchronize()
def crop_to(kv, keep):
    cur = kv.get_seq_length()
    if cur > keep:
        try: kv.crop(keep - cur)
        except ValueError: kv.crop(keep)
    assert kv.get_seq_length() == keep, (kv.get_seq_length(), keep)
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


c1=[{"role":"system","content":"You are a concise production database assistant. Answer in one short sentence."},{"role":"user","content":"My name is Arun and our production Postgres database runs on port 5439. Acknowledge briefly."}]
call1_tokens=int(tok(tok.apply_chat_template(c1,tokenize=False,add_generation_prompt=True,enable_thinking=False),return_tensors="pt").input_ids.shape[1])
history=[system_msg]; prompts=[]; replies=[]
for u in user_turns:
    history.append({"role":"user","content":u})
    ids=tok(tok.apply_chat_template(history,tokenize=False,add_generation_prompt=True,enable_thinking=False),return_tensors="pt").input_ids.to(dev)
    with torch.no_grad(): out=model.generate(ids,max_new_tokens=24,do_sample=False,pad_token_id=tok.eos_token_id)
    a=tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True).strip()
    replies.append(a); history.append({"role":"assistant","content":a}); prompts.append(ids)
with torch.no_grad(): model(prompts[0][:,:32]); sync()
def timed(fn):
    ts=[]
    for _ in range(R):
        sync(); t0=time.perf_counter(); r=fn(); sync(); ts.append((time.perf_counter()-t0)*1000)
    return statistics.median(ts), ts, r
rows=[]; prev_kv=None
for k,ids in enumerate(prompts):
    n=ids.shape[1]
    def full():
        with torch.no_grad(): return model(ids,use_cache=True)
    st_ms,st_all,full_out=timed(full)
    if k==0:
        lcp=0
        def cached(): 
            with torch.no_grad(): return model(ids,use_cache=True)
    else:
        p=prompts[k-1][0].tolist(); c=ids[0].tolist()
        lcp=next((i for i,(x,y) in enumerate(zip(p,c)) if x!=y),min(len(p),len(c)))
        def cached():
            kv=copy.deepcopy(prev_kv); crop_to(kv, lcp)
            with torch.no_grad(): return model(ids[:,lcp:],past_key_values=kv,use_cache=True)
    # time without the deepcopy: build copies first
    if k>0:
        copies=[]
        for _ in range(R):
            kv=copy.deepcopy(prev_kv); crop_to(kv, lcp); copies.append(kv)
        it=iter(copies)
        def cached2():
            with torch.no_grad(): return model(ids[:,lcp:],past_key_values=next(it),use_cache=True)
        ca_ms,ca_all,ca_out=timed(cached2)
    else:
        ca_ms,ca_all,ca_out=timed(cached)
    diff=(full_out.logits[0,-1].float()-ca_out.logits[0,-1].float()).abs().max().item()
    same=int(full_out.logits[0,-1].argmax())==int(ca_out.logits[0,-1].argmax())
    prev_kv=full_out.past_key_values
    rows.append({"turn":k+1,"context_tokens":int(n),"reused_prefix_tokens":int(lcp),"delta_tokens":int(n-lcp),
      "stateless_ttft_ms_median":round(st_ms,2),"cached_ttft_ms_median":round(ca_ms,2),
      "stateless_runs_ms":[round(x,2) for x in st_all],"cached_runs_ms":[round(x,2) for x in ca_all],
      "last_logit_max_abs_diff":round(diff,4),"argmax_match":same})
    print(rows[-1],flush=True)
res={"call1_tokens":call1_tokens,"turns":rows,
 "total_stateless_tokens":sum(r["context_tokens"] for r in rows),"total_cached_tokens":sum(r["delta_tokens"] for r in rows),
 "total_stateless_ms":round(sum(r["stateless_ttft_ms_median"] for r in rows),2),"total_cached_ms":round(sum(r["cached_ttft_ms_median"] for r in rows),2),
 "gpu":torch.cuda.get_device_name(0),"torch":torch.__version__,"replies":replies}
print("===M1FIX===");print(json.dumps(res));print("===END===")
