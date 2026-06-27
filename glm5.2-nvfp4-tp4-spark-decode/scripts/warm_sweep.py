#!/usr/bin/env python3
# WARM prefill+decode sweep: run each depth TWICE with different salts (cache-cold, kernel-warm
# on the 2nd). Report the warm pass. Fixes the cold-JIT contamination in the first sweep.
import json, time, http.client
HOST,PORT,MODEL="127.0.0.1",8000,"glm52"
RATIO=3.6  # tokens per integer-word, measured
GEN=48
# target REAL prompt tokens (kept < 65536-GEN)
TARGETS=[500,2000,5000,10000,20000,35000,50000,62000]

def run(real, salt, gent):
    nwords=max(1,int(real/RATIO))
    prompt=" ".join(f"{salt}{i}" for i in range(nwords))
    body=json.dumps({"model":MODEL,"prompt":prompt,"max_tokens":gent,"temperature":0,
        "stream":True,"stream_options":{"include_usage":True}})
    c=http.client.HTTPConnection(HOST,PORT,timeout=900); t0=time.time()
    c.request("POST","/v1/completions",body,{"Content-Type":"application/json"})
    r=c.getresponse(); st=r.status; tf=tl=None; n=0; u=None; buf=b""; eb=b""
    for raw in r:
        buf+=raw
        if st!=200: eb+=raw; continue
        while b"\n" in buf:
            ln,buf=buf.split(b"\n",1); ln=ln.strip()
            if not ln.startswith(b"data:"): continue
            d=ln[5:].strip()
            if d==b"[DONE]": continue
            try: o=json.loads(d)
            except: continue
            if o.get("usage"): u=o["usage"]
            ch=o.get("choices") or []
            if ch and ch[0].get("text"):
                now=time.time()
                if tf is None: tf=now
                tl=now; n+=1
    c.close()
    if st!=200: return {"status":st,"err":eb[:140].decode("utf-8","ignore")}
    u=u or {}; pt=u.get("prompt_tokens"); ct=u.get("completion_tokens",n)
    ttft=(tf-t0) if tf else None; dec=(tl-tf) if (tf and tl and tl>tf) else None
    return {"prompt_tokens":pt,"gen":ct,"ttft_s":round(ttft,2) if ttft else None,
        "prefill_tps":round(pt/ttft,1) if (pt and ttft) else None,
        "decode_tps":round((ct-1)/dec,2) if (ct and ct>1 and dec) else None}

for tgt in TARGETS:
    # warm the shape (1st), then measure (2nd) with a different salt -> kernel warm, cache cold
    a=run(tgt,f"a{tgt}",GEN)
    b=run(tgt,f"b{tgt}",GEN)
    print(json.dumps({"target":tgt,"cold":a,"warm":b}),flush=True)
