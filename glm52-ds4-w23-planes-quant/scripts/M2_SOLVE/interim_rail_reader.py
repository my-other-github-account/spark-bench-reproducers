import glob, torch, os, math

# Correct key structure: teacher {'idx','logprob'} (top-8192 ids + logprobs); cand {'q_lp_at_ref','q_argmax'}
# (cand logprobs gathered at ref ids). Replicate sealed convention: renormalize both on support S, KL(ref||cand),
# pos_cutoff 1024.
cand_dir = "~/missions/DS4_TEACHER/q8192_eval_r6pp_94g"
tdir = "~/missions/DS4_TEACHER/t8192_eval"
kls = []
tops = []
for cp in sorted(glob.glob(cand_dir + "/q8192_win*.pt")):
    w = cp.split("win")[-1].split(".")[0]
    tp = f"{tdir}/t8192_win{w}.pt"
    if not os.path.exists(tp):
        continue
    try:
        c = torch.load(cp, map_location="cpu", weights_only=False)
        t = torch.load(tp, map_location="cpu", weights_only=False)
    except Exception:
        continue
    tlp = t["logprob"].float()   # [P, K] teacher logprobs at top-K ids
    clp = c["q_lp_at_ref"].float()  # [P, K] cand logprobs at same ids
    P = min(tlp.shape[0], clp.shape[0], 1024)
    tlp = tlp[:P]; clp = clp[:P]
    # renormalize on support:
    tn = tlp - torch.logsumexp(tlp, dim=-1, keepdim=True)
    cn = clp - torch.logsumexp(clp, dim=-1, keepdim=True)
    p = tn.exp()
    kl = (p * (tn - cn)).sum(-1).mean().item()
    kls.append(kl)
    # top1 agreement if teacher argmax id == cand argmax:
    if "q_argmax" in c and "idx" in t:
        t_arg = t["idx"][:P, 0]
        tops.append((c["q_argmax"][:P] == t_arg).float().mean().item())
m = sum(kls)/len(kls)
sd = math.sqrt(sum((x-m)**2 for x in kls)/max(1,len(kls)-1)/len(kls))
print(f"INTERIM 94G per-proj: mean KLD {m:.6f} +/- {sd:.6f} over {len(kls)} windows")
if tops:
    print(f"top1_agree: {sum(tops)/len(tops):.6f}")
