import torch, glob, os
for p in sorted(glob.glob(os.path.expanduser("DIR/experts_L*.pt"))):
    if p.endswith(".ok"): continue
    d = torch.load(p, map_location="cpu")
    ch = False
    for k in list(d.keys()):
        if d[k].dtype != torch.uint8:
            d[k] = d[k].view(torch.uint8); ch = True
    if ch:
        torch.save(d, p + ".fix"); os.rename(p + ".fix", p)
        print("fixed", os.path.basename(p), flush=True)
print("DTYPE_FIX_DONE")