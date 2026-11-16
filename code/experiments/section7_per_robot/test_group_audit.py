import sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "."); sys.path.insert(0, "eval")
from collections import Counter
from floor_geometry import GROUPS, group_of
ck = torch.load("runs/group_classifier.pt", map_location="cpu")
import group_classifier as gc, pathnet
pathnet.FEATURES = "v3"
test = pathnet.load_split("test", with_labels=False)
X = (gc.image_features(test).cpu() - ck["mean"].cpu()) / ck["std"].cpu()
clf = torch.nn.Linear(X.shape[1], len(GROUPS)); clf.load_state_dict({k: v.cpu() for k, v in ck["state"].items()})
with torch.no_grad(): p = clf(X).softmax(-1)
cls = p.argmax(1).numpy(); conf = p.max(1).values.numpy()
print("classifier counts:", {GROUPS[k]: int((cls == k).sum()) for k in range(len(GROUPS))})
print("classifier confidence on test: <0.9 for", f"{np.mean(conf < 0.9):.1%}", "| <0.6 for", f"{np.mean(conf < 0.6):.1%}")
# nearest labelled photos (train + val_seen + val_unseen) by SigLIP2 + DINOv3 vectors
def vec(f): return torch.cat([F.normalize(f["sig_pooled"][f["image_index"]].float(), dim=-1), F.normalize(f["dino_cls"][f["image_index"]].float(), dim=-1)], 1)
labs, vs = [], []
for split in ("train", "val_seen", "val_unseen"):
    f = pathnet.load_split(split, with_labels=False)
    eps = [json.loads(l) for l in open(f"data/lcvn/{split}.jsonl")]
    keep = [i for i, e in enumerate(eps) if e["episode"].endswith("_2")]
    vs.append(vec(f)[keep]); labs += [GROUPS.index(group_of(eps[i]["trajectory"], split)) for i in keep]
V = torch.cat(vs); L = torch.tensor(labs)
tv = vec(test)
knn = []
for i in range(0, len(tv), 2048):
    s = tv[i:i + 2048] @ V.T
    nb = s.topk(10, dim=1).indices
    knn.append(torch.mode(L[nb], dim=1).values)
knn = torch.cat(knn).numpy()
print("10-NN counts:      ", {GROUPS[k]: int((knn == k).sum()) for k in range(len(GROUPS))})
agree = cls == knn
print(f"agreement {agree.mean():.1%}; disagreements (classifier -> 10-NN):", Counter((GROUPS[a], GROUPS[b]) for a, b in zip(cls[~agree], knn[~agree])).most_common(8))
print("classifier confidence where they disagree: median", round(float(np.median(conf[~agree])), 2), "| where they agree:", round(float(np.median(conf[agree])), 2))
np.save("runs/test_group_audit.npy", dict(cls=cls, knn=knn, conf=conf), allow_pickle=True)
