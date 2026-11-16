import json, sys, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "eval")
from score import load
from path_labels import start_frame
from post_turn import net_turn
run = sys.argv[1]
eps = load("data/lcvn/val_seen.jsonl")
idx = np.load(f"runs/{run}/val_index.npy"); ours = np.load(f"runs/{run}/val_routes.npy")
parse = {json.loads(l)["episode"]: json.loads(l) for l in open("features/val_seen_parsed2.jsonl")}
rows = []
for k, i in enumerate(idx):
    e = eps[i]
    if not e["episode"].endswith("_2"): continue
    t = [s for s in parse[e["episode"]]["parsed"]["segments"] if s["move"] in ("turn", "curve")]
    side = t[0]["side"] if t else "none"
    if side not in ("left", "right"): continue
    sg = 1.0 if side == "right" else -1.0
    rows.append((sg * net_turn(start_frame(e["position"], e["yaw"][0])[1:]), sg * net_turn(ours[k])))
r = np.array(rows); big = np.abs(r[:, 0]) > np.radians(30)
print(f"{run}: {len(r)} routes whose parse turns left/right; true turns the parse side (30+ deg) {np.mean(r[big,0] > 0):.0%} of {big.sum()}; "
      f"true against parse {np.mean(r[:,0] < -0.1):.0%}; ours against parse {np.mean(r[:,1] < -0.1):.0%}")
