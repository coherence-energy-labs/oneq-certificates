"""Independent exhaustive small-graph controls for native metric-dual export."""
import argparse
import copy
import json
from pathlib import Path
import random
from metric_dual_check import check
from native_wrapper import NativeDual

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--library",type=Path,default=None)
args=parser.parse_args()

rng=random.Random(20261103)
out={"seed":20261103,"graphs":0,"feasible_checked":0,"infeasible_refused":0,"blossom_certificates":0,"forgeries_refused":0}
for case in range(160):
    n=rng.randrange(2,10)
    edges=[(u,v,2*rng.randrange(0,11)) for u in range(n) for v in range(u+1,n) if rng.random()<.35]
    edges += [(u,-1,2*rng.randrange(0,11)) for u in range(n) if rng.random()<.4]
    # Independent exact error-subset optimization by syndrome dynamic programming.
    best=[None]*(1<<n);best[0]=0
    for u,v,w in edges:
        mask=(1<<u)^((1<<v) if v>=0 else 0)
        old=best[:]
        for s,c in enumerate(old):
            if c is not None and (best[s^mask] is None or c+w<best[s^mask]):best[s^mask]=c+w
    with NativeDual(n,edges,args.library) as decoder:
        for mask in rng.sample(range(1<<n),min(20,1<<n)):
            syndrome=[u for u in range(n) if mask>>u&1]
            if best[mask] is None:
                try:decoder.decode(syndrome)
                except ValueError:out["infeasible_refused"]+=1
                else:raise AssertionError("native accepted impossible syndrome")
                continue
            proof=decoder.decode(syndrome)
            result=check(n,edges,syndrome,proof)
            assert result["accepted"],(case,edges,syndrome,proof,result)
            assert result["weight"]==best[mask],(case,result,best[mask])
            out["feasible_checked"]+=1
            out["blossom_certificates"]+=bool(result["blossom_terms"])
            if proof["dual"]:
                forged=copy.deepcopy(proof);forged["dual"][0]["value"]+=1
                assert not check(n,edges,syndrome,forged)["accepted"]
                forged=copy.deepcopy(proof);forged["pairs"]+=forged["pairs"][:1]
                assert not check(n,edges,syndrome,forged)["accepted"]
                forged=copy.deepcopy(proof);forged["dual"][0]["value"]=0.9
                assert not check(n,edges,syndrome,forged)["accepted"]
                out["forgeries_refused"]+=3
    out["graphs"]+=1
out["status"]="PASS"
Path(__file__).with_name("validation.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
