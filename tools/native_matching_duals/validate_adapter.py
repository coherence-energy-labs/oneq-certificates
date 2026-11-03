"""Portable interface, model binding and original-weight epsilon controls."""
import argparse
import copy
from fractions import Fraction
import itertools
import json
from pathlib import Path
from native_metric_producer import NativeMetricProducer
from native_metric_verify import verify_document

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--library",type=Path,default=None)
args=parser.parse_args()

out={"valid":0,"refused":0,"epsilon_refusals":0}
graphs=[[(0,1,1.),(1,2,1.),(0,2,1.),(0,-1,10.),(1,-1,10.),(2,-1,10.)],[(0,1,1.123),(1,2,2.777),(0,2,3.129),(0,-1,3.331),(1,-1,5.213),(2,-1,1.917)]]
for edges in graphs:
    p=NativeMetricProducer(edges,library_path=args.library)
    for mask in range(8):
        syn=[u for u in range(3) if mask>>u&1]
        d=p.produce(syn)
        checked=verify_document(d,edges,syn,epsilon=Fraction(1,10000))
        assert checked["accepted"] and checked["bound_valid"]
        out["valid"]+=1
        zero=verify_document(d,edges,syn,epsilon=0)
        out["epsilon_refusals"]+=not zero["accepted"]
        assert zero["bound_valid"]
        # Independently enumerate original graph error subsets: interval covers optimum.
        optimum=None
        for bits in itertools.product((0,1),repeat=len(edges)):
            syndrome=set();cost=Fraction()
            for bit,(u,v,w) in zip(bits,edges):
                if not bit:continue
                cost+=Fraction(w)
                for t in (u,v):
                    if t==-1:continue
                    if t in syndrome:syndrome.remove(t)
                    else:syndrome.add(t)
            if syndrome==set(syn) and (optimum is None or cost<optimum):optimum=cost
        assert Fraction(*checked["lower_bound"])<=optimum<=Fraction(*checked["primal"])
        wrong=copy.deepcopy(d);wrong["model_sha256"]="0"*64
        assert not verify_document(wrong,edges,syn)["accepted"];out["refused"]+=1
        changed=edges[:];u,v,w=changed[0];changed[0]=(u,v,w+.01)
        assert not verify_document(d,changed,syn)["accepted"];out["refused"]+=1
        wrong=copy.deepcopy(d);wrong["syndrome"]=syn+[3]
        assert not verify_document(wrong,edges,syn)["accepted"];out["refused"]+=1
        if syn:
            wrong=copy.deepcopy(d);wrong["correction"]=[]
            assert not verify_document(wrong,edges,syn)["accepted"];out["refused"]+=1
    p.close()
    try:p.produce([])
    except ValueError:out["refused"]+=1
    else:raise AssertionError("closed producer accepted")
assert out["epsilon_refusals"]>0
out["status"]="PASS"
Path(__file__).with_name("adapter_validation.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
