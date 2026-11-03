"""Portable native-dual receipt verifier. Python standard library only."""
from fractions import Fraction
import hashlib
import json
import math
if __package__:
    from .metric_dual_check import check
    from .reconcile_original import reconcile
else:
    from metric_dual_check import check
    from reconcile_original import reconcile

SCHEMA="oneq.native-metric-receipt.v1"

def model_identity(edges):
    canonical=[];seen=set()
    for row in edges:
        if len(row)!=3:raise ValueError("edge width")
        u,v,w=row
        if type(u) is not int or type(v) is not int or u< -1 or v< -1 or u==v:raise ValueError("endpoint domain")
        if u==-1:u,v=v,u
        if v>=0:u,v=min(u,v),max(u,v)
        if type(w) not in (int,float) or not math.isfinite(w) or w<0:raise ValueError("weight domain")
        w=float(w)
        if (u,v) in seen:raise ValueError("parallel graph edge")
        seen.add((u,v));canonical.append([u,v,w.hex()])
    canonical.sort()
    return hashlib.sha256(json.dumps(canonical,separators=(",",":")).encode("ascii")).hexdigest()

def verify_document(document,edges,syndrome,*,epsilon=Fraction(0)):
    """Returns portable proof validity and separately whether its gap meets budget.

    accepted is false when a sound lower bound misses epsilon; bound_valid stays
    true. This never promotes quantized optimality into original-weight exactness.
    """
    try:
        if not isinstance(document,dict):raise ValueError("receipt must be an object")
        if isinstance(epsilon,bool):raise ValueError("epsilon domain")
        epsilon=Fraction(epsilon)
        if epsilon<0:raise ValueError("negative epsilon")
        if document.get("schema")!=SCHEMA:raise ValueError("receipt schema")
        if document.get("model_sha256")!=model_identity(edges):raise ValueError("model binding mismatch")
        if document.get("syndrome")!=list(syndrome):raise ValueError("syndrome binding mismatch")
        if any(type(x) is not int for x in syndrome):raise ValueError("syndrome domain")
        qedges=document["integer_edges"]
        original_topology={tuple(sorted((u,v))) for u,v,_ in edges}
        if {tuple(sorted((u,v))) for u,v,_ in qedges}!=original_topology:raise ValueError("integer graph topology mismatch")
        nodes=max(max(u,v) for u,v,_ in edges)+1
        result=check(nodes,qedges,syndrome,document["native_proof"])
        if not result["accepted"]:raise ValueError("native metric proof: "+result["reason"])
        normalized=[(v,u,w) if u==-1 else (u,v,w) for u,v,w in edges]
        bounds=reconcile(nodes,normalized,syndrome,document["native_proof"],result,document["correction"])
        accepted=Fraction(*bounds["epsilon"])<=epsilon
        return {"accepted":accepted,"bound_valid":True,"reason":"exact original-model bound checked" if accepted else "valid original-model bound exceeds epsilon budget","claim":"EXACT_OPTIMAL" if bounds["exact_optimal"] else "EPSILON_BOUND","model_sha256":document["model_sha256"],"correction":document["correction"],**bounds}
    except (ValueError,TypeError,KeyError,IndexError,OverflowError,ZeroDivisionError) as exc:
        return {"accepted":False,"bound_valid":False,"reason":str(exc)}
