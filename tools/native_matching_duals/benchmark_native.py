"""Same-syndrome native-dual versus post-hoc ladder benchmark.

Every native proof is checked independently for the DECLARED quantized integer
graph. Original float weights are retained and an exact conservative epsilon
bound is calculated separately; quantized exactness is never relabelled float
exactness. Decoding/dual export/checking, Python overhead, and ladder times are
separate measurements on identical instances and corrections.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import pymatching
import stim
from native_wrapper import NativeDual
from metric_dual_check import check,key
from reconcile_original import reconcile

ap=argparse.ArgumentParser()
ap.add_argument("--oneq",type=Path,required=True)
ap.add_argument("--library",type=Path,default=None)
ap.add_argument("--shots",type=int,default=64)
ap.add_argument("--ladder-shots",type=int,default=12)
ap.add_argument("--output",type=Path,required=True)
a=ap.parse_args()
sys.path.insert(0,str(a.oneq/"src"))
from oneq.certifying_decoder import CertifyingDecoder

def stats(values):
    v=sorted(values)
    return {"n":len(v),"mean":statistics.mean(v),"median":statistics.median(v),"p95":v[min(len(v)-1,math.ceil(.95*len(v))-1)]} if v else None

records=[];cells=[];models=[]
for d,rounds in ((3,3),(5,5),(7,7),(5,25)):
    circuit=stim.Circuit.generated("surface_code:rotated_memory_z",distance=d,rounds=rounds,after_clifford_depolarization=.003,after_reset_flip_probability=.003,before_measure_flip_probability=.003)
    matching=pymatching.Matching.from_detector_error_model(circuit.detector_error_model(decompose_errors=True))
    floating=[(int(u),-1 if v is None else int(v),float(attr["weight"])) for u,v,attr in matching.edges()]
    maximum=max(w for _,_,w in floating)
    scale=1.0 if all(w.is_integer() for _,_,w in floating) else ((1<<24)-1)/maximum
    edges=[(u,v,2*math.floor(w*scale+.5)) for u,v,w in floating]
    original={key(u,v):Fraction(w) for u,v,w in floating}
    integer={key(u,v):w for u,v,w in edges}
    alpha=min(original[e]/w for e,w in integer.items() if w>0)
    n=circuit.num_detectors
    models.append({"distance":d,"rounds":rounds,"nodes":n,"original_edges":floating,"integer_edges":edges})
    all_shots=circuit.compile_detector_sampler(seed=20261103+d*100+rounds).sample(a.shots)
    # Integers supplied to normal PyMatching are already exact and bypass re-rounding.
    quantized=pymatching.Matching()
    for u,v,w in edges:
        if v==-1:quantized.add_boundary_edge(u,weight=w/2)
        else:quantized.add_edge(u,v,weight=w/2)
    ladder_edges=[(u,v,float(w)) for u,v,w in edges]
    ladder=CertifyingDecoder(ladder_edges)
    local=[]
    with NativeDual(n,edges,a.library) as native:
        warm=circuit.compile_detector_sampler(seed=719+d+rounds).sample(32)
        for bits in warm:
            syn=np.flatnonzero(bits).astype(int).tolist()
            if not syn:continue
            p=native.decode(syn);r=check(n,edges,syn,p)
            assert r["accepted"]
            ladder.certify(syn,r["correction"])
            quantized.decode_to_matched_dets_array(bits)
            break
        for shot,bits in enumerate(all_shots):
            syn=np.flatnonzero(bits).astype(int).tolist()
            mode_times={False:[],True:[]}
            for repetition in range(6):
                for mode in ((False,True) if (shot+repetition)%2 else (True,False)):
                    start=time.perf_counter_ns();item=native.decode(syn,export=mode);elapsed=time.perf_counter_ns()-start
                    mode_times[mode].append(elapsed)
                    if mode:proof=item
                    else:base=item
            baseline=statistics.median(mode_times[False]);exported=statistics.median(mode_times[True])
            start=time.perf_counter_ns();checked=check(n,edges,syn,proof);checker=time.perf_counter_ns()-start
            assert checked["accepted"],(d,rounds,shot,checked)
            assert sorted(map(sorted,base["pairs"]))==sorted(map(sorted,proof["pairs"]))
            stock_edges=[tuple(map(int,e)) for e in quantized.decode_to_edges_array(bits)]
            stock_cost=sum(integer[key(u,v)] for u,v in stock_edges)
            assert stock_cost==checked["weight"],(stock_cost,checked)
            start=time.perf_counter_ns();quantized.decode_to_matched_dets_array(bits);stock_time=time.perf_counter_ns()-start
            correction=checked["correction"]
            actual_original=sum((original[key(u,v)] for u,v in correction),Fraction())
            native_bound_original=alpha*checked["dual_bound"]
            epsilon=actual_original-native_bound_original
            assert epsilon>=0
            start=time.perf_counter_ns();reconciled=reconcile(n,floating,syn,proof,checked);reconcile_time=time.perf_counter_ns()-start
            row={"distance":d,"rounds":rounds,"shot":shot,"syndrome":syn,"proof":proof,"checked":checked,"baseline_native_wrapper_ns":baseline,"export_native_wrapper_ns":exported,"independent_checker_ns":checker,"stock_pymatching_pairs_ns":stock_time,"original_exact_cost":[actual_original.numerator,actual_original.denominator],"original_epsilon_bound":[epsilon.numerator,epsilon.denominator],"original_metric_reconciliation":reconciled,"original_reconcile_ns":reconcile_time}
            if syn and sum("ladder_ns" in r for r in local)<a.ladder_shots:
                start=time.perf_counter_ns();res=ladder.certify(syn,correction);row["ladder_ns"]=time.perf_counter_ns()-start
                row["ladder_accepted"]=bool(res.accepted)
                row["ladder_rung"]=res.rung
            records.append(row);local.append(row)
    nontriv=[r for r in local if r["syndrome"]]
    measured=[r for r in local if "ladder_ns" in r]
    cells.append({"distance":d,"rounds":rounds,"nodes":n,"edges":len(edges),"shots":len(local),"nontrivial":len(nontriv),"native_verified":len(local),"blossom_certificates":sum(bool(r["checked"]["blossom_terms"]) for r in local),"native_solve_ns":stats([r["proof"]["native_solve_ns"] for r in nontriv]),"native_snapshot_ns":stats([r["proof"]["native_snapshot_ns"] for r in nontriv]),"native_export_wrapper_ns":stats([r["export_native_wrapper_ns"] for r in nontriv]),"independent_checker_ns":stats([r["independent_checker_ns"] for r in nontriv]),"same_shot_native_plus_check_ns":stats([r["export_native_wrapper_ns"]+r["independent_checker_ns"] for r in measured]),"same_shot_ladder_ns":stats([r["ladder_ns"] for r in measured]),"ladder_accepted":sum(r["ladder_accepted"] for r in measured),"max_original_epsilon_bound":max(float(Fraction(*r["original_epsilon_bound"])) for r in local),"mean_original_epsilon_bound":statistics.mean(float(Fraction(*r["original_epsilon_bound"])) for r in local),"scale_float":scale,"model_integer_sha256":hashlib.sha256(json.dumps(edges,separators=(",",":")).encode()).hexdigest()})
    cells[-1].update(stock_pymatching_pairs_ns=stats([r["stock_pymatching_pairs_ns"] for r in nontriv]),native_decode_noexport_wrapper_ns=stats([r["baseline_native_wrapper_ns"] for r in nontriv]),original_metric_epsilon_max=max(float(Fraction(*r["original_metric_reconciliation"]["epsilon"])) for r in local),original_metric_exact=sum(r["original_metric_reconciliation"]["exact_optimal"] for r in local),original_metric_within_1e_9=sum(r["original_metric_reconciliation"]["within_1e_9"] for r in local),original_metric_nontrivial_within_1e_9=sum(r["original_metric_reconciliation"]["within_1e_9"] for r in nontriv),original_metric_reconcile_ns=stats([r["original_reconcile_ns"] for r in nontriv]),warmup="one independent nontrivial warmup per graph; native mode comparison takes median of six counterbalanced warm same-syndrome repetitions")
    print(json.dumps(cells[-1]),flush=True)
out={"schema":"oneq.native_metric_dual_benchmark.v1","scope":"research prototype; quantized graph exactness; original IEEE graph epsilon bound only","numpy":np.__version__,"pymatching":pymatching.__version__,"stim":stim.__version__,"models":models,"cells":cells,"records":records}
a.output.write_text(json.dumps(out,indent=2)+"\n")
