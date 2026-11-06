#!/usr/bin/env python3
"""R158 independent, no-provider analysis of future 60-circuit IBM counts.

Primary test: sign-by-time-by-echo difference-in-differences of target Y.
Independent bounded-shot Hoeffding intervals (Bonferroni over 4 waits).
DO NOT infer a coherent J from this statistic alone. This consumes ONLY
files, cannot access IBM or spend quota, and labels the missing provider
attestation and cross-PUB temporal independence as explicit limitations.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import sys

WAITS=(0,512,2048,4096,8192)
AXES=("X","Y","Z")
SIGNS=(1,-1)
ECHOS=(False,True)
SHOTS=1024
ALPHA=0.01


class AnalysisRefused(ValueError):
    pass


def need(condition,reason):
    if not condition:
        raise AnalysisRefused(reason)


def name(t,z,e,axis):
    return f"r158_w{t:05d}_z{'P' if z==1 else 'M'}_e{int(e)}_{axis}"


def observed(hist):
    """Classical bit 0 (rightmost) is target; bit 1 is neighbor."""
    need(type(hist) is dict and bool(hist),"missing shot histogram")
    shots=0
    target0=0
    neighbour1=0
    for bits,count in hist.items():
        need(type(bits) is str and len(bits)==2 and set(bits)<={"0","1"}
             and type(count) is int and count>=0,
             "invalid 2-bit circuit count or nonbinary output")
        shots+=count
        if bits[1]=="0":
            target0+=count
        if bits[0]=="1":
            neighbour1+=count
    need(shots==SHOTS,"frozen circuit shot count mismatch")
    return {"shots":shots,
            "target_expectation":2*target0/shots-1,
            "neighbor_measured_one":neighbour1/shots}


def primary(record,counts):
    need(record.get("schema")=="oneq.r158.native-frozen-sign-probe/1"
         and record.get("physical_pair")==[148,149]
         and record.get("backend")=="ibm_kingston",
         "R158 must use its original registered physical design")
    need(record.get("provider_submit_performed") is False,
         "a frozen design cannot claim provider submission")
    keys={name(t,z,e,a) for t in WAITS for z in SIGNS
          for e in ECHOS for a in AXES}
    need(type(counts) is dict and set(counts)==keys
         and set(record.get("shots",{}))==keys,
         "missing or extra registered circuit/histogram")
    need(all(record["shots"][k]==SHOTS for k in keys),
         "shot allocation changed")
    stats={k:observed(counts[k]) for k in keys}
    def Y(t,z,e):
        return stats[name(t,z,e,"Y")]["target_expectation"]

    test=[]
    for t in WAITS[1:]:
        # Each row uses EXACTLY 8 distinct, preregistered Y PUBs.
        # The z-symmetric sign contrast cancels generic target readout
        # offsets; the t=0 and echo arms cancel additional offsets.
        terms=[
            (t,+1,False,+1), (t,-1,False,-1),
            (0,+1,False,-1), (0,-1,False,+1),
            (t,+1,True,-1), (t,-1,True,+1),
            (0,+1,True,+1), (0,-1,True,-1),
        ]
        value=math.fsum(coef*Y(w,z,e) for w,z,e,coef in terms)
        # Hoeffding: each of n Bernoulli ±1 shots enters as
        # ± 1/n in ONE of eight means, with per-shot range 2/n.
        # Bonferroni two-sided across 4 nonzero waits, delta=0.01.
        sum_range_sq=math.fsum(4/stats[name(w,z,e,"Y")]["shots"]
                              for w,z,e,_ in terms)
        width=math.sqrt(.5*sum_range_sq*math.log(2*(len(WAITS)-1)/ALPHA))
        test.append({
            "wait_dt":t,
            "sign_echo_interaction_estimate":value,
            "hoeffding_99pct_familywise_halfwidth_IID":width,
            "familywise_99pct_interval_IID":[value-width,value+width],
            "zero_outside_conditional_99pct_interval":abs(value)>width,
            "ideal_static_ZZ_prediction":"2 sin(J t); no unique J inference",
            "null_pair_pauli_prediction":0.,
            "shot_model_assumption":"independent same-distribution shots, no time-dependent drift",
        })
    diag={"condition_count":len(keys),"total_registered_shots":len(keys)*SHOTS}
    by={
        name(t,z,e,a):{
            "target_mean":stats[name(t,z,e,a)]["target_expectation"],
            "neighbor_wrong_rate":
                stats[name(t,z,e,a)]["neighbor_measured_one"] if z==1
                else 1-stats[name(t,z,e,a)]["neighbor_measured_one"],
        }
        for t in WAITS for z in SIGNS for e in ECHOS for a in AXES
    }
    diag["by_condition"]=by
    return {
        "schema":"oneq.r158.independent-posthoc-conditional-analysis/1",
        "registered_qpy_sha256":record["qpy_sha256"],
        "primary":test,
        "diagnostics":diag,
        "evidence_status":"OFFLINE_HISTOGRAM_ARITHMETIC_NOT_IBM_PROVIDER_AUTHENTICATED",
        "limitations":[
            "Bonferroni-Hoeffding only controls independent shots; time blocks unmodeled",
            "This is a proposed mechanism discriminator; nonzero Y cannot uniquely prove ZZ",
            "Neighbor preparation may induce coherent microwave/spectator cross-talk",
            "Readout drift and QPY/native pulse timing remain separate physical gates",
            "No actual R158 device result has been analyzed or authenticated",
        ],
    }


def main(args):
    if len(args)!=2:
        raise SystemExit("usage: python tools/analyze_r158_independent.py FROZEN.json COUNTS.json")
    fr,co=[pathlib.Path(x).read_bytes() for x in args]
    result=primary(json.loads(fr),json.loads(co))
    result["input_sha256"]={"frozen":hashlib.sha256(fr).hexdigest(),
                            "counts":hashlib.sha256(co).hexdigest()}
    print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False))


if __name__=="__main__":
    try:
        main(sys.argv[1:])
    except AnalysisRefused as e:
        print("REFUSED:",e,file=sys.stderr)
        sys.exit(2)
