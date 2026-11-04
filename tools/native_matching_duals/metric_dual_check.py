"""Independent integer-only checker for terminal metric-matching duals.

No PyMatching, ONE_Q, optimizer, native code, or floating point is imported.
The graph's weights and all dual variables are exact integers in the SAME units.
Each terminal is matched once, to another terminal or to the boundary. Singleton
dual variables are free; every other odd-set cut multiplier is nonnegative.
All metric pair/boundary inequalities are verified using fresh integer Dijkstra.
Attainment proves the minimum T-join weight on this nonnegative declared graph.
It does not certify original pre-quantization floats or a logical coset posterior.
"""
from __future__ import annotations
import heapq

class Refused(ValueError):pass

def integer(v):
    if type(v) is not int:raise Refused("lossless integer required")
    return v

def key(u,v):return (min(u,v),max(u,v))

def check(nodes,edges,syndrome,proof):
    try:return _check(nodes,edges,syndrome,proof)
    except (ValueError,TypeError,KeyError,IndexError,OverflowError) as exc:
        return {"accepted":False,"reason":str(exc)}

def _check(nodes,edges,syndrome,proof):
    if not isinstance(proof,dict):raise Refused("proof must be an object")
    if integer(nodes)<=0:raise Refused("empty graph")
    adj={i:[] for i in range(-1,nodes)}
    weights={}
    for row in edges:
        if len(row)!=3:raise Refused("edge width")
        u,v,w=map(integer,row)
        if u<0 or u>=nodes or v< -1 or v>=nodes or u==v or w<0:raise Refused("graph domain")
        k=key(u,v)
        if k in weights:raise Refused("parallel edges must be canonicalized")
        weights[k]=w
        adj[u].append((v,w));adj[v].append((u,w))
    terminals=[integer(s) for s in syndrome]
    if len(set(terminals))!=len(terminals) or any(s<0 or s>=nodes for s in terminals):raise Refused("syndrome domain")
    terminal_set=set(terminals)
    if proof.get("schema")!="oneq.native_metric_dual.v1":raise Refused("unknown proof schema")
    dual=[]
    seen=set()
    for row in proof["dual"]:
        support=tuple(integer(x) for x in row["support"])
        value=integer(row["value"])
        if not support or len(support)%2!=1 or len(set(support))!=len(support):raise Refused("dual support must be odd")
        if not set(support)<=terminal_set:raise Refused("dual support contains nonterminal")
        if len(support)>1 and value<0:raise Refused("negative nonsingleton dual")
        ss=frozenset(support)
        if ss in seen:raise Refused("duplicate dual support")
        seen.add(ss);dual.append((ss,value))

    def distances(source):
        dist={source:0};prev={source:None};todo=[(0,source)]
        while todo:
            d,u=heapq.heappop(todo)
            if d!=dist[u] or u==-1:continue
            for v,w in adj[u]:
                nd=d+w
                if v not in dist or nd<dist[v]:
                    dist[v]=nd;prev[v]=u;heapq.heappush(todo,(nd,v))
        return dist,prev

    all_dist={t:distances(t) for t in terminals}
    constraints=0
    for i,u in enumerate(terminals):
        dist,_=all_dist[u]
        load=sum(z for support,z in dual if u in support)
        if -1 in dist and load>dist[-1]:raise Refused("boundary dual infeasible")
        constraints+=(-1 in dist)
        for v in terminals[i+1:]:
            if v not in dist:continue
            load=sum(z for support,z in dual if (u in support)!=(v in support))
            if load>dist[v]:raise Refused("pair dual infeasible")
            constraints+=1
    covered=[];primal=0;correction=set()
    for row in proof["pairs"]:
        if len(row)!=2:raise Refused("pair width")
        u,v=map(integer,row)
        if u not in terminal_set or v==u or (v!=-1 and v not in terminal_set):raise Refused("pair endpoint")
        covered.append(u)
        if v!=-1:covered.append(v)
        dist,prev=all_dist[u]
        if v not in dist:raise Refused("pair has no path")
        primal+=dist[v]
        at=v
        while at!=u:
            before=prev[at];edge=key(at,before)
            if edge in correction:correction.remove(edge)
            else:correction.add(edge)
            at=before
    if sorted(covered)!=sorted(terminals):raise Refused("matching fails exact terminal coverage")
    bound=sum(z for _,z in dual)
    if primal!=bound:raise Refused("dual does not attain primal")
    if "correction" in proof:
        correction=set()
        for row in proof["correction"]:
            if len(row)!=2:raise Refused("correction width")
            u,v=map(integer,row);edge=key(u,v)
            if edge not in weights or edge in correction:raise Refused("unknown or duplicate correction edge")
            correction.add(edge)
    actual=sum(weights[e] for e in correction)
    if actual!=primal:raise Refused("expanded XOR correction cost differs from matched paths")
    parity=set()
    for u,v in correction:
        for t in (u,v):
            if t==-1:continue
            if t in parity:parity.remove(t)
            else:parity.add(t)
    if parity!=terminal_set:raise Refused("expanded correction syndrome mismatch")
    return {"accepted":True,"status":"EXACT_OPTIMAL_FOR_DECLARED_INTEGER_GRAPH","weight":actual,"dual_bound":bound,"constraints_checked":constraints,"correction":[list(e) for e in sorted(correction)],"blossom_terms":sum(len(s)>1 for s,_ in dual)}
