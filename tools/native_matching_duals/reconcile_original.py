"""Exact original-IEEE graph lower bound from a verified native metric dual.

Chooses the largest uniform nonnegative scaling obeying every original metric
pair/boundary constraint, then checks them all. The resulting gap is an exact
epsilon bound, never exact optimality unless the gap is exactly zero.
"""
from fractions import Fraction
import heapq
from math import lcm

def reconcile(nodes,original_edges,syndrome,proof,integer_check,correction=None):
    if not integer_check.get("accepted"):raise ValueError("integer proof must verify first")
    adj={i:[] for i in range(-1,nodes)};weights={};denominator=1
    for u,v,w in original_edges:
        q=Fraction(w)
        if q<0:raise ValueError("negative original weights unsupported")
        key=(min(u,v),max(u,v))
        if key in weights:raise ValueError("duplicate original edge")
        weights[key]=q;denominator=lcm(denominator,q.denominator)
    # Binary64 weights share a power-of-two lattice. Integer Dijkstra computes
    # exactly the same rational distances without Fraction arithmetic per edge.
    for (u,v),q in weights.items():
        iw=q.numerator*(denominator//q.denominator)
        adj[u].append((v,iw));adj[v].append((u,iw))
    terms=[(set(row["support"]),row["value"]) for row in proof["dual"]]
    constraints=[]
    for i,u in enumerate(syndrome):
        dist={u:0};queue=[(0,u)]
        while queue:
            d,x=heapq.heappop(queue)
            if d!=dist[x] or x==-1:continue
            for v,w in adj[x]:
                nd=d+w
                if v not in dist or nd<dist[v]:dist[v]=nd;heapq.heappush(queue,(nd,v))
        if -1 in dist:constraints.append((sum(z for support,z in terms if u in support),dist[-1]))
        for v in syndrome[i+1:]:
            if v in dist:constraints.append((sum(z for support,z in terms if (u in support)!=(v in support)),dist[v]))
    scales=[Fraction(distance,load*denominator) for load,distance in constraints if load>0]
    theta=min(scales) if scales else Fraction()
    assert all(theta*load*denominator<=distance for load,distance in constraints)
    bound=theta*sum(z for _,z in terms)
    correction=integer_check["correction"] if correction is None else correction
    parity=set();seen=set()
    for row in correction:
        if len(row)!=2 or any(type(x) is not int for x in row):raise ValueError("correction domain")
        u,v=row;key=(min(u,v),max(u,v))
        if key not in weights or key in seen:raise ValueError("unknown or duplicate correction edge")
        seen.add(key)
        for t in (u,v):
            if t==-1:continue
            if t in parity:parity.remove(t)
            else:parity.add(t)
    if parity!=set(syndrome):raise ValueError("correction syndrome mismatch")
    actual=sum((weights[e] for e in seen),Fraction())
    epsilon=actual-bound
    if epsilon<0:raise ValueError("invalid rational bound")
    pair=lambda q:[q.numerator,q.denominator]
    return {"scale":pair(theta),"lower_bound":pair(bound),"primal":pair(actual),"epsilon":pair(epsilon),"exact_optimal":epsilon==0,"within_1e_9":epsilon<=Fraction(1,10**9),"constraints_checked":len(constraints)}
