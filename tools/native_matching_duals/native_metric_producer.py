"""Produce a portable bound receipt from a native terminal metric dual."""
import math
if __package__:
    from .native_wrapper import NativeDual
    from .native_metric_verify import model_identity,SCHEMA
else:
    from native_wrapper import NativeDual
    from native_metric_verify import model_identity,SCHEMA

class NativeMetricProducer:
    def __init__(self,edges,library_path=None):
        self.edges=[(v,u,float(w)) if u==-1 else (u,v,float(w)) for u,v,w in edges]
        self.model_sha256=model_identity(self.edges)
        self.nodes=max(max(u,v) for u,v,_ in self.edges)+1
        maxweight=max(w for _,_,w in self.edges)
        if maxweight>(1<<24)-1:raise ValueError("PyMatching integer dynamic range exceeded")
        self.scale=1. if all(w.is_integer() for _,_,w in self.edges) else ((1<<24)-1)/maxweight
        self.integer_edges=[(u,v,2*math.floor(w*self.scale+.5)) for u,v,w in self.edges]
        self.native=NativeDual(self.nodes,self.integer_edges,library_path)

    def produce(self,syndrome,correction=None):
        """correction may be a caller's existing decoder edge set.

        If omitted, uses native Sparse Blossom's original SearchFlooder path
        expansion from the SAME solve. No duplicate decoding or Python shortest
        path reconstruction is needed in the producer.
        """
        syndrome=list(syndrome)
        proof=self.native.decode(syndrome)
        if correction is None:
            correction=proof["correction"]
        return {"schema":SCHEMA,"model_sha256":self.model_sha256,"syndrome":list(syndrome),"integer_edges":[list(e) for e in self.integer_edges],"native_proof":proof,"correction":[list(e) for e in correction],"scope":"nonnegative simple matching graph; original IEEE weights reconciled by independent exact rational metric constraints","negative_weights_supported":False,"correlated_reweighting_supported":False}

    def close(self):self.native.close()
