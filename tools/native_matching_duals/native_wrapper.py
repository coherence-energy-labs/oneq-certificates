"""ctypes binding to the research native terminal-dual snapshot bridge."""
from __future__ import annotations
import ctypes
import json
from pathlib import Path

I=ctypes.c_int64
P=ctypes.POINTER(I)

class NativeDual:
    def __init__(self,nodes,edges,library=None):
        self.lib=ctypes.CDLL(str(library or Path(__file__).with_name("native_dual.so")))
        self.lib.nd_create.argtypes=[I,I,P]
        self.lib.nd_create.restype=ctypes.c_void_p
        self.lib.nd_decode.argtypes=[ctypes.c_void_p,I,P,ctypes.c_int]
        self.lib.nd_decode.restype=ctypes.c_char_p
        self.lib.nd_free.argtypes=[ctypes.c_void_p]
        self.lib.nd_error.restype=ctypes.c_char_p
        flat=[x for e in edges for x in e]
        if any(type(x) is not int for x in [nodes,*flat]): raise ValueError("integer input required")
        self.handle=self.lib.nd_create(nodes,len(edges),(I*len(flat))(*flat))
        if not self.handle: raise ValueError(self.lib.nd_error().decode())

    def decode(self,syndrome,export=True):
        if not self.handle:raise ValueError("native producer is closed")
        if any(type(x) is not int for x in syndrome):raise ValueError("integer syndrome required")
        raw=self.lib.nd_decode(self.handle,len(syndrome),(I*len(syndrome))(*syndrome),int(export))
        if raw is None:raise ValueError(self.lib.nd_error().decode())
        return json.loads(raw)

    def close(self):
        if self.handle:self.lib.nd_free(self.handle);self.handle=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
