from types import SimpleNamespace

import numpy as np
import pytest

from oneq.matching_cert_epsilon import check_epsilon


def check(endpoints):
    cert=SimpleNamespace(matched_edges=[endpoints], node_potentials={},
                         blossom_duals={frozenset({0}):1.0})
    return check_epsilon(edges=[(0,-1,1.0)],syndrome=[0],cert=cert,eps_max=0)


@pytest.mark.parametrize("value",[0.9,0.0,"0",False,np.bool_(False),np.float64(0),float('nan'),float('inf')])
def test_malformed_endpoint_cannot_be_truncated_to_valid_edge(value):
    result=check([value,-1])
    assert result.status=="INVALID_INPUT" and not result.accepted
    assert result.primal is None and result.lower_bound is None


@pytest.mark.parametrize("edge",[[],[0],[0,-1,0],0,None])
def test_edge_arity_and_shape_refused(edge):
    assert check(edge).status=="INVALID_INPUT"


def test_python_and_numpy_integer_edges_keep_exact_semantics():
    for edge in ([0,-1],np.array([0,-1],dtype=np.int64)):
        result=check(edge)
        assert result.accepted and result.status=="EXACT_OPTIMAL"
        assert result.primal==result.lower_bound==1
