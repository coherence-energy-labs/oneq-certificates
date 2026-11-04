"""Certificate IDs are Python ints; producer-owned arrays are normalized first."""
from types import SimpleNamespace

import numpy as np
import pytest

from oneq.matching_cert_epsilon import check_epsilon


def check(endpoints):
    cert = SimpleNamespace(matched_edges=[endpoints], node_potentials={},
                           blossom_duals={frozenset({0}): 1.0})
    return check_epsilon(edges=[(0, -1, 1.0)], syndrome=[0], cert=cert, eps_max=0)


@pytest.mark.parametrize("value", [
    0.9, 0.0, "0", False, np.bool_(False), np.float64(0),
    np.int64(0), np.uint64(0), float("nan"), float("inf"),
])
def test_certificate_endpoint_domain_is_not_silently_coerced(value):
    result = check([value, -1])
    assert result.status == "INVALID_INPUT" and not result.accepted
    assert result.primal is None and result.lower_bound is None


@pytest.mark.parametrize("edge", [[], [0], [0, -1, 0], 0, None])
def test_edge_arity_and_shape_refused(edge):
    assert check(edge).status == "INVALID_INPUT"


def test_python_integer_edges_keep_exact_semantics():
    result = check([0, -1])
    assert result.accepted and result.status == "EXACT_OPTIMAL"
    assert result.primal == result.lower_bound == 1


def test_producer_normalizes_model_owned_numpy_corrections_before_checking():
    from oneq.certifying_decoder import CertifyingDecoder

    edges = [(0, -1, 1.0)]
    decoder = CertifyingDecoder(edges, ladder=((4, False),), epsilon_max=0)
    result = decoder.certify([0], np.array([[0, -1]], dtype=np.int64))
    assert result.accepted and result.cert is not None
    assert all(type(vertex) is int for edge in result.cert.matched_edges for vertex in edge)
    assert all(type(vertex) is int for support in result.cert.blossom_duals for vertex in support)
    verdict = check_epsilon(edges=edges, syndrome=[0], cert=result.cert, eps_max=0)
    assert verdict.accepted and verdict.status == "EXACT_OPTIMAL"
    assert verdict.primal == verdict.lower_bound == 1
