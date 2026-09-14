r"""Lane 3 gate: the streaming path on real Willow shots, with a clock on it.

Three numbers this artifact exists to pin, per the design doc
(docs/REALTIME_EMISSION_DESIGN.md):

  EQUIVALENCE  on a sampled prefix, every streaming receipt's certificate is
               BYTE-identical (canonical JSON document) to the batch path's --
               a streaming layer that changes verdicts is a second decoder
               wearing the checker's coat;
  THROUGHPUT   sustained shots/s through the ordered stream on N workers, on
               real hardware syndromes, per config -- the number that decides
               whether the audit stream keeps pace with a device;
  LATENCY      per-shot pull-to-receipt time, p50/p99 -- the number the
               post-selection row of the latency map needs.

No fabricated target is asserted. The design doc's original "10x acquisition
rate" line was unmeetable arithmetic and is corrected to: report the measured
rate beside the corpus's own per-config shot count so a reader computes the
ratio for the device they care about.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from oneq.cert_format import to_document  # noqa: E402
from oneq.certifying_decoder import BOUNDARY, CertifyingDecoder  # noqa: E402
from oneq.streaming import StreamingCertifier  # noqa: E402


def load_config(name: str, limit: int):
    import pymatching
    import stim
    d = ROOT / "data" / "google_corpus" / name
    dem = stim.DetectorErrorModel.from_file(str(d / "error_model.dem"))
    m = pymatching.Matching.from_detector_error_model(dem)
    nb = (dem.num_detectors + 7) // 8
    raw = np.frombuffer((d / "detection_events.b8").read_bytes(),
                        dtype=np.uint8).reshape(-1, nb)
    det = np.unpackbits(raw[:limit], axis=1,
                        bitorder="little")[:, :dem.num_detectors].astype(bool)
    edges = [(int(e[0]), BOUNDARY if e[1] is None else int(e[1]),
              float(e[2].get("weight", 1.0))) for e in m.edges()]
    return m, det, edges


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="d5_at_q8_7_X_r30")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--equiv-sample", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    m, det, edges = load_config(a.config, a.shots)
    cd = CertifyingDecoder(edges)

    shots = []
    for s in range(det.shape[0]):
        f = [int(i) for i in np.flatnonzero(det[s])]
        if not f:
            continue
        corr = tuple(cd.key((u, v if v >= 0 else BOUNDARY))
                     for u, v in m.decode_to_edges_array(det[s]))
        shots.append((f, corr))

    # EQUIVALENCE on the sampled prefix: canonical documents, byte-compared.
    n_eq = min(a.equiv_sample, len(shots))
    batch_docs = []
    for (syn, corr) in shots[:n_eq]:
        r = cd.certify(syn, corr)
        batch_docs.append(json.dumps(
            to_document(edges=edges, syndrome=syn, cert=r.cert,
                        proven_optimal=r.accepted, primal=r.primal,
                        bound=r.bound, reason=r.reason), sort_keys=True))

    pull_t: dict[int, float] = {}

    def timed(seq):
        for i, item in enumerate(seq):
            pull_t[i] = time.perf_counter()
            yield item

    sc = StreamingCertifier(edges, workers=a.workers,
                            max_pending=4 * a.workers)
    lat = []
    tiers = {"certified": 0, "degraded": 0, "deferred": 0}
    mismatch = 0
    t0 = time.perf_counter()
    for r in sc.run(timed(shots)):
        lat.append(time.perf_counter() - pull_t[r.shot_id])
        tiers[r.tier] += 1
        if r.shot_id < n_eq:
            sd = json.dumps(
                to_document(edges=edges, syndrome=shots[r.shot_id][0],
                            cert=r.cert,
                            proven_optimal=(r.tier == "certified"),
                            primal=r.primal, bound=r.bound,
                            reason=r.reason), sort_keys=True)
            if sd != batch_docs[r.shot_id]:
                mismatch += 1
    wall = time.perf_counter() - t0
    assert sc.stats.certified_through == len(shots) - 1

    la = np.array(lat)
    out = {
        "schema": "oneq-streaming-gate/1",
        "config": a.config,
        "nontrivial_shots": len(shots),
        "workers": a.workers,
        "equivalence": {"sampled": n_eq, "mismatches": mismatch,
                        "claim": "streaming certificates are byte-identical "
                                 "(canonical JSON) to the batch path on the "
                                 "sampled prefix"},
        "tiers": tiers,
        "certified_rate": tiers["certified"] / max(1, len(shots)),
        "throughput_shots_per_s": len(shots) / wall,
        "wall_s": wall,
        "latency_s": {"p50": float(np.percentile(la, 50)),
                      "p90": float(np.percentile(la, 90)),
                      "p99": float(np.percentile(la, 99)),
                      "max": float(la.max())},
        "watermark_final": sc.stats.certified_through,
        "reading": ("no fabricated target: compare throughput to the device "
                    "rate YOU care about; latency p99 answers the "
                    "post-selection row of the latency map"),
    }
    if mismatch:
        print(f"EQUIVALENCE BROKEN: {mismatch}/{n_eq}", file=sys.stderr)
        return 2
    dest = pathlib.Path(a.out) if a.out else (
        ROOT / "evidence" / f"streaming_gate_{a.config}.json")
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"{a.config}: {len(shots)} shots, {tiers['certified']} certified, "
          f"{out['throughput_shots_per_s']:.1f} shots/s on {a.workers} "
          f"workers, latency p50 {1e3*out['latency_s']['p50']:.0f} ms "
          f"p99 {1e3*out['latency_s']['p99']:.0f} ms, "
          f"equivalence {n_eq - mismatch}/{n_eq}")
    print(f"evidence -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
