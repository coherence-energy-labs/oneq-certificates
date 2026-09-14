r"""Fetch a working set of Google's surface-code corpus by HTTP range.

Zenodo record 10.5281/zenodo.13273331, archive
`google_105Q_surface_code_d3_d5_d7.zip` (5.72 GB, 9959 members) -- the data
behind "Quantum error correction below the surface code threshold". We need
detector syndromes and the matching graph the experimenters themselves used,
which is a few dozen MB, so the archive is read in place rather than
downloaded.

WHY REAL SYNDROMES MATTER HERE. The million-shot run establishes that the
checker's soundness does not degrade with volume, but it does so on simulated
circuit noise, which is i.i.d. in a way hardware is not. Leakage, crosstalk and
drift produce detector patterns that a depolarizing model does not generate,
and the certification RATE is exactly the quantity that could fall apart on
them. Soundness cannot: a forged certificate is refused whatever produced the
syndrome. So this run measures the rate honestly and re-runs the forgeries to
confirm the floor holds.

WHAT IS FETCHED per configuration:
  detection_events.b8  -- the device's detector outcomes, bit-packed
  error_model.dem      -- the decoder's own graph, so the weights are the
                          experimenters' and not ours to choose
  metadata.json        -- shot counts and provenance

The DEM taken is `correlated_matching_decoder_with_si1000_prior`, the graphlike
model a matching decoder consumes. Harmony and Libra outputs exist in the same
tree and are NOT used: they are not matching decoders, so certifying their
answers against a matching dual would be comparing two different problems.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from zenodo_range import RemoteZip  # noqa: E402

RECORD = "13273331"
ARCHIVE = "google_105Q_surface_code_d3_d5_d7.zip"
URL = f"https://zenodo.org/api/records/{RECORD}/files/{ARCHIVE}/content"
PRIOR = "correlated_matching_decoder_with_si1000_prior"
ROOT = pathlib.Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "google_corpus"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, nargs="+", default=[13, 30])
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--bases", type=str, nargs="+", default=["X", "Z"])
    ap.add_argument("--max-patches", type=int, default=1,
                    help="patches per distance (the corpus has several "
                         "physical placements of the same code distance)")
    a = ap.parse_args()

    print(f"opening {ARCHIVE} by range ...")
    z = RemoteZip(URL)
    print(f"  {len(z.entries)} members, {z.size/1e9:.2f} GB, not downloaded")

    pat = re.compile(
        r"google_105Q_surface_code_d3_d5_d7/(d(\d+)_at_(\S+?))/([XZ])/r(\d+)/"
        r"detection_events\.b8$")
    want: dict[tuple, str] = {}
    patches: dict[int, list[str]] = {}
    for k in z.entries:
        m = pat.match(k)
        if not m:
            continue
        dist, patch, basis, rnd = int(m.group(2)), m.group(1), m.group(4), int(m.group(5))
        if dist not in a.distances or rnd not in a.rounds or basis not in a.bases:
            continue
        patches.setdefault(dist, [])
        if patch not in patches[dist]:
            patches[dist].append(patch)
        if patches[dist].index(patch) >= a.max_patches:
            continue
        want[(dist, patch, basis, rnd)] = k

    DEST.mkdir(parents=True, exist_ok=True)
    manifest = []
    total = 0
    for (dist, patch, basis, rnd), det_key in sorted(want.items()):
        base = det_key.rsplit("/", 1)[0]
        out = DEST / f"{patch}_{basis}_r{rnd}"
        out.mkdir(parents=True, exist_ok=True)
        rec = {"distance": dist, "patch": patch, "basis": basis, "rounds": rnd,
               "prior": PRIOR, "files": {}}
        for label, key in (("detection_events.b8", det_key),
                           ("metadata.json", f"{base}/metadata.json"),
                           ("error_model.dem",
                            f"{base}/decoding_results/{PRIOR}/error_model.dem"),
                           ("obs_flips_actual.b8", f"{base}/obs_flips_actual.b8")):
            if key not in z.entries:
                print(f"  MISSING {key}")
                continue
            dst = out / label
            if dst.exists() and dst.stat().st_size == z.entries[key]["usize"]:
                data = dst.read_bytes()
            else:
                data = z.read(key)          # CRC-checked inside RemoteZip
                dst.write_bytes(data)
            total += len(data)
            rec["files"][label] = {
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "zip_crc32": z.entries[key]["crc"],
                "member": key,
            }
        manifest.append(rec)
        print(f"  {patch} {basis} r={rnd:<4d} {sum(f['bytes'] for f in rec['files'].values())/1e6:7.2f} MB")

    (DEST / "MANIFEST.json").write_text(json.dumps({
        "schema": "oneq-google-corpus-manifest/1",
        "doi": "10.5281/zenodo.13273331",
        "archive": ARCHIVE,
        "archive_bytes": z.size,
        "fetched_by": "HTTP range requests; the archive was never downloaded whole",
        "integrity": ("each member's CRC-32 was checked against the ZIP central "
                      "directory on fetch; sha256 recorded here so a reader "
                      "holding the full archive can confirm these are the same "
                      "bytes"),
        "configurations": manifest,
    }, indent=2), encoding="utf-8")
    print(f"\n{len(manifest)} configurations, {total/1e6:.1f} MB -> {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
