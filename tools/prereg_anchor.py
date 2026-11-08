r"""Anchor a preregistration outside the project before its run, and draw its seeds from a public beacon.

The protocol and what it proves are in docs/PREREG_ANCHORING.md. In short: a commit's date is set by whoever makes
it, so a preregistration in a private repository proves nothing about WHEN it was written (external audit 4, X3).
This tool replaces that with two clocks nobody here controls:

  1. DECLARE.  The preregistration is committed; this writes an ANCHOR naming its sha256 and a drand round far in
     the future, and stamps the anchor with OpenTimestamps.
  2. The stamp is attested in a Bitcoin block (hours). The round is published later (the declared lead, 48 h by
     default). The anchor is valid only if that block's timestamp is at least `margin_hours` before the round's
     publication time.
  3. REDEEM.  Once both exist, the round's randomness is fetched and the run's seeds are derived from it and from
     the preregistration's bytes (`oneq.prereg_seeds.derive_prereg_seeds`). Data that depends on those seeds cannot have
     existed before the round, and the round came after a block that proves the preregistration existed.
  4. VERIFY.  Anyone re-checks every link from public sources: the preregistration's bytes, the proof executed to
     each attested block's merkle root (two explorers), the block's time against the round's, the round's
     randomness (two drand relays), and the seeds.

    python tools/prereg_anchor.py declare docs/PREREG_x.md [--lead-hours 48] [--margin-hours 6]
    python tools/prereg_anchor.py status  docs/prereg_anchors/PREREG_x.anchor.json
    python tools/prereg_anchor.py redeem  docs/prereg_anchors/PREREG_x.anchor.json
    python tools/prereg_anchor.py verify  docs/prereg_anchors/PREREG_x.anchor.json
    python tools/prereg_anchor.py seeds   docs/prereg_anchors/PREREG_x.anchor.json --n 8

Network lives only in the two HTTP sources below; `--offline FIXTURE.json` replaces them with recorded data (the
tests use it). The derivation (`oneq.prereg_seeds`) and the proof execution (`oneq.ots_proof`) are pure.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oneq.beacon import declare_future_round  # noqa: E402
from oneq.prereg_seeds import derive_prereg_seeds, round_time  # noqa: E402
from oneq.ots_proof import ProofError, parse_detached  # noqa: E402

SCHEMA = "oneq-prereg-anchor/1"
REDEEMED_SCHEMA = "oneq-prereg-redemption/1"
ANCHOR_DIR = ROOT / "docs" / "prereg_anchors"
#: drand mainnet "quicknet": 3 s rounds, unchained, so a round's value cannot be computed from earlier ones.
DRAND_CHAIN = "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971"
DRAND_RELAYS = ("https://api.drand.sh", "https://drand.cloudflare.com")
EXPLORERS = ("https://blockstream.info/api", "https://mempool.space/api")
SEED_DOMAIN = "oneq-prereg-seed/1"


class AnchorError(Exception):
    pass


def _lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc(t: int) -> str:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------------------------------- public sources
class HttpSources:
    """Two drand relays and two block explorers. Every answer must come back identical from both, or nothing is
    believed: one source can be wrong or lying, and the protocol's whole point is not to trust ourselves either."""

    def _get(self, url: str) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": "oneq-prereg-anchor/1"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read()

    def _agree(self, what: str, answers: list):
        if any(a != answers[0] for a in answers[1:]):
            raise AnchorError(f"{what}: the two sources disagree: {answers}")
        return answers[0]

    def drand_info(self) -> dict:
        got = [json.loads(self._get(f"{u}/{DRAND_CHAIN}/info")) for u in DRAND_RELAYS]
        keep = [{k: g[k] for k in ("hash", "genesis_time", "period", "public_key")} for g in got]
        return self._agree("drand chain info", keep)

    def drand_latest(self) -> int:
        return min(int(json.loads(self._get(f"{u}/{DRAND_CHAIN}/public/latest"))["round"]) for u in DRAND_RELAYS)

    def drand_round(self, n: int) -> dict | None:
        out = []
        for u in DRAND_RELAYS:
            try:
                g = json.loads(self._get(f"{u}/{DRAND_CHAIN}/public/{n}"))
            except Exception:                            # not published yet at this relay
                return None
            out.append({"round": int(g["round"]), "randomness": g["randomness"], "signature": g["signature"]})
        return self._agree(f"drand round {n}", out)

    def block(self, height: int) -> dict:
        out = []
        for api in EXPLORERS:
            h = self._get(f"{api}/block-height/{height}").decode().strip()
            b = json.loads(self._get(f"{api}/block/{h}"))
            out.append({"hash": b["id"], "merkle_root": b["merkle_root"], "timestamp": int(b["timestamp"])})
        return self._agree(f"Bitcoin block {height}", out)


class FixtureSources(HttpSources):
    """Recorded answers, one dict per source, so the same agreement rule runs offline."""

    def __init__(self, path: pathlib.Path):
        self.doc = json.loads(path.read_text(encoding="utf-8"))

    def drand_info(self) -> dict:
        return self._agree("drand chain info", [s["drand_info"] for s in self.doc["sources"]])

    def drand_latest(self) -> int:
        return min(int(s["drand_latest"]) for s in self.doc["sources"])

    def drand_round(self, n: int) -> dict | None:
        got = [s["drand_rounds"].get(str(n)) for s in self.doc["sources"]]
        if any(g is None for g in got):
            return None
        return self._agree(f"drand round {n}", [{"round": n, **g} for g in got])

    def block(self, height: int) -> dict:
        got = [s["blocks"].get(str(height)) for s in self.doc["sources"]]
        if any(g is None for g in got):
            raise AnchorError(f"Bitcoin block {height}: not in the fixture")
        return self._agree(f"Bitcoin block {height}", got)


# ----------------------------------------------------------------------------------------------- declare
def _git(*args: str) -> str:
    r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        raise AnchorError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


def committed_prereg(rel: str) -> tuple[bytes, str, str]:
    """(bytes as committed, blob id, HEAD). Refuses a preregistration with uncommitted changes: the anchor must
    name the bytes the repository holds, not a working copy that may never be committed (law: a receipt that hashes
    a working copy verifies only there)."""
    if subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", rel],
                      stdin=subprocess.DEVNULL).returncode != 0:
        raise AnchorError(f"{rel} has uncommitted changes; commit it first")
    blob = _git("rev-parse", f"HEAD:{rel}")
    data = subprocess.run(["git", "-C", str(ROOT), "cat-file", "blob", blob], capture_output=True,
                          stdin=subprocess.DEVNULL, check=True).stdout
    return data, blob, _git("rev-parse", "HEAD")


def anchor_path(rel: str) -> pathlib.Path:
    return ANCHOR_DIR / (pathlib.PurePosixPath(rel).stem + ".anchor.json")


def declare(rel: str, *, sources, lead_hours: float, margin_hours: float, stamp: bool = True,
            prereg: tuple[bytes, str, str] | None = None, out: pathlib.Path | None = None) -> pathlib.Path:
    if lead_hours < 2 * margin_hours or margin_hours < 3:
        raise AnchorError("the lead must be at least twice the margin, and the margin at least 3 hours: Bitcoin "
                          "block times may sit up to two hours from real time, and an attestation takes hours")
    data, blob, head = prereg or committed_prereg(rel)
    info = sources.drand_info()
    if info["hash"] != DRAND_CHAIN:
        raise AnchorError(f"drand answered for chain {info['hash']}, not quicknet")
    latest = sources.drand_latest()
    target = declare_future_round(latest, lead_seconds=int(lead_hours * 3600), period_s=int(info["period"]))
    when = round_time(target, genesis_time=int(info["genesis_time"]), period_s=int(info["period"]))
    doc = {
        "schema": SCHEMA,
        "prereg": rel, "prereg_sha256": _sha(_lf(data)), "prereg_blob": blob, "prereg_commit": head,
        "beacon": {"name": "drand mainnet quicknet", "chain_hash": DRAND_CHAIN,
                   "genesis_time": int(info["genesis_time"]), "period_seconds": int(info["period"]),
                   "public_key": info["public_key"], "round": target, "round_time_utc": _utc(when)},
        "declared_at_round": latest, "lead_hours": lead_hours, "margin_hours": margin_hours,
        "seed_domain": SEED_DOMAIN,
        "rule": ("Valid only if this file's OpenTimestamps proof is attested in a Bitcoin block whose timestamp is "
                 "at least margin_hours before round_time_utc. The run's seeds are then "
                 "oneq.prereg_seeds.derive_prereg_seeds(randomness of that round, prereg_sha256=..., n=...), and nothing "
                 "else: data drawn with other seeds is not this preregistration's run."),
    }
    dest = out or anchor_path(rel)
    if dest.exists():
        raise AnchorError(f"{dest} exists: an anchor is never rewritten (declare a new preregistration instead)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    if stamp:
        r = subprocess.run(["ots", "stamp", str(dest)], capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if r.returncode != 0 or not dest.with_name(dest.name + ".ots").is_file():
            raise AnchorError(f"ots stamp failed: {r.stderr.strip()[-400:]}")
    return dest


# ----------------------------------------------------------------------------------------------- check
def _proof_path(anchor: pathlib.Path) -> pathlib.Path:
    return anchor.with_name(anchor.name + ".ots")


def upgrade_in_a_copy(proof: pathlib.Path) -> bool:
    """`ots upgrade` in a scratch copy (it refuses beside an existing .bak, ledger R231); keep the result only if it
    gained a Bitcoin attestation. Returns whether it did."""
    with tempfile.TemporaryDirectory() as d:
        copy = pathlib.Path(d) / proof.name
        shutil.copyfile(proof, copy)
        subprocess.run(["ots", "upgrade", str(copy)], capture_output=True, stdin=subprocess.DEVNULL)
        new = copy.read_bytes()
        try:
            if any(a.kind == "bitcoin" for a in parse_detached(new)[1]):
                proof.write_bytes(new)
                return True
        except ProofError:
            return False
    return False


def check(anchor: pathlib.Path, *, sources, prereg_bytes: bytes | None = None) -> dict:
    """Every link, from public sources. Returns what a redemption records; raises AnchorError on any break."""
    raw = anchor.read_bytes()
    doc = json.loads(raw)
    if doc.get("schema") != SCHEMA:
        raise AnchorError(f"{anchor.name}: not a {SCHEMA} anchor")
    # 1. the preregistration is the one the anchor names
    if prereg_bytes is None:
        p = ROOT / doc["prereg"]
        if not p.is_file():
            raise AnchorError(f"{doc['prereg']} is not in this tree")
        prereg_bytes = p.read_bytes()
    if _sha(_lf(prereg_bytes)) != doc["prereg_sha256"]:
        raise AnchorError(f"{doc['prereg']} is not the preregistration this anchor names (sha256 differs)")
    # 2. the proof is about these anchor bytes, and executes to a Bitcoin block's merkle root
    proof = _proof_path(anchor)
    if not proof.is_file():
        raise AnchorError(f"no timestamp proof beside the anchor ({proof.name})")
    try:
        covered, atts = parse_detached(proof.read_bytes())
    except ProofError as exc:
        raise AnchorError(f"{proof.name}: {exc}") from exc
    if covered.hex() not in (_sha(raw), _sha(_lf(raw))):
        raise AnchorError(f"{proof.name} timestamps other bytes than {anchor.name}")
    btc = sorted((a for a in atts if a.kind == "bitcoin"), key=lambda a: a.height)
    if not btc:
        raise AnchorError(f"{proof.name} has no Bitcoin attestation yet (a calendar receipt only); run `redeem`, "
                          "which upgrades a copy, after a few hours")
    confirmed, bad = None, []
    for a in btc:
        blk = sources.block(a.height)
        if blk["merkle_root"] == a.merkle_root_display:
            confirmed = (a, blk)
            break
        bad.append(a.height)
    if confirmed is None:
        raise AnchorError(f"no attestation executes to its block's merkle root (heights {bad})")
    att, blk = confirmed
    # 3. the block came well before the round
    b = doc["beacon"]
    if b["chain_hash"] != DRAND_CHAIN:
        raise AnchorError("the anchor names another drand chain")
    info = sources.drand_info()
    if (info["hash"], int(info["genesis_time"]), int(info["period"])) != (
            b["chain_hash"], b["genesis_time"], b["period_seconds"]):
        raise AnchorError("the anchor's beacon parameters are not the chain's")
    published = round_time(b["round"], genesis_time=b["genesis_time"], period_s=b["period_seconds"])
    gap = published - blk["timestamp"]
    if gap < doc["margin_hours"] * 3600:
        raise AnchorError(f"block {att.height} ({_utc(blk['timestamp'])}) is only {gap / 3600:.1f} h before round "
                          f"{b['round']} ({_utc(published)}); the anchor requires {doc['margin_hours']} h. The "
                          f"preregistration is not proven to predate its beacon: void, declare a longer lead")
    # 4. the round is out, and both relays give the same value
    rnd = sources.drand_round(b["round"])
    if rnd is None:
        raise AnchorError(f"round {b['round']} is not published yet (due {_utc(published)}); the wait is the "
                          "property, do not substitute another round")
    return {"schema": REDEEMED_SCHEMA, "anchor": anchor.name, "anchor_sha256": _sha(_lf(raw)),
            "prereg_sha256": doc["prereg_sha256"],
            "bitcoin": {"height": att.height, "block_hash": blk["hash"], "merkle_root": blk["merkle_root"],
                        "block_time_utc": _utc(blk["timestamp"])},
            "beacon": {"round": b["round"], "published_utc": _utc(published), "randomness": rnd["randomness"],
                       "signature": rnd["signature"]},
            "hours_between_block_and_round": round(gap / 3600, 2),
            "first_seeds": derive_prereg_seeds(rnd["randomness"], prereg_sha256=doc["prereg_sha256"], n=4)}


def redeemed_path(anchor: pathlib.Path) -> pathlib.Path:
    return anchor.with_name(anchor.name.replace(".anchor.json", ".redeemed.json"))


def redeem(anchor: pathlib.Path, *, sources, upgrade: bool = True) -> dict:
    proof = _proof_path(anchor)
    if upgrade and proof.is_file():
        try:
            pending = not any(a.kind == "bitcoin" for a in parse_detached(proof.read_bytes())[1])
        except ProofError:
            pending = False
        if pending:
            upgrade_in_a_copy(proof)
    rec = check(anchor, sources=sources)
    dest = redeemed_path(anchor)
    if dest.exists() and json.loads(dest.read_text(encoding="utf-8")) != rec:
        raise AnchorError(f"{dest.name} exists and differs from what the public sources give now")
    dest.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    return rec


def verify(anchor: pathlib.Path, *, sources) -> dict:
    rec = check(anchor, sources=sources)
    dest = redeemed_path(anchor)
    if dest.exists():
        if json.loads(dest.read_text(encoding="utf-8")) != rec:
            raise AnchorError(f"{dest.name} does not match what the public sources give")
    return rec


def seeds(anchor: pathlib.Path, n: int) -> list[int]:
    rec = json.loads(redeemed_path(anchor).read_text(encoding="utf-8"))
    return derive_prereg_seeds(rec["beacon"]["randomness"], prereg_sha256=rec["prereg_sha256"], n=n)


# ----------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("declare", "status", "redeem", "verify", "seeds"))
    ap.add_argument("path")
    ap.add_argument("--lead-hours", type=float, default=48.0)
    ap.add_argument("--margin-hours", type=float, default=6.0)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--offline", help="a fixture of recorded source answers instead of the network")
    a = ap.parse_args(argv)
    sources = FixtureSources(pathlib.Path(a.offline)) if a.offline else HttpSources()
    try:
        if a.command == "declare":
            dest = declare(a.path, sources=sources, lead_hours=a.lead_hours, margin_hours=a.margin_hours)
            doc = json.loads(dest.read_text(encoding="utf-8"))
            shown = dest.relative_to(ROOT).as_posix() if dest.is_relative_to(ROOT) else str(dest)
            print(f"ANCHORED {a.path} -> {shown} (+ .ots, a calendar receipt)\n"
                  f"  beacon round {doc['beacon']['round']}, published {doc['beacon']['round_time_utc']}\n"
                  f"  next: commit the anchor and its proof, publish them, wait for the Bitcoin attestation and "
                  f"the round, then `redeem`. Do not draw any data before.")
            return 0
        anchor = pathlib.Path(a.path)
        if a.command == "status":
            doc = json.loads(anchor.read_text(encoding="utf-8"))
            atts = parse_detached(_proof_path(anchor).read_bytes())[1]
            print(f"{anchor.name}: round {doc['beacon']['round']} due {doc['beacon']['round_time_utc']}; proof: "
                  f"{sorted(x.height for x in atts if x.kind == 'bitcoin') or 'calendar receipts only'}")
            return 0
        if a.command == "seeds":
            print(" ".join(str(s) for s in seeds(anchor, a.n)))
            return 0
        rec = (redeem if a.command == "redeem" else verify)(anchor, sources=sources)
        print(f"{'REDEEMED' if a.command == 'redeem' else 'ANCHOR VERIFIED'}: {anchor.name} -- preregistration in "
              f"Bitcoin block {rec['bitcoin']['height']} ({rec['bitcoin']['block_time_utc']}), "
              f"{rec['hours_between_block_and_round']} h before beacon round {rec['beacon']['round']} "
              f"({rec['beacon']['published_utc']}); seeds begin {rec['first_seeds'][0]}")
        return 0
    except (AnchorError, ProofError, OSError, KeyError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
