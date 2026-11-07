"""The preregistered scaling campaign replays from its archived inputs, and only from them.

External audit 3 of release v2026.11.05 (paper A = our paper 2): the campaign's records could be replayed only by
calling the producer's model-building code. The archive (evidence/qldpc_scaling_prereg/inputs/) and its replay
(tools/replay_qldpc_scaling_campaign.py) close that. These tests require:
  * the archive to equal what the producer's build() makes (the one place the producer is imported here), and the
    archived samples to equal a re-draw of the campaign's own seeded samplers;
  * the replay to load no producer-side module at all, in a fresh interpreter;
  * the replay to refuse a tampered weight, a tampered receipt and a swapped input file, each for its own reason --
    with every hash re-bound, so the refusal cannot come from the hash layer -- and an intact copy to pass.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import replay_qldpc_scaling_campaign as rq  # noqa: E402

INPUTS = ROOT / "evidence" / "qldpc_scaling_prereg" / "inputs"
MANIFEST = INPUTS / "MANIFEST.json"
WRITER = ROOT / "experiments" / "m0_prior_art" / "qldpc_scaling_prereg_export_inputs.py"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture(scope="module")
def writer():
    spec = importlib.util.spec_from_file_location("prereg_export_inputs_test", WRITER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prereg_export_inputs_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def runner(writer):
    return writer.load_runner()


def test_the_live_manifest_is_the_pinned_one():
    assert _sha(MANIFEST.read_bytes()) == rq.PINNED_MANIFEST_SHA256


@pytest.mark.parametrize("name", ["uniform_p01", "uniform_p02", "hetero_p02", "phenom_p01", "phenom_p02",
                                  "xsector_p02", "stress_p05", "bb_refsched"])
def test_the_archived_instance_is_what_build_makes(writer, runner, name):
    """Checks, integer weights, decoder input and logical test: the archive against a fresh build()."""
    stored = json.loads((INPUTS / f"{name}.json").read_text(encoding="utf-8"))
    fresh = writer.export(name, runner, samples=False)
    stored.pop("samples")
    assert stored == fresh
    D = runner.build(name, runner.SEED_DOMAIN)
    assert stored["weights"] == [int(D["wint"][i]) for i in range(D["n"])]
    assert stored["decoder"]["channel"] == D["chan"]


def test_the_whole_archive_and_its_manifest_are_current(writer):
    """Every archived file, samples included, equals a fresh export byte for byte: the samples are a re-draw of the
    campaign's seeded samplers, each accepted only where it reproduces the recorded syndrome."""
    res = writer.write_all(check=True)
    assert res["stale"] == [], res


def _copy(tmp: pathlib.Path) -> pathlib.Path:
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    files = {man["summary"]["file"]}
    for d in man["datasets"].values():
        files |= {d["input"]["file"], d["records"]["file"], *d["instances"]}
    for f in files:
        (tmp / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, tmp / f)
    shutil.copy2(MANIFEST, tmp / MANIFEST.relative_to(ROOT))
    return tmp / MANIFEST.relative_to(ROOT)


def _rebind(tmp: pathlib.Path, man_path: pathlib.Path) -> None:
    """Re-hash every bound file into the copy's manifest (and the records into its summary), as a forger would."""
    man = json.loads(man_path.read_text(encoding="utf-8"))
    summ_p = tmp / man["summary"]["file"]
    summ = json.loads(summ_p.read_text(encoding="utf-8"))
    for name, d in man["datasets"].items():
        d["input"]["sha256"] = _sha((tmp / d["input"]["file"]).read_bytes())
        d["records"]["sha256"] = _sha((tmp / d["records"]["file"]).read_bytes())
        summ["datasets"][name]["records_sha256"] = d["records"]["sha256"]
    summ_p.write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8", newline="\n")
    man["summary"]["sha256"] = _sha(summ_p.read_bytes())
    man_path.write_text(json.dumps(man, sort_keys=True, indent=1) + "\n", encoding="utf-8", newline="\n")


def _replay(tmp, man_path, datasets):
    return rq.replay(man_path, pinned=None, root=tmp, datasets=datasets)


def _tags(fails):
    return {f.split("]")[0] + "]" for f in fails}


def test_an_intact_copy_passes(tmp_path):
    man_path = _copy(tmp_path)
    ok, fails, out = _replay(tmp_path, man_path, ["uniform_p01", "stress_p05"])
    assert ok, fails[:5]
    assert out["uniform_p01"]["receipts"] == 1000 and out["stress_p05"]["receipt_kinds"]["tree"] > 0


def test_an_altered_file_is_refused_by_its_hash(tmp_path):
    man_path = _copy(tmp_path)
    p = tmp_path / "evidence" / "qldpc_scaling_prereg" / "inputs" / "hetero_p02.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["weights"][0] += 1
    p.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    ok, fails, _ = _replay(tmp_path, man_path, ["hetero_p02"])
    assert not ok and _tags(fails) == {"[altered]"}, fails[:5]


def test_a_tampered_weight_is_refused_by_the_objective(tmp_path):
    man_path = _copy(tmp_path)
    p = tmp_path / "evidence" / "qldpc_scaling_prereg" / "inputs" / "hetero_p02.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    with gzip.open(ROOT / "evidence" / "qldpc_scaling_prereg" / "hetero_p02.jsonl.gz", "rt", encoding="utf-8") as f:
        first = json.loads(f.readline())
    i = first["answers"]["1.0"]["support"][0]
    doc["weights"][i] += 1                                    # one weight, on a column the first answer uses
    doc["weights_sha256"] = _sha(json.dumps(doc["weights"]).encode())
    p.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    _rebind(tmp_path, man_path)
    ok, fails, _ = _replay(tmp_path, man_path, ["hetero_p02"])
    assert not ok
    assert "[objective]" in _tags(fails) and not _tags(fails) & {"[altered]", "[identity]", "[pin]"}, fails[:5]
    assert any("hetero_p02 shot 0" in f for f in fails if f.startswith("[objective]"))


def test_a_tampered_receipt_is_refused_by_its_checkers(tmp_path):
    man_path = _copy(tmp_path)
    p = tmp_path / "evidence" / "qldpc_scaling_prereg" / "uniform_p01.jsonl.gz"
    with gzip.open(p, "rt", encoding="utf-8") as f:
        recs = [json.loads(x) for x in f]
    R = recs[0]["receipt"]
    assert R["kind"] == "flat"
    R["facet_duals"] = [[r, F, 2 * a, b] for (r, F, a, b) in R["facet_duals"]]   # a dual inflated past the optimum
    p.write_bytes(gzip.compress("".join(json.dumps(r, sort_keys=True) + "\n" for r in recs).encode(), mtime=0))
    _rebind(tmp_path, man_path)
    ok, fails, _ = _replay(tmp_path, man_path, ["uniform_p01"])
    assert not ok
    assert _tags(fails) == {"[receipt]", "[analysis]"}, fails[:5]   # refused, so that shot is missing from the counts
    assert any("uniform_p01 shot 0" in f and "production checker False" in f for f in fails)


def test_a_swapped_input_file_is_refused_by_its_identity(tmp_path):
    man_path = _copy(tmp_path)
    d = tmp_path / "evidence" / "qldpc_scaling_prereg" / "inputs"
    a, b = (d / "uniform_p01.json").read_bytes(), (d / "uniform_p02.json").read_bytes()
    (d / "uniform_p01.json").write_bytes(b)                  # same checks and weights; another dataset's samples
    (d / "uniform_p02.json").write_bytes(a)
    _rebind(tmp_path, man_path)
    ok, fails, _ = _replay(tmp_path, man_path, ["uniform_p01"])
    assert not ok and _tags(fails) == {"[identity]"}, fails[:5]
    assert "uniform_p01: the bound input file is for dataset 'uniform_p02'" in fails[0]


def test_the_replay_imports_no_producer_module():
    """In a fresh interpreter: replay two datasets (flat receipts and trees, both second checkers), then list every
    loaded module. No experiment module, no producer, no solver, no BP-OSD, no sampler, no NumPy."""
    code = ("import json, sys; sys.path.insert(0, sys.argv[1]); import replay_qldpc_scaling_campaign as r; "
            "ok, fails, out = r.replay(datasets=['uniform_p01', 'stress_p05']); "
            "print(json.dumps({'ok': ok, 'fails': fails[:5], 'trees': out['stress_p05']['receipt_kinds']['tree'], "
            "'modules': {k: str(getattr(v, '__file__', None)) for k, v in list(sys.modules.items())}}))")
    res = subprocess.run([sys.executable, "-B", "-c", code, str(ROOT / "tools")], capture_output=True, text=True,
                         cwd=ROOT, timeout=900)
    assert res.returncode == 0, res.stderr[-2000:]
    rep = json.loads(res.stdout.strip().splitlines()[-1])
    assert rep["ok"], rep["fails"]
    assert rep["trees"] > 0
    mods = rep["modules"]
    exp = (ROOT / "experiments").as_posix().lower()
    assert not [k for k, f in mods.items() if f.replace("\\", "/").lower().startswith(exp)]
    for bad in ("numpy", "scipy", "ldpc", "stim", "oneq.qldpc_cert", "qldpc_scaling_prereg_campaign"):
        assert not [k for k in mods if k == bad or k.startswith(bad + ".")], bad
    assert sorted(k for k in mods if k.split(".")[0] == "oneq") == ["oneq", "oneq.qldpc_check"]
