r"""ONE command, EVERY gate. The ritual that ends the drift pattern.

The measured failure mode of this project is not broken code -- the suite
catches that -- it is CLAIMS drifting from artifacts between deliberate
audit passes: stale totals, prose numbers their artifacts contradict,
bundles describing contents they do not carry, mutants whose killers cannot
fire, dead modules nobody would notice going wrong. Each of those got a
gate; this runs all of them, in order, and one red is THE answer.

    python tools/gate_all.py            # everything
    python tools/gate_all.py --fast     # skip the test suite (pre-edit loop)

Order: derived claims regenerate FIRST (so provenance sweeps fresh numbers),
then the suite, then every prose/provenance/registry gate, then the bench
ledger appends its entry. Coverage floors guard the campaign-core modules:
new code arrives tested or this goes red.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable

# Modules whose coverage may never fall below the floor. 0%-modules are
# either legacy lanes with their own gates or audit debt named in
# docs/AUDIT_DEBT.md -- they are not silently exempt.
COVERAGE_FLOORS = {
    "matching_cert.py": 90, "moat_dual.py": 88, "cert_format.py": 80,
    "certifying_decoder.py": 75, "moat_growth.py": 65, "forgeries.py": 90,
    "qldpc_cert.py": 85, "streaming.py": 88, "tjoin_exact.py": 88,
    # The four-class decomposition is the campaign's headline instrument
    # and it lived in experiments/ -- outside every floor, every mutant
    # and every coverage report -- in four copies that had already drifted
    # apart on the tie test. It is a small module with no excuse.
    "decomposition.py": 100,
    # The Phase 4 seal: freeze, RFC-6962 receipt tree, anomaly lane,
    # scoring. Small, pure, and load-bearing for a PUBLISHED artifact, so
    # an uncovered line here is a line no stranger's verification
    # exercises either.
    "rehearsal.py": 100,
    # The beacon is what makes the blind schedule blind. Small,
    # pure, and every uncovered line would be a refusal branch
    # nobody has watched fire.
    "beacon.py": 100,
    # The per-shot body BOTH lanes call. It shipped untested and
    # the wall caught it twice in one run (audit-debt 0%, ratchet
    # coverage 79.2 -> 79.0). A shared builder with no floor is
    # how the four-copy drift started.
    "shot_receipt.py": 100,
    # The enclosure tier. The producer may be as clever as it likes --
    # nothing it does can make a bracket wrong -- but the CHECKER owns
    # every verdict, and an uncovered line there is a refusal path
    # nobody has watched fire.
    "coset_enclosure.py": 90,
    "coset_enclosure_check.py": 90,
    "enclosure_forgery.py": 95,
    # The batch bound. Small, pure, and the only thing standing between a
    # certified posterior and a number somebody quotes as a logical error
    # rate. Every refusal path in it is a place that authority could leak.
    "certified_selection.py": 95,
    # The parity-matching PRODUCER. Not an accept path, but its
    # soundness (never below the oracle) is what lets an upper bound
    # compose with a certified lower bound later. Two real bugs were
    # found in it by its own tests; both are now aimed mutants.
    "coset_matching.py": 100,
    # DEM-CERT. The detector error model is the ancestor of every
    # decoder-assurance leg, so an unexercised line here is a fault class
    # nothing downstream can see. The surface-code circuit walks only
    # H/CX/R/MR, which left CZ, SWAP, S, the X and Y measurement bases
    # and both explicit Pauli channels untouched until they were aimed
    # at -- the floor is what keeps them aimed at.
    "dem_cert.py": 100,
    # The capability negotiation. Its whole job is REFUSING, and a
    # refusal branch nobody exercises is a plan admitted by accident.
    # Both reached 100% once the uncovered lines were READ rather
    # than counted: every one was a refusal branch, which is where
    # a checker's value lives and where this repository has twice
    # found a landmine.
    "capability.py": 100, "capability_probe.py": 100,
    # R_HW(C). Floored at 100 from birth: every branch of it is a REFUSAL
    # path, and an unexecuted refusal is the shape this module exists to
    # stop -- a plan blocked for a reason nobody ever ran.
    "capability_plan.py": 100,
    # The inference engine. An estimator branch that never runs is a
    # rate nobody has seen computed, and every refusal in here is a
    # guard against reporting a number the data did not determine.
    "latent_parity.py": 100,
    # The evidence algebra. Every line of it can only ever make a bound
    # look BETTER than its inputs support, which is the one direction
    # that ships an unearned soundness claim.
    "evidence_algebra.py": 100,
    # The evidence FRONTIER. It exists because two scalars that could
    # desync should not have been two scalars; an unexercised line
    # here is a place they could start disagreeing again.
    "evidence_frontier.py": 100,
    # The WIRING. Every line of it turns a refusal on or off, and a
    # refusal nobody exercises is a claim minted by accident.
    "soundness_budget.py": 100,
    # The admission ladder. Small, pure, and every uncovered line
    # is a rung that could be stepped over without anyone noticing.
    "dem_admission.py": 100,
    # L0 ABFT. It covers the fault class `H x_hat = s` provably
    # cannot see, and both of its named no-op constructions look
    # exactly like the working one -- so an unexercised line here
    # is a layer that might already detect nothing.
    "abft_l0.py": 100,
    # A pin that can name bytes no revision ever held is unfalsifiable:
    # nobody can reproduce it and nobody can refute it. Every line here
    # is a refusal that keeps an orphan hash unconstructible, and one
    # already reached a published binding.
    "git_pin.py": 100,
    # The impossibility certificate. Every branch is a REFUSAL TO PROVE,
    # and an unexercised one is a path that could hand back a skip nobody
    # has watched being declined. It nearly shipped a false skip twice.
    "repair_impossible.py": 100,
    # The robustness radius. It is the one module here whose headline is
    # a THEOREM CANDIDATE rather than a proof, so an unexercised line is
    # a case the falsification never reached.
    "min_ratio_cycle.py": 100,
    # The v2 rehearsal wrapper. Every line is a REFUSAL or a grade, and a
    # wrapper more permissive than the thing it wraps is a bypass with
    # better manners -- an unexercised branch here is where that hides.
    "rehearsal_v2.py": 100,
    # The B2 reduction. Its OUTPUT is three questions somebody will spend
    # solver time on, so a wrong reduction spends it on the wrong ones --
    # and three wrong questions look exactly like three right ones.
    "quotient_reduction.py": 100,
    # The solver portfolio. Its dangerous path is an exhausted budget
    # read as NO -- which would turn every hard instance into a distance
    # claim -- so every engine's budget branch is exercised.
    "distance_portfolio.py": 100,
    # The rank-closure barcode sweep. It was exported in `__all__` with
    # ZERO callers and ZERO tests at 15% coverage while advertising a
    # solve-count collapse, and it stamped every level `exact: True`
    # including the ones at a floor nothing had certified. The floor is
    # 90 rather than 100 because the MILP's refusal branches need a code
    # large enough to time out, which no unit test may own.
    "quotient_escape.py": 90,
    # The exact per-shot error budget and the 16-cell cube. Its whole
    # value is that a term charged to the wrong stage is VISIBLE, so an
    # unexercised branch is a stage that could be miscounted silently --
    # and the hardware spend turns on the search-vs-model distinction it
    # draws. `docs/HARDWARE_PREFLIGHT_GATES.md` names it a pilot
    # prerequisite.
    "error_budget.py": 100,
    # The published counterexamples to the dual form. An unexercised arm
    # is an attack nobody runs, in the battery that stands behind the
    # repository's central claim.
    "dual_forgery.py": 100,
}


#: pytest's own exit codes. Anything outside this range from a pytest
#: gate means the interpreter never got to report, which is a different
#: event with different causes and a different fix.
_PYTEST_EXITS = {
    1: "tests FAILED -- pytest's own output above names them",
    2: "interrupted",
    3: "internal error",
    4: "usage error",
    5: "NO TESTS COLLECTED -- a selector matched nothing, which is a dark "
       "gate, not a pass",
}


#: Signatures of a process that DIED rather than decided. An unhandled
#: Python exception exits 1 -- the same code a tool uses to say "no" --
#: so the exit code alone cannot tell a verdict from a death.
_DEATH_MARKS = ("Traceback (most recent call last):", "MemoryError",
                "std::bad_alloc", "RecursionError", "Killed",
                "MemoryError: std::bad_alloc")


def looks_like_a_death(output: str | None) -> bool:
    r"""Did this output come from a crash rather than from a decision?

    *** MEASURED 2026-08-24, AND THE WALL SAID THE OPPOSITE. ***
    `closures re-decided independently (HiGHS)` exited 1 after HiGHS
    raised `MemoryError: std::bad_alloc` under memory pressure, and the
    wall printed:

        'closures re-decided independently (HiGHS)' refused with a
        documented exit code -- its own output above is the verdict,
        not a crash report.

    The output above it was a Python traceback ending in MemoryError.
    It was exactly a crash report. Re-run on a quiet machine the same
    command confirms 2/2 closures and exits 0, so the "verdict" the wall
    reported as the tool's own was a fact about the machine's RAM.

    A gate that cannot separate "the tool decided NO" from "the tool
    died" will, under load, present a resource failure as a soundness
    finding -- and that is the direction that costs the most, because a
    reader believes it.
    """
    if not output:
        return False
    return any(mark in output for mark in _DEATH_MARKS)


def _explain_exit(name, cmd, code: int, output: str | None = None) -> None:
    """Say whether the gate FAILED or DIED, because they are not the same.

    *** THIS EXISTS BECAUSE A GATE WENT RED HAVING PRINTED THIRTEEN DOTS
    AND NOTHING ELSE. *** `d=8 branching certificates` died partway
    through 16 tests; standalone the same command passed 16/16 in 146 s.
    The wall reported `RED   d=8 branching certificates` and named no
    test, gave no exit code and offered no next step -- output almost
    indistinguishable from a gate that failed an assertion, and useless
    for telling the two apart.

    A red that cannot say WHY is the same defect class this repository
    keeps finding elsewhere: a check whose output does not depend on what
    happened. So the exit code is printed, and when it is outside the
    range the tool itself can produce, the three causes this machine
    ACTUALLY has a history of are named with the command to check each.
    None of them is asserted here -- they are where to look.
    """
    is_pytest = "pytest" in cmd
    print(f"\n  EXIT {code} (0x{code & 0xFFFFFFFF:08X})", flush=True)
    if is_pytest and code in _PYTEST_EXITS:
        print(f"  {_PYTEST_EXITS[code]}", flush=True)
        return
    # AN ORDINARY TOOL EXIT IS A VERDICT, NOT A DEATH, and the first
    # version of this got that wrong on its first real firing. The
    # ratchet gate exits 2 to say "MUTATION REPORT STALE: it covers 147
    # mutants but 150 are registered" -- a documented, correct, entirely
    # legible refusal -- and this printed crash diagnostics under it,
    # sending a reader to check free RAM and Defender for a gate that had
    # already said precisely what was wrong.
    #
    # A red that misdiagnoses is worse than one that only says RED,
    # because the reader now has somewhere confidently wrong to go. So
    # the death branch is reserved for codes no tool chooses: 126+ and
    # the Windows exception codes, which is the convention both
    # platforms already use.
    if not is_pytest and 0 < code < 126 and not looks_like_a_death(output):
        note = ("" if output is not None else
                " (classified from the exit code alone -- this node's "
                "output was streamed, not captured)")
        print(f"  {name!r} refused with a documented exit code -- its own "
              f"output above is the verdict, not a crash report.{note}",
              flush=True)
        return
    if not is_pytest and 0 < code < 126:
        # THE CODE SAYS VERDICT AND THE OUTPUT SAYS DEATH. An unhandled
        # Python exception exits 1, the same code a tool chooses to say
        # "no", so the output is the only thing that separates them.
        print(f"  *** {name!r} EXITED {code} WITH A CRASH IN ITS OUTPUT. "
              f"*** That code is one a tool chooses, but the output "
              f"above ends in a traceback -- this is a DEATH being "
              f"reported as a verdict. Do not read it as a finding "
              f"about the subject until the command is re-run "
              f"standalone:", flush=True)
    if is_pytest:
        print("  *** THE PROCESS DIED BEFORE PYTEST COULD REPORT. *** That "
              "is not a failing test; no assertion was reached. On this "
              "machine it has three known causes, in the order they have "
              "actually occurred:", flush=True)
    else:
        print(f"  *** {name!r} EXITED {code}, which is not a code a tool "
              f"chooses. *** Treating it as a death rather than a verdict:",
              flush=True)
    print("    1. MEMORY PRESSURE, not a hang. Check free RAM and whether "
          "another project is running:\n"
          "         Get-CimInstance Win32_OperatingSystem | "
          "select FreePhysicalMemory\n"
          "       Never kill the neighbour to make room -- that is "
          "someone else's run.", flush=True)
    print("    2. DEFENDER quarantining a freshly built artifact under "
          "%TEMP% (Errno 22 is quarantine, not I/O):\n"
          "         Get-MpThreatDetection | select -Last 5 "
          "InitialDetectionTime,Resources", flush=True)
    print("    3. A HARD CRASH that skips SEH and leaves no traceback:\n"
          "         Get-WinEvent -FilterHashtable @{LogName='Application'} "
          "-MaxEvents 20", flush=True)


#: Free RAM that must REMAIN after the test swarm is launched. Not the
#: being's size -- the being is already resident and its memory is not
#: ours to subtract. The measured danger line is 0.8-1.1 GB free (the
#: being was trimmed and froze there on 2026-09-03 and again on 09-04);
#: a 1-worker wall ran beside a recovering being at 5.1 GB free without
#: incident. 3 GB remaining is three times the freeze margin.
FREE_FLOOR_GB = 3.0

#: Measured cost of one xdist worker running this suite.
WORKER_GB = 0.7


def _auto_jobs() -> int:
    r"""Workers, sized by MEMORY as well as cores.

    *** THIS WAS `max(2, min(8, cpu_count // 3))` AND IT FROZE THE
    BEING. *** On a 24-core workstation that is eight full pytest
    processes and no consideration of whether the RAM exists. On
    2026-09-03 the swarm plus a bakeoff drove available memory to about
    1 GB; Windows trimmed the live A.C.E being from 4.6 GB to 1.3 GB and
    it then could not page back in, sitting at zero CPU for roughly 50
    minutes. The wall's own load interlock did not help: it guards
    ONE_Q against ONE_Q and knows nothing about anything else on the
    box.

    So the cap is the smaller of what the cores want and what the RAM
    allows after the being's reserve.

    *** IT REFUSES AT ZERO. IT DOES NOT SCALE TO ONE. *** The first
    version floored at one worker on the theory that a gate which
    declines to run is a gate nobody has run. Another session then
    MEASURED the counter-example on 2026-09-04: a collection clamped to
    ONE worker, about 0.3 GB, froze the being for 33+ minutes -- both
    processes at 0 CPU in page-in wait. A long job denied memory does
    not go slower; it deadlocks the thing it was sized to protect. So
    when the headroom does not cover a single worker this returns 0 and
    the caller refuses the suite stage with a documented exit, and the
    honest alarm is the being LOSING resident memory, not low free RAM
    alone. See memory `a_resident_being_consumes_headroom_it_does_not_free_it`.
    """
    cpu_cap = max(2, min(8, (os.cpu_count() or 4) // 3))
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    except Exception as exc:                              # pragma: no cover
        print(f"  worker sizing: psutil unavailable ({exc}); falling back "
              f"to {cpu_cap} on cores alone (declared, not silent)")
        return cpu_cap
    # *** THE FIRST VERSION SUBTRACTED THE BEING'S SIZE AND REFUSED AT
    # 6.7 GB FREE. *** It read `(avail - 6 GB) // 0.7`, which is the right
    # shape for a box where the being is NOT yet resident and wrong on
    # this one, where it is: the 6 GB is already spent, and what protects
    # the being is the free RAM that REMAINS. Measured consequence: the
    # headroom runner launched at >= 5 GB free and every wall it launched
    # refused its own suite stage, so coverage.json was never regenerated
    # and the mutation baseline stayed red on a module that was absent
    # from a stale artifact -- two guards with inconsistent thresholds.
    mem_cap = int((avail_gb - FREE_FLOOR_GB) // WORKER_GB)
    if mem_cap < 1:
        print(f"  worker sizing: {avail_gb:.1f} GB available; "
              f"{FREE_FLOOR_GB:.0f} GB must remain free -> headroom covers 0 "
              f"workers. REFUSING the suite stage: a 1-worker run at ~1 GB "
              f"free measured as a 33-minute being freeze, not a slow pass. "
              f"Re-run when free RAM exceeds "
              f"{FREE_FLOOR_GB + WORKER_GB:.1f} GB, or on a second machine.")
        return 0
    jobs = min(cpu_cap, mem_cap)
    if jobs < cpu_cap:
        print(f"  worker sizing: {avail_gb:.1f} GB available, "
              f"{FREE_FLOOR_GB:.0f} GB kept free -> {jobs} worker(s), not "
              f"{cpu_cap}. The wall is re-runnable; the being is not.")
    return jobs


#: how long the suite stage will wait for memory before recording RED, and
#: how often it re-checks. Bounded on purpose: a wall that hangs is worse
#: than one that reports.
SUITE_WAIT_S = 45 * 60
SUITE_POLL_S = 60


def _wait_for_workers(budget_s: float = SUITE_WAIT_S,
                      poll_s: float = SUITE_POLL_S,
                      sizer=None, sleep=None, clock=None) -> int:
    r"""Wait, bounded, for enough memory to size one worker.

    *** THE INTERLOCK CHECKED HEADROOM AT THE WRONG MOMENT, AND A REAL RUN
    WAS THROWN AWAY FOR IT. *** MEASURED 2026-09-10: the wall's start-up
    interlock cleared at 08:30:10 with headroom to spare, the wall then
    spent **twenty-five minutes** on the other gates, and at 08:55 the
    suite stage -- the one stage whose result anybody wants -- found the
    memory gone and recorded RED. Every gate before it had passed. The
    verdict said RED for a reason that says nothing about the code, and
    the 25 minutes were spent for nothing.

    The check was not wrong; its TIMING was. A resource read at `t = 0`
    does not describe `t = 25 min` on a box shared with an eight-hour
    search job and a service that leaks commit. So the stage that needs
    the memory now waits for it, at the moment it needs it, instead of
    refusing on a reading taken before the wall started.

    It stays BOUNDED and it stays RED at the end: a gate that waits
    forever is a gate nobody can run in CI, and the refusal at zero
    workers is itself a measured protection (2026-09-04: one worker at
    ~0.3 GB froze the resident being for 33+ minutes -- a long job denied
    memory does not go slower, it deadlocks the thing the sizing exists to
    protect). Waiting can only turn a wasted run into a complete one; it
    never lowers the floor.
    """
    sizer = sizer or _auto_jobs
    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    jobs = sizer()
    if jobs > 0 or budget_s <= 0:
        return jobs
    deadline = clock() + budget_s
    print(f"  waiting up to {budget_s / 60:.0f} min for memory rather than "
          f"discarding a run that has already passed every earlier gate; "
          f"re-checking every {poll_s:.0f} s", flush=True)
    while clock() < deadline:
        sleep(poll_s)
        jobs = sizer()
        if jobs > 0:
            print(f"  memory freed -> {jobs} worker(s); running the suite",
                  flush=True)
            return jobs
    print(f"  still no headroom after {budget_s / 60:.0f} min. Recording RED "
          f"-- a fact about the box, not about the code. Holding it:",
          flush=True)
    for line in biggest_holders():
        print(f"    {line}", flush=True)
    return 0


def biggest_holders(n: int = 3, procs=None) -> list[str]:
    r"""The `n` largest commit holders, named. PURE given `procs`.

    *** A RED THAT CANNOT SAY WHY IS THE DEFECT THIS REPOSITORY KEEPS
    FINDING, AND "NO MEMORY" WITHOUT A NAME IS ONE. *** The reader's next
    question is always "held by what", and the answer has twice been a
    service leaking commit with a working set near zero -- invisible to
    Task Manager's default column and to any check that looks at RAM in
    use. So the tell is printed: a large PRIVATE with a small WORKING SET
    is a process holding memory it never touches.
    """
    if procs is None:
        try:
            import psutil
            procs = []
            for p in psutil.process_iter(["name", "pid", "memory_info"]):
                mi = p.info.get("memory_info")
                if mi:
                    procs.append((p.info["name"], p.info["pid"],
                                  mi.private / 2 ** 30 if hasattr(mi, "private")
                                  else mi.vms / 2 ** 30, mi.rss / 2 ** 30))
        except Exception:                                   # noqa: BLE001
            return ["(could not read the process table -- which is not the "
                    "same as nothing holding memory)"]
    out = []
    for name, pid, priv, rss in sorted(procs, key=lambda r: -r[2])[:n]:
        tell = ("  <- large private, small working set: memory held and "
                "never touched" if priv > 4 and rss < priv / 8 else "")
        out.append(f"{name} pid {pid}: {priv:.1f} GB private, "
                   f"{rss:.1f} GB working set{tell}")
    return out


def _has_xdist() -> bool:
    """Parallel test execution available, checked rather than assumed.

    If it is missing the suite still runs -- serially, slowly, and SAYING
    SO. Silently falling back to the slow path is how a 12-minute gate
    becomes something everybody works around instead of fixing.
    """
    import importlib.util
    return importlib.util.find_spec("xdist") is not None


def _already_running():
    """Another heavy ONE_Q job in flight, by explicit command match.

    *** THE WALL STARVED THE WORKSTATION AND FROZE THE EDITOR. *** Two of
    these ran concurrently -- a full `gate_all.py` plus a separate
    `pytest tests/` with eight xdist workers -- and between them took the
    machine to 8 GB free of 31, with a third project's daemon also on it.
    The symptom was VS Code becoming unresponsive, which points nowhere
    near the cause.

    Matched on the SPECIFIC command, never on a bare "python": a filter
    that catches its own invocation, or a neighbour's, is the recorded
    way to kill the run you meant to keep. Returns a list of (pid, cmd).
    """
    return scan_processes(_process_lines(), os.getpid(),
                          excluded_pids=_own_process_family())


def _own_process_family() -> set[int]:
    """This interpreter and its launchers, which are one invocation.

    The Microsoft Store ``python.exe`` alias remains alive as a Python-named
    parent while the real interpreter runs. Process-name filtering therefore
    sees two ``gate_all.py`` processes unless ancestors are excluded.
    """
    family = {os.getpid()}
    try:
        import psutil
        family.update(parent.pid for parent in psutil.Process().parents())
    except Exception:                                       # noqa: BLE001
        pass
    return family


def _is_heavy_command(command: str) -> bool:
    """Whether *command* can contend with, or observe, the full wall.

    Paths are normalised before matching because the same invocation is
    rendered with ``/`` by POSIX tools and ``\\`` by Windows process APIs.
    A mutation gate is included even when it uses its normal sandbox: it is
    deliberately CPU-heavy, and an explicitly requested in-place run can
    transiently replace files the wall imports.
    """
    normal = command.lower().replace("\\", "/")
    if "gate_all.py" in normal or "mutation_gate.py" in normal:
        return True
    return "pytest" in normal and (
        " tests/" in normal or normal.rstrip().endswith(" tests")
    )


def _process_lines():
    """Raw process listing, or None when it could not be obtained.

    *** "COULD NOT LOOK" AND "NOTHING RUNNING" MUST NOT BE THE SAME
    VALUE. *** The first version opened with

        if sys.platform != "win32":
            return []

    and an empty list means ALL CLEAR. So on Linux and macOS -- and on
    Windows whenever wmic was absent, blocked or slow -- the interlock
    reported that nothing heavy was running WITHOUT HAVING LOOKED. A
    skip read as a pass, on the check that exists because two walls at
    once took this workstation to 8 GB free and froze the editor.

    `tools/mutation_gate.py` had exactly this defect and had already
    been repaired, with the note that "portability alone would not have
    fixed it; it would have moved the silence to the next environment
    nobody tested." I reproduced it in a new file the same week, so this
    one both LOOKS on POSIX and says when it could not.
    """
    # Prefer a process API over a shell command. ``wmic`` is absent on current
    # Windows installations, which made this interlock return UNKNOWN on the
    # machine it was written to protect.
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            lines = []
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    name = (proc.info.get("name") or "").lower()
                    if not ("python" in name or "pytest" in name
                            or name in {"py", "py.exe"}):
                        continue
                    command = " ".join(proc.info.get("cmdline") or ())
                    lines.append(f"{proc.info['pid']} {name} {command}")
                except Exception:                           # process exited
                    continue
            return lines
        except Exception:                                   # noqa: BLE001
            pass

    procfs = pathlib.Path("/proc")
    if procfs.is_dir():
        try:
            lines = []
            for entry in procfs.iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    name = (entry / "comm").read_text(
                        encoding="utf-8", errors="replace").strip().lower()
                    if not ("python" in name or "pytest" in name
                            or name == "py"):
                        continue
                    command = (entry / "cmdline").read_bytes().replace(
                        b"\0", b" ").decode("utf-8", "replace")
                except OSError:
                    continue
                lines.append(f"{entry.name} {name} {command}")
            return lines
        except Exception:                                   # noqa: BLE001
            pass

    commands = (
        ["wmic", "process", "where",
         "name like 'python%.exe' or name like 'pytest%.exe'", "get",
         "ProcessId,CommandLine", "/format:csv"],
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process | Where-Object { "
         "$_.Name -match '^(python|pytest|py)' } | ForEach-Object { "
         "$_.ProcessId.ToString() + ' ' + $_.CommandLine }"],
        ["pwsh", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process | Where-Object { "
         "$_.Name -match '^(python|pytest|py)' } | ForEach-Object { "
         "$_.ProcessId.ToString() + ' ' + $_.CommandLine }"],
    )
    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=30)
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.splitlines()

    # Last portable fallback. Filter by the executable name before handing
    # lines to the command matcher so the shell that launched this gate does
    # not look like a second gate merely because its argv contains ours.
    try:
        result = subprocess.run(["ps", "-eo", "pid=,comm=,args="],
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        result = None
    if result is not None and result.returncode == 0:
        lines = []
        for line in result.stdout.splitlines():
            parts = line.strip().split(None, 2)
            name = parts[1].lower() if len(parts) >= 2 else ""
            if "python" in name or "pytest" in name or name == "py":
                lines.append(line)
        return lines
    return None


def scan_processes(lines, me: int, *, excluded_pids=()):
    """[(pid, command)] for heavy jobs; None when the listing is absent.

    Pure, so both platforms' formats are testable anywhere -- which is
    the point: with verification running locally on one OS, a check that
    only works on that OS has nobody left to catch it.
    """
    if lines is None:
        return None
    busy = []
    excluded = {int(pid) for pid in excluded_pids}
    excluded.add(int(me))
    for line in lines:
        if not _is_heavy_command(line):
            continue
        stripped = line.strip()
        # wmic CSV puts the pid LAST; `ps -eo pid=,args=` puts it FIRST.
        tail = stripped.rsplit(",", 1)[-1].strip()
        head = stripped.split(None, 1)[0] if stripped else ""
        pid = tail if tail.isdigit() else head
        if pid.isdigit() and int(pid) not in excluded:
            busy.append((pid, stripped[:100]))
    return busy


#: The recorded freeze point is 8 GB available of 31.4. A floor set AT the
#: observed failure admits the failure, so it sits above it.
MIN_AVAILABLE_GB = 10.0
MIN_COMMIT_HEADROOM_GB = 8.0

#: *** THE FLOOR MUST MATCH THE HAZARD THE MODE ACTUALLY CREATES. ***
#: MEASURED 2026-09-10: `--fast` was refused at 7.7 GB available against the
#: 10 GB floor -- a floor whose entire stated justification is the TEST SUITE
#: ("took this box to 8 GB free of 31.4 and froze the editor"). `--fast`
#: SKIPS the suite. So the cheap pre-edit loop was blocked for a hazard it
#: cannot create: the premise had flipped, and a gate whose premise has
#: flipped can only fail.
#:
#: The answer is not to exempt `--fast` -- it still spawns single-process
#: gates and a truly dead box fails those too (at ~0.3 GB commit headroom
#: this machine could not `fork` bash at all, error 0xC0000142). It is to
#: give it a floor sized to what it actually does: one subprocess at a time,
#: a few hundred MB each, against the suite's 6-8 concurrent xdist workers.
MIN_AVAILABLE_FAST_GB = 3.0
MIN_COMMIT_HEADROOM_FAST_GB = 4.0


def _commit_headroom_gb() -> float:
    r"""Free COMMIT, which is not free physical memory and not swap.

    *** A DERIVED NUMBER IN A REFUSAL MESSAGE HAS TO BE THE RIGHT NUMBER.
    *** The first version estimated this as
    ``(total + swap_total) - used - swap_used`` from psutil and reported
    **50.9 GB** where Windows' own `\Memory\Commit Limit` counter said
    **12.3 GB** -- a 4x overstatement, in the direction that would let a
    starved box through. `ullAvailPageFile` from `GlobalMemoryStatusEx` IS
    the quantity, so it is read rather than reconstructed.

    Off Windows this returns infinity: the recorded hazard is a Windows
    service leaking commit, and the available-memory floor still applies.
    """
    if sys.platform != "win32":
        return float("inf")
    import ctypes

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    st = _MemoryStatusEx()
    st.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        raise OSError("GlobalMemoryStatusEx failed")
    return st.ullAvailPageFile / 2 ** 30


def _headroom():
    """``(available_gb, commit_headroom_gb)``, or ``None`` if unmeasurable."""
    try:
        import psutil
        return (psutil.virtual_memory().available / 2 ** 30,
                _commit_headroom_gb())
    except Exception:                                       # noqa: BLE001
        return None


def _headroom_verdict(headroom, runs_suite: bool = True) -> tuple[bool, str]:
    r"""Refuse when the machine cannot take the wall. PURE.

    *** THE INTERLOCK CLAIMED TO PROTECT MEMORY AND COULD NOT SEE MEMORY.
    *** Its whole justification is a load argument -- "took the machine to
    8 GB free of 31 and froze the editor" -- but it decided purely on
    whether some other heavy PROCESS existed, which fails in both
    directions:

      * OVER-REFUSAL. MEASURED 2026-09-09: four `pytest tests/bootstrap`
        runs in `Coherence/wt_*` worktrees, a different repository, blocked
        the ONE_Q wall. On a box where a peer session runs tests
        continuously that is not a delay, it is an unrunnable gate -- and
        walls did wait hours for exactly this.
      * UNDER-REFUSAL, WHICH IS THE DANGEROUS ONE. Also MEASURED
        2026-09-09: AppXSvc alone held 18.74 GB of commit with a 0.24 GB
        working set, and the box sat at 3.8 GB available of 31.4 -- HALF
        the recorded freeze point -- with no heavy peer process at all. The
        interlock would have started the wall into that and said nothing,
        which is precisely the incident it exists to prevent.

    So the hazard it names is now measured directly, and a process scan is
    kept for what it is actually evidence of: a tree collision, and
    incoming load an instantaneous reading has not seen yet.
    """
    if headroom is None:
        return True, (
            "REFUSING: could not measure memory headroom, and 'could not "
            "look' must not read the same as 'plenty free'. Install psutil "
            "or pass --force after checking Task Manager.")
    avail, commit = headroom
    # The floor is chosen by what THIS RUN will do, not by what the wall can
    # do at its heaviest. See MIN_AVAILABLE_FAST_GB.
    min_avail = MIN_AVAILABLE_GB if runs_suite else MIN_AVAILABLE_FAST_GB
    min_commit = (MIN_COMMIT_HEADROOM_GB if runs_suite
                  else MIN_COMMIT_HEADROOM_FAST_GB)
    if avail < min_avail or commit < min_commit:
        if not runs_suite:
            return True, (
                f"REFUSING even --fast: {avail:.1f} GB available and "
                f"{commit:.1f} GB of commit headroom, against the reduced "
                f"floors of {min_avail:.0f} and {min_commit:.0f} GB that "
                f"--fast is held to. This mode skips the test suite, so it "
                f"is judged on what it does spawn -- single gates, one at a "
                f"time. Under this, subprocess creation itself starts "
                f"failing: at ~0.3 GB commit headroom this box could not "
                f"fork bash (0xC0000142). Check for a service holding "
                f"commit with a small working set.")
        return True, (
            f"REFUSING: {avail:.1f} GB available and {commit:.1f} GB of "
            f"commit headroom, against floors of {min_avail:.0f} and "
            f"{min_commit:.0f} GB. The wall took this box to "
            f"8 GB free of 31.4 once and froze the editor; a floor at the "
            f"observed failure point would admit the failure, so it sits "
            f"above it. Check for a service holding commit with a small "
            f"working set -- AppXSvc has leaked 11-18.7 GB repeatedly and "
            f"needs an ELEVATED `Restart-Service AppXSvc -Force`.")
    return False, ""


def _same_tree(pid) -> bool | None:
    """Is *pid* working inside THIS repository? ``None`` when unreadable.

    A heavy job in this tree is refused for a reason that has nothing to do
    with load: it clobbers `.coverage` and `coverage.json`, and a long gate
    cannot finish in a tree another process is editing.
    """
    try:
        import psutil
        cwd = pathlib.Path(psutil.Process(int(pid)).cwd()).resolve()
    except Exception:                                       # noqa: BLE001
        return None
    return cwd == ROOT or ROOT in cwd.parents


def _partition_busy(busy, same_tree=_same_tree):
    """Split heavy jobs into (this tree, elsewhere). Unreadable counts as
    THIS TREE, because "could not look" may not read as "somewhere else"."""
    here, elsewhere = [], []
    for pid, cmd in busy:
        (elsewhere if same_tree(pid) is False else here).append((pid, cmd))
    return here, elsewhere


def _interlock_verdict(busy, force: bool, headroom=None,
                       same_tree=_same_tree,
                       runs_suite: bool = True) -> tuple[bool, str]:
    """Return ``(refuse, explanation)`` for the wall's process interlock.

    ``runs_suite`` is False for ``--fast``, which skips the test suite. Only
    the MEMORY floor relaxes with it -- see ``MIN_AVAILABLE_FAST_GB``. The
    tree-collision and could-not-scan refusals below are NOT about load and
    do not move: ``--fast`` still writes gate outputs into this tree, and a
    long gate still cannot finish in a tree another process is editing.
    """
    if force:
        return False, ""
    if busy is None:
        return True, (
            "REFUSING: the process interlock could not determine whether "
            "another full wall, mutation gate, or full test suite is live. "
            "Install psutil or provide ps/PowerShell, or pass --force only "
            "after checking the process table manually.")
    here, elsewhere = _partition_busy(busy, same_tree)
    if here:
        # A HEAVY JOB IN THIS TREE IS A COLLISION, NOT A LOAD PROBLEM. It
        # clobbers `.coverage` and `coverage.json`, and a long gate cannot
        # finish in a tree another process edits. No amount of free memory
        # makes that safe, so this refusal has no headroom escape.
        return True, ("REFUSING: another heavy job is running IN THIS TREE. "
                      "It would clobber .coverage and coverage.json, and a "
                      "long gate cannot finish in a tree another process is "
                      "editing. A pid whose working directory could not be "
                      "read is counted here, because 'could not look' must "
                      "not read as 'somewhere else'.")
    # NOT NECESSARILY A ONE_Q JOB, AND SAYING SO SENT AN OPERATOR TO THE
    # WRONG REPOSITORY. MEASURED 2026-09-03: the wall refused against
    # `pytest tests/test_idem_authority_charge.py` -- an Idem run whose
    # command line does not contain "ONE_Q" at all -- while naming ONE_Q as
    # the culprit. The refusal was right and the subject was wrong.
    #
    # It is now decided on the MEASURED hazard rather than on the presence
    # of a neighbour, because identity was wrong in both directions -- see
    # `_headroom_verdict`. A foreign job with room to spare no longer
    # blocks; a starved box now does, whether or not a neighbour is there.
    refuse, why = _headroom_verdict(
        _headroom() if headroom is None else headroom, runs_suite=runs_suite)
    if refuse:
        return True, why
    if elsewhere:
        print(f"NOTE: {len(elsewhere)} heavy job(s) running elsewhere on this "
              f"box, and the headroom floors are met, so proceeding:")
        for pid, cmd in elsewhere:
            print(f"  pid {pid}: {cmd[:88]}")
    return False, ""


def run(name, cmd, *, env=None):
    print(f"\n=== {name} ===", flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        _explain_exit(name, cmd, r.returncode)
    return (name, r.returncode == 0, dt)


#: Gates started EARLY and collected at the end.
#:
#: After the suite was parallelised the wall's own timing table showed
#: the remaining cost was three long, independent subprocesses -- an
#: independent MILP re-decision and the two arbiter sensitivity proofs --
#: worth 217 s of a 431 s serial wall while 24 cores sat idle.
#:
#: ONLY GATES THAT SHARE NO OUTPUT FILE GO HERE, checked rather than
#: assumed: each writes a distinct artifact or none at all, and none
#: reads what another writes. Anything that consumes `derived_claims`,
#: `coverage.json` or the mutation artifact stays in the serial order it
#: depends on -- concurrency that reorders a dependency is not a speedup,
#: it is a race that passes most of the time.
_DEFERRED: list = []


#: Deferred gates that produce a TRACKED artifact, staged and published
#: only after the collection point. {name: (staging path, real path)}
_STAGED: dict = {}

#: Names whose producer both exited successfully and created its previously
#: absent staging file during this run.  Publication is permitted only for
#: this set; mere file existence is not provenance.
_STAGED_READY: set[str] = set()


def defer(name, cmd, *, env=None, publishes=None):
    """Queue an independent gate to run alongside the serial ones.

    *** A DEFERRED GATE THAT WRITES TRACKED EVIDENCE RACES THE SUITE. ***
    Measured: `evidence/phase7/oisc_d8_independent.json` is written by
    the HiGHS re-decision, which this scheme launches BEFORE the suite
    and lets run alongside it. `tests/test_published_bytes.py` compares
    every tracked file under `evidence/` against the object database --
    so it read that file while another process was rewriting it, and the
    wall went red on a clean, fully committed tree.

    When I parallelised I checked that no two DEFERRED gates share an
    output. That was the wrong set. The suite is a reader of EVERY
    evidence file, so any deferred writer races it, and the check I
    performed could not have found this.

    `publishes` names the artifact. The gate is given a staging path
    instead, and the bytes are moved into place during collection --
    after the suite has finished reading the tree.
    """
    if publishes:
        import tempfile
        # A deterministic path in the shared temp directory retained bytes
        # from an earlier crashed run.  A later producer that exited zero
        # without writing anything then "published" those stale bytes.  Give
        # every registration a private run directory instead; _start_deferred
        # also proves the output is absent immediately before launch.
        stage_dir = pathlib.Path(tempfile.mkdtemp(prefix="oneq_staged_"))
        staged = stage_dir / pathlib.Path(publishes).name
        _STAGED[name] = (staged, ROOT / publishes)
        cmd = [str(staged) if c == publishes else c for c in cmd]
    _DEFERRED.append((name, cmd, env))


def _publish_staged() -> None:
    """Atomically publish only outputs linked to a green producer run."""
    import os
    import shutil
    import tempfile

    for name, (staged, real) in _STAGED.items():
        try:
            if name not in _STAGED_READY:
                print(f"  {name}: no publication (producer/output gate was "
                      f"not green)", flush=True)
                continue
            if not staged.is_file():
                # This should be impossible after collection, but fail closed
                # if another process removed the file between the two steps.
                print(f"  {name}: no publication (staged output vanished)",
                      flush=True)
                continue
            real.parent.mkdir(parents=True, exist_ok=True)
            # Do not expose a partially-copied tracked artifact.  The stage
            # directory may live on another volume, so copy into a sibling
            # temp file and make only the final rename atomic.
            with tempfile.NamedTemporaryFile(
                    dir=real.parent, prefix=f".{real.name}.", suffix=".tmp",
                    delete=False) as fh:
                publish_tmp = pathlib.Path(fh.name)
                with staged.open("rb") as src:
                    shutil.copyfileobj(src, fh)
                fh.flush()
                os.fsync(fh.fileno())
            try:
                os.replace(publish_tmp, real)
            finally:
                publish_tmp.unlink(missing_ok=True)
            print(f"  published {real.relative_to(ROOT)}", flush=True)
        finally:
            # The directory was created solely for this registration.  Its
            # output is no longer useful after publish/refusal.
            shutil.rmtree(staged.parent, ignore_errors=True)


def _start_deferred():
    """Launch the deferred gates now; they run while the wall continues."""
    # *** A FILE, NOT A PIPE. *** The first version gave each child a
    # `subprocess.PIPE` and drained none of them until the end. An OS
    # pipe buffer is tens of kilobytes: a gate that prints more than that
    # BLOCKS, forever, and takes the whole wall with it. None of these
    # three happened to be chatty enough to hit it, which is not a
    # property anybody should rely on -- the next deferred gate would
    # have inherited a deadlock that only appears when a gate has
    # something to say.
    #
    # It also explains the timing: with nothing draining them the three
    # gates reported the same duration to a tenth of a second, twice
    # running, because their cost was being read off the collection point
    # rather than off their own exits.
    import tempfile
    started = []
    for name, cmd, env in _DEFERRED:
        staged_info = _STAGED.get(name)
        if staged_info is not None:
            staged, _real = staged_info
            # Absence immediately before Popen is the creation witness.  It
            # also makes the invariant robust if a caller supplied/reused a
            # path or a prior attempt crashed after registration.
            staged.unlink(missing_ok=True)
            _STAGED_READY.discard(name)
        t0 = time.perf_counter()
        log = tempfile.NamedTemporaryFile(
            mode="w+", encoding="utf-8", errors="replace",
            suffix=".gatelog", delete=False)
        p = subprocess.Popen(cmd, cwd=ROOT, env=env,
                             stdout=log, stderr=subprocess.STDOUT)
        started.append((name, cmd, p, t0, log))
    if started:
        print(f"\n=== {len(started)} independent gates started in "
              f"parallel; their output is collected at the end ===",
              flush=True)
        for row in started:
            print(f"  ... {row[0]}", flush=True)
        # *** A WATCHER, BECAUSE NOBODY WAS LOOKING UNTIL THE END. ***
        # The completion time was being stamped the first time anything
        # called `poll()` -- and that only happened in the collector,
        # after every serial gate had finished. So all three deferred
        # gates were recorded as taking almost exactly the wall's own
        # duration: 219.4 s, then 190.2 s, then 186.6 s, three times
        # running, for workloads that measure 108 s, 66 s and 42 s alone.
        #
        # Three independent gates agreeing to a tenth of a second is not
        # a coincidence to shrug at. I read it first as a pipe deadlock
        # (which was ALSO real, and is fixed) and the numbers did not
        # move -- which is what said the cause was still elsewhere. A
        # duration is measured when the thing ENDS, and that needs
        # somebody watching while the wall gets on with its work.
        threading.Thread(target=_watch, args=(started,), daemon=True).start()
    return started


#: {gate name: seconds}, filled by the watcher as each deferred gate exits.
_DEFERRED_ELAPSED: dict = {}


def _watch(started) -> None:
    """Stamp each deferred gate's duration at the moment it actually ends."""
    pending = list(started)
    while pending:
        for row in list(pending):
            name, _cmd, p, t0, _log = row
            if p.poll() is not None:
                _DEFERRED_ELAPSED[name] = time.perf_counter() - t0
                pending.remove(row)
        if pending:
            time.sleep(0.25)


def _collect_deferred(started):
    """Wait for them, print each one's output in registration order.

    *** A QUEUED GATE THAT NEVER LAUNCHED IS A RED, NOT AN ABSENCE. ***
    The first version of this scheme called `defer()` AFTER the launch
    point, so the queue was empty when it fired: three gates never ran
    and the wall printed GREEN. 29 gates instead of 32, and the only
    visible sign was a count nobody was counting.

    A gate that silently does not run is strictly worse than one that
    fails -- it is the whole defect class this repository keeps finding
    in other people's checks, produced here by a speedup. So the two
    lists are reconciled, and a mismatch is reported as a failed gate
    rather than raised: the summary must still print.
    """
    # *** POLL, DO NOT communicate() IN ORDER. *** The first version
    # blocked on each process in turn, so every gate's "duration" was
    # measured from ITS launch to ITS collection -- and since they were
    # collected together, all three reported an identical 219.4 s. Three
    # gates with the same cost to a tenth of a second is obviously an
    # artifact, and a timing table that misattributes cost defeats the
    # only reason it was added.
    finished = _DEFERRED_ELAPSED

    out = []
    launched = {row[0] for row in started}
    orphaned = [n for n, _cmd, _env in _DEFERRED if n not in launched]
    if orphaned:
        print(f"\n*** {len(orphaned)} DEFERRED GATE(S) WERE QUEUED AND "
              f"NEVER LAUNCHED ***", flush=True)
        for n in orphaned:
            print(f"    {n}", flush=True)
        print("  `defer()` was called after `_start_deferred()`. These "
              "gates did NOT run; the wall is red because it cannot say "
              "they passed.", flush=True)
        out.extend((f"{n} -- QUEUED BUT NEVER RAN", False, None)
                   for n in orphaned)
    for name, cmd, p, t0, log in started:
        p.wait()
        dt = finished.get(name, time.perf_counter() - t0)
        log.flush()
        log.close()
        try:
            text = pathlib.Path(log.name).read_text(encoding="utf-8",
                                                    errors="replace")
        finally:
            pathlib.Path(log.name).unlink(missing_ok=True)
        print(f"\n=== {name} (ran in parallel) ===", flush=True)
        print(text.rstrip() if text else "  (no output)", flush=True)
        if p.returncode != 0:
            # THE DEFERRED PATH CAPTURES ITS OUTPUT, so it can tell a
            # crash from a verdict. `run()` streams instead and passes
            # nothing, which the message then says out loud rather than
            # implying a classification it did not make.
            _explain_exit(name, cmd, p.returncode, output=text)
        process_ok = p.returncode == 0
        output_ok = True
        if name in _STAGED:
            staged, real = _STAGED[name]
            output_ok = staged.is_file()
            output_problem = None
            if output_ok and staged.stat().st_size == 0:
                output_ok = False
                output_problem = "declared artifact is empty"
            if output_ok and real.suffix.lower() == ".json":
                try:
                    parsed = json.loads(staged.read_text(encoding="utf-8"))
                    if not isinstance(parsed, dict):
                        raise ValueError("top level is not an object")
                except Exception as exc:
                    output_ok = False
                    output_problem = f"declared JSON artifact is invalid: {exc}"
            if process_ok and not output_ok:
                print(f"  !! {name}: exited 0 but produced no declared "
                      f"valid artifact at {staged}"
                      + (f" ({output_problem})" if output_problem else "")
                      + "; zero exit is not evidence of production",
                      flush=True)
            if process_ok and output_ok:
                _STAGED_READY.add(name)
            else:
                _STAGED_READY.discard(name)
        out.append((name, process_ok and output_ok, dt))
    return out


#: Filled by the suite gate, read by the census. Empty when --fast, and
#: the census says so rather than reporting zero -- "the suite did not
#: run" and "the suite ran and found nothing" must never look alike.
_SUITE_CENSUS: dict = {}


def _read_junit(path: pathlib.Path) -> dict:
    """Executed / skipped / failed, from the suite's OWN report.

    A MISSING REPORT IS NOT ZERO SKIPS. It is no information, and it is
    recorded as such: `doc_claims_gate` treats an absent count as
    UNCHECKABLE and reds any prose that states one, rather than silently
    agreeing with whatever the README happens to say.
    """
    if not path.exists():
        return {"suite_report": "MISSING -- the suite's size and skip count "
                                "are unknown for this run, which is not the "
                                "same as zero"}
    import xml.etree.ElementTree as ET
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        return {"suite_report": f"UNPARSEABLE ({exc})"}
    suites = ([root] if root.tag == "testsuite"
              else list(root.iter("testsuite")))
    g = lambda k: sum(int(s.get(k, 0)) for s in suites)   # noqa: E731
    return {"tests_ran": g("tests"), "tests_skipped": g("skipped"),
            "tests_failed": g("failures"), "tests_errored": g("errors"),
            "suite_report": "from the suite's junit xml"}


def _roster_reconciliation(rows, *, fast: bool):
    r"""DID THIS WALL RUN THE GATES IT DECLARES? Returns extra result rows.

    *** THE WALL ONCE SHIPPED GREEN HAVING RUN 29 GATES OF 32. *** Three
    were queued after the launch point and never started, and the output
    of a 29-gate run was indistinguishable from a 32-gate one. The orphan
    guard in `_collect_deferred` closed that specific route; this closes
    the class, by reading the roster out of this file's own SOURCE and
    comparing it to what actually appended a result.

    Two directions, and they fail differently:

        DECLARED but did not run    a gate was swallowed. Only excused
                                    when its guard names a condition
                                    that is genuinely false here -- and
                                    the excuse is PRINTED, never silent
        RAN but not declared        the roster parser is blind to a
                                    registration shape, so every count
                                    it feeds (including the one in the
                                    README) is wrong in the direction
                                    that hides gates

    The second is a RED. A census that under-counts is the same defect
    this reconciliation exists to catch, one level up.
    """
    try:
        sys.path.insert(0, str(ROOT / "tools"))
        from gate_roster import roster
        r = roster()
    except Exception as exc:                      # pragma: no cover
        print(f"\n  ROSTER UNREADABLE ({exc}) -- this wall cannot say "
              f"whether it ran the gates it declares")
        return [("gate roster reconciliation (roster unreadable)",
                 False, None)]

    if r["unnameable_sites"] or r["duplicate_names"]:
        print(f"\n  ROSTER INCOMPLETE: unnameable sites "
              f"{r['unnameable_sites']}, duplicates {r['duplicate_names']}")
        return [("gate roster reconciliation (roster incomplete)",
                 False, None)]

    ran = {n for n, _ok, _dt in rows}
    declared = {g["name"] for g in r["gates"]}
    missing = sorted(declared - ran)
    extra = sorted(ran - declared)

    print(f"\n  ROSTER: {len(declared)} gates declared in this file, "
          f"{len(ran)} ran.")
    if missing:
        print(f"  {len(missing)} declared but did NOT run:")
        for name in missing:
            g = next(x for x in r["gates"] if x["name"] == name)
            why = " & ".join(g["guards"]) or "NOTHING -- unconditional"
            print(f"     {name}\n        guard: {why}")
    out = []
    unconditional_missing = [
        n for n in missing
        if not next(x for x in r["gates"] if x["name"] == n)["guards"]]
    if unconditional_missing:
        print(f"  *** {len(unconditional_missing)} UNCONDITIONAL gates did "
              f"not run. Nothing guards them, so something swallowed them.")
        out.append(("gate roster: unconditional gates did not run",
                    False, None))
    if extra:
        print(f"  *** {len(extra)} gates RAN but are not in the roster: "
              f"{extra}\n      The roster parser is blind to how they "
              f"register, so every count derived from it under-reports.")
        out.append(("gate roster: undeclared gates ran", False, None))

    census = {
        "schema": "oneq-wall-census/1",
        "gates_declared": len(declared),
        "gates_ran": len(ran),
        "gates_declared_not_run": missing,
        "gates_ran_not_declared": extra,
        "fast_mode": bool(fast),
        "suite": dict(_SUITE_CENSUS) or {
            "suite_report": "NOT RUN (--fast)"},
        "roster_source_sha256": hashlib.sha256(
            (ROOT / "tools" / "gate_all.py").read_bytes()).hexdigest(),
        "means": ("gates_ran is the number of VERDICT ROWS this wall "
                  "produced, which is what a reader means by 'N gates'. "
                  "It is not gates_declared: branch guards (node, "
                  "iverilog, --fast) legitimately remove some, and each "
                  "removal is listed above with the guard that removed "
                  "it"),
        "does_not_claim": (
            "that the gates PASSED, only that they ran and were counted. "
            "A census is not a verdict"),
    }
    dst = ROOT / "evidence" / "wall_census.json"
    dst.write_text(json.dumps(census, indent=1, sort_keys=True),
                   encoding="utf-8", newline="")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="run even if another gate_all or full pytest is "
                         "already in flight")
    ap.add_argument("--jobs", type=int, default=0,
                    help="xdist workers for the test suite; 0 picks a "
                         "value that leaves the workstation usable")
    ap.add_argument("--fast", action="store_true",
                    help="skip the test suite + coverage (pre-edit loop)")
    a = ap.parse_args()
    busy = _already_running()
    refuse, explanation = _interlock_verdict(busy, a.force,
                                             runs_suite=not a.fast)
    if refuse:
        print(explanation)
        if busy is None:
            return 3
        # ONLY when there IS a neighbour: a headroom refusal on an idle box
        # printing "wait for it, stop it by PID" would send an operator
        # looking for a process that is not there.
        if busy:
            for pid, cmd in busy:
                print(f"  pid {pid}: {cmd}")
            print("Two of these at once took this workstation to 8 GB free "
                  "of 31 and froze the editor -- a symptom that points "
                  "nowhere near its cause. Wait for it, stop it by PID, or "
                  "pass --force if the machine can take it.")
        return 3

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(ROOT / "src")

    import shutil as _sh_early
    results = []
    # *** THE HOOKS ARE ONLY ENFORCED IF THEY ARE INSTALLED. *** Until
    # 2026-08-23 the pre-push gates lived in `.git/hooks`, which is neither
    # cloned nor reviewed: every clone but one had NO coupling gates while
    # the documentation described them as enforced. Moving them to a
    # versioned `hooks/` directory only helps if `core.hooksPath` points
    # there, so the wall now checks the pointer rather than the file.
    results.append(run("hooks installed (core.hooksPath)",
                       [PY, "tools/install_hooks.py", "--check"], env=env))
    results.append(run("derive claims",
                       [PY, "tools/derive_claims.py"], env=env))
    # *** REGISTERED HERE, BEFORE THE LAUNCH. *** The first version
    # called `defer()` further down, inside the `if not a.fast:` block --
    # AFTER this launch point -- so the queue was empty when it fired and
    # three gates never ran while the wall printed GREEN. 29 gates
    # instead of 32, and the only visible sign was a gate count nobody
    # was counting.
    if not a.fast:
        defer("closures re-decided independently (HiGHS)",
              [PY, "tools/verify_d8_closure.py", "--skip-bnb",
               "--milp-time-limit", "1800",
               "--json", "evidence/phase7/oisc_d8_independent.json"],
              env=env,
              # THE ONE DEFERRED GATE THAT WRITES TRACKED EVIDENCE. It
              # is staged, because the suite compares every tracked file
              # under evidence/ against the object database and this ran
              # alongside it.
              publishes="evidence/phase7/oisc_d8_independent.json")
        defer("checker gate re-proves it can FAIL",
              [PY, "tools/checker_equivalence_gate.py",
               "--prove-sensitive", "--cases", "900"], env=env)
        defer("producer gate re-proves it can FAIL",
              [PY, "tools/producer_equivalence_gate.py",
               "--prove-sensitive", "--cases", "150"], env=env)
        defer("latent-parity gate re-proves it can FAIL",
              [PY, "tools/latent_parity_gate.py", "--prove-sensitive"],
              env=env)
        defer("prior-art gate re-proves it can FAIL",
              [PY, "tools/prior_art_gate.py", "--prove-sensitive"], env=env)
        defer("robustness theorem gate re-proves it can FAIL",
              [PY, "tools/robustness_theorem_gate.py", "--prove-sensitive"],
              env=env)
        if _sh_early.which("iverilog"):
            defer("L0 RTL guard re-proves it can FAIL",
                  [PY, "tools/abft_l0_rtl_gate.py", "--prove-sensitive",
                   "--scenarios", "70"], env=env)
    # Launched here so they overlap the whole rest of the wall, and
    # collected before the summary -- a red among them still reds the run.
    _started = _start_deferred()
    if not a.fast:
        cov_json = ROOT / "evidence" / "coverage.json"
        # *** THE SUITE IS THE WALL. *** Measured across five runs it took
        # 698-1076 s of a 13-18 minute wall -- the single dominant term,
        # on a 24-core machine, in ONE process, under coverage tracing.
        # Nothing about this suite needs to be serial.
        #
        # `--dist loadfile` keeps a file's tests on one worker, which
        # matters here rather than being a preference: several test files
        # copy `src/` into a sandbox or build a throwaway git repository,
        # and splitting those across workers multiplies the copies. The
        # worker count deliberately leaves headroom -- this wall running
        # beside another job is what froze the editor, and taking all 24
        # cores would reproduce that on its own.
        workers = a.jobs if a.jobs > 0 else _wait_for_workers()
        if workers == 0:
            # *** REFUSED, RED, AND ON THE RECORD -- NOT SKIPPED. *** The
            # suite is the one stage on this wall that can deadlock the
            # resident being (MEASURED 2026-09-04: 1 worker, ~0.3 GB, 33+
            # min frozen). It is recorded as a red gate so the census,
            # the roster reconciliation and the summary all see that it
            # was declared and did not run; the junit and coverage
            # readers below already tolerate the files being absent.
            # An explicit `--jobs N` still overrides this on purpose.
            print("\n=== test suite + coverage ===\n  REFUSED by worker "
                  "sizing (see above). The seven deferred gates keep "
                  "running; this stage is RED, not skipped.", flush=True)
        par = ["-n", str(workers), "--dist", "loadfile"] if _has_xdist() else []
        if not par:
            print("  (xdist not installed: the suite runs in one process, "
                  "which is the slowest thing this wall does)", flush=True)
        # *** "0 SKIPS" WAS A HEADLINE NOBODY COULD CHECK. *** The README
        # asserted it; nothing recomputed it, because collection cannot
        # know a skip and the suite's summary went straight to the
        # terminal. In a repository whose doctrine is that a skip is a
        # dark gate, that is the one count that must not be typed by
        # hand. The junit report is written to a SCRATCH path -- an
        # artifact under evidence/ would be rewritten by every run and
        # could never be byte-stable, which is how I once built a
        # permanently red gate out of a timing table.
        junit = pathlib.Path(tempfile.gettempdir()) / "oneq_suite_junit.xml"
        junit.unlink(missing_ok=True)
        if workers == 0:
            # A DISTINCT name, deliberately: the roster reads gate names
            # from this file's AST, including literal tuples, and a
            # refusal record carrying the suite's own name registered as
            # a DUPLICATE declaration -- which reds the roster
            # reconciliation itself. Under this name the two branches are
            # two declared gates, each guarded by `workers == 0` and its
            # negation, and the roster reports whichever did not run as
            # guarded-missing rather than as swallowed.
            results.append(("test suite + coverage (REFUSED: no headroom "
                            "for one worker)", False, None))
        else:
            # `run(` stays a DIRECT argument of `results.append(` here:
            # the roster reads the gate list from this file's AST, and a
            # first version wrapped this call in a conditional expression
            # -- the roster dropped from 80 declared gates to 79, and the
            # one stage that can deadlock the being had silently left the
            # reconciliation that would notice it not running.
            results.append(run(
                "test suite + coverage",
                [PY, "-W", "ignore", "-m", "pytest", "tests/", "-q",
                 "--junit-xml", str(junit),
                 # an unregistered marker must be a COLLECTION ERROR here,
                 # not a warning: `-m "not slow"` silently selected nothing
                 # while test_cross_engine wore an unregistered @slow
                 "--strict-markers", *par,
                 "-m", "not slow_exhaustive", "--cov=oneq",
                 f"--cov-report=json:{cov_json}", "--cov-report="], env=env))
        _SUITE_CENSUS.update(_read_junit(junit))
        # A MISSING REPORT IS A RED, NOT A PASS. This began `ok = True`
        # and only ever lowered it inside `if cov_json.exists()`, so no
        # report meant every floor passed. And the suffix scan meant a
        # module ABSENT from the report was never compared to its floor
        # at all -- the floor was declared and silently never applied.
        ok = True
        if not cov_json.exists():
            print("  COVERAGE REPORT MISSING -- floors cannot be "
                  "checked, so this is a failure rather than a pass")
            ok = False
        else:
            cov = json.loads(cov_json.read_text(encoding="utf-8"))
            for fname, floor in COVERAGE_FLOORS.items():
                found = False
                for k, v in cov["files"].items():
                    if k.replace("\\", "/").endswith("oneq/" + fname):
                        found = True
                        pct = v["summary"]["percent_covered"]
                        if pct < floor:
                            print(f"  COVERAGE FLOOR BROKEN: {fname} at "
                                  f"{pct:.0f}% < {floor}%")
                            ok = False
                if not found:
                    # EVERY DECLARED FLOOR MUST BE LOCATED. A module the
                    # report never mentions is not a module at 100%.
                    print(f"  COVERAGE FLOOR UNCHECKED: {fname} does not "
                          f"appear in the report at all -- a floor that "
                          f"cannot be located is not a floor")
                    ok = False
        results.append(("coverage floors", ok))
    results.append(run("doc-claims gate",
                       [PY, "tools/doc_claims_gate.py"], env=env))
    # THE GOAL, CHECKED LIKE ANY OTHER CLAIM. ONE-Q carried two finish
    # lines in two documents with nothing reconciling them, and the
    # smaller one drove the worklist while the larger sat unmeasured.
    # docs/ROADMAP.md now derives every gate's status from artifacts on
    # disk; this refuses if the document and the tree disagree. A gate
    # cannot be closed by writing that it is closed.
    # GA-4a. THE SHIELD IS WHAT MAKES AN UNCERTIFIED POLICY ADMISSIBLE,
    # so it is re-derived every wall rather than trusted from a committed
    # artifact. Each of the eight legs is removed in process and the
    # violating action must BOTH reach A_safe -- proving the negative
    # control isolates that leg -- and be caught by a checker sharing no
    # code with the shield. Runs in about a second; there is no excuse
    # for pinning it.
    results.append(run("shield: 8/8 legs load-bearing (GA-4a)",
                       [PY, "experiments/shield/ga4a_shield_gate.py"],
                       env=env))
    # GA-2. MODEL_STATUS = UNKNOWN is a first-class OUTPUT, and the
    # floor it rests on is one line that an ordinary Bayes update
    # would undo. Re-derived every wall rather than pinned: it runs
    # in seconds, and a floor nobody rechecks is a floor.
    results.append(run("open-world floor fires on held-out truth (GA-2)",
                       [PY, "experiments/open_world/ga2_open_world_gate.py"],
                       env=env))
    # GA-1. A posterior over indistinguishable models is a
    # false-discrimination generator: normalisation names a leader
    # between two models the data cannot tell apart. A false
    # SEPARABLE is the fatal class -- the disjunction merge becomes
    # unconditional -- so the gate demands ZERO of them.
    results.append(run("identifiability certificate honest (GA-1)",
                       [PY, "experiments/identifiability/ga1_identifiability_gate.py"],
                       env=env))
    # GA-5. RUL-grow(0.01) had the LOWEST logical error rate in the
    # entire study -- 22.6x better than baseline -- and completed
    # 7.2% of jobs. Optimising protection without a completion
    # constraint has a degenerate optimum, and the degenerate
    # optimum is a machine that never answers.
    results.append(run("freeze detector: a win nobody finishes (GA-5)",
                       [PY, "experiments/freeze/ga5_freeze_gate.py"],
                       env=env))
    # GA-12. THE TERMINAL CLAIM, and the one gate the architecture has
    # no graceful successor for. Its own spec -- "the certified bound
    # never exceeds the measured P_VCC" -- is passed perfectly by a
    # bound of 0, so the gate carries three more legs: the bound must
    # SAY something, the tighter posterior-weighted control must be
    # CAUGHT inverting, and shifts must actually be detected.
    results.append(run("composition: bound never above measured (GA-12)",
                       [PY, "experiments/composition/ga12_composition_gate.py"],
                       env=env))
    # GA-13. The FIRST of the three finish-line conditions. Its mechanism
    # is one line of calculus, so the gate is built around what the
    # calculus leaves out: the suppression is SWEPT (a single Lambda is a
    # single riggable number), the check's false-accept rate is CHARGED
    # (more shots buy more exposure -- 0.0101*eps at q=0.99 against
    # 0.695*eps at q=0.59), and the corpus's own independence assumption
    # is simulated rather than assumed.
    results.append(run("checkability operating point (GA-13)",
                       [PY, "experiments/checkability/"
                            "ga13_operating_point_gate.py"], env=env))
    # GA-4b. THE AUTHORITY RULE: may the kernel actuate at all? This gate
    # passes by MEASURING Delta well, not by finding it small -- Sec 3.3.2
    # says "a measured Delta > 0.05 is a PASS of the gate and a DEMOTION
    # of the mechanism", and every default in the module leans against
    # granting authority.
    results.append(run("certified policy gap / authority (GA-4b)",
                       [PY, "experiments/policy_gap/"
                            "ga4b_policy_gap_gate.py"], env=env))
    # GA-14. The competitor becomes a component: RL steers parameters,
    # ONE-Q steers protection, the shield decides which RL moves may be
    # applied at all. The stated bar -- additive to within 30% -- is
    # satisfied by two INERT arms and cannot separate additive from
    # independent-multiplicative composition below g~0.6, so the gate
    # reports against both references and says which its band can tell
    # apart.
    results.append(run("RL wrap additivity + shield rejection (GA-14)",
                       [PY, "experiments/rl_wrap/ga14_rl_wrap_gate.py"],
                       env=env))
    # GT-XR. The exchange-rate law, derived rather than transcribed. Its
    # external validation is BLOCKED on arXiv:2604.15603's Pareto front,
    # so this runs the four legs that do not depend on their numbers --
    # including the finding that the corpus's closed form is the large-d
    # ASYMPTOTIC, wrong by +15% at d=9, near where GA-13 puts a checkable
    # workload.
    results.append(run("exchange-rate law, local legs (GT-XR)",
                       [PY, "experiments/exchange_rate/"
                            "gtxr_exchange_rate_gate.py"], env=env))
    # GA-0. "No outcome kills the architecture" is an ASSERTION until
    # something computes the break point. This gate does: latency binds
    # only at gamma = 0.374, which is 8.6x below the fitted exponent and
    # 5.4x below the band's own floor. gamma ITSELF is blocked on the
    # estate's round-locality harness and the gate says so.
    results.append(run("dwell exponent robustness (GA-0)",
                       [PY, "experiments/dwell/ga0_dwell_gate.py"], env=env))
    # E0/E1 -- the corpus says "Run them before anything else" and they
    # had not been run. Static DAG analysis, no simulator: SECONDS over
    # three circuit families, where it was milliseconds over two --
    # modular exponentiation is O(n^3) in both gates and boundaries.
    # The exact figure is the artifact's own `seconds` field rather than
    # a number copied into this comment: I wrote "MEASURED 2.7 s" here
    # and made it 2.47 s an hour later by deleting a redundant pass, so
    # the hand-copied version was stale before the commit.
    # The six-way outcome table fires on every circuit family, and every
    # row of it is exercised by the test suite rather than only the rows
    # the live circuits happen to reach.
    results.append(run("transparency census (E0/E1)",
                       [PY, "experiments/transparency/e1_census_gate.py"],
                       env=env))
    # E2 -- the six-policy bake-off. Real Stim shots, a calibrated
    # complementary-gap soft decoder, and every distance sampled until
    # its own relative standard error clears 10%: a fixed shot count
    # that resolves d=5 measures nothing at d=11. Priced by the corpus
    # at 500-1,500 core-h and done in minutes, which is stated in the
    # artifact rather than left for a reader to notice.
    results.append(run("six-policy bake-off (E2)",
                       [PY, "experiments/bakeoff/e2_bakeoff_gate.py"],
                       env=env))
    # THE DUAL READ AS AN INSTRUMENT. Every certified shot already
    # carried a proven lower bound on the error that occurred, and
    # `million_shot_gate_d5` stored 960,181 of them and kept NONE.
    # Runs at p=1e-2 rather than the corpus 3e-3 on purpose: at the
    # quieter point no gap bin expects enough errors to verify its own
    # predicted rate, and the gate correctly reports UNCHECKABLE
    # instead of a verdict. A gate that can only run where it cannot
    # read is not a gate.
    results.append(run("certified noise floor / dual as instrument",
                       [PY, "experiments/dual_observable/"
                            "certified_noise_floor.py",
                        "--noise", "0.01", "--budget", "300"],
                       env=env))
    # *** THE PUBLISHED HARDWARE NUMBERS, RE-DERIVED FROM THEIR OWN
    # BYTES. *** The same 4,000 `ibm_marrakesh` shots reported a
    # certified weight of 4.9615 and then 5.1264 -- a 3.32% shift --
    # because `snapshot_calibration` reads the backend LIVE and the
    # device's published calibration had moved between two analyses of
    # one completed job. Neither number was wrong, and neither could be
    # checked by anyone.
    #
    # The weights are now PINNED into the artifact as exact rationals,
    # and this re-scores the stored detectors against them with no
    # network, no device and no calibration snapshot, then compares 18
    # published claims to what it recomputed. It fires on a single
    # weight perturbed by one part in 1e9 (MEASURED: five claims move).
    #
    # So it is a regression test on every producer in the lane at once:
    # change the decoder, the lift, the estimator's debias or the
    # calibration binning, and the artifact stops reproducing here.
    results.append(run("hardware noise floor reproduces from pinned bytes",
                       [PY, "experiments/dual_observable/"
                            "hardware_noise_floor.py", "--from-artifact"],
                       env=env))
    # NOT A GATE: `tools/audit_claim_producers.py` asks a real question
    # -- is a cited number produced by anything that re-runs? -- and its
    # classifier is not good enough to gate on. It was wired here, ran
    # once, and had to be corrected three times in ten minutes: it read
    # its own exemption list as evidence; it counted a DOCSTRING mention
    # of `million_shot_gate_d5.json` as production; and tightening it to
    # require a nearby write then missed `roadmap`/`ratchet`, which are
    # written through variable paths. Its answer moved 20 -> 52 -> 29
    # under those corrections.
    #
    # A frozen list of that is churn, and a gate that cries wolf gets
    # switched off, taking whatever real signal it had with it. The tool
    # stays as a diagnostic a human reads; it does not vote.
    # *** TWO LANES THAT RAN ONLY WHEN A HUMAN TYPED THEM. *** Both
    # were written, both produced real findings, and neither was on the
    # wall -- so nothing would have noticed if they broke, went stale,
    # or stopped running. A check nobody runs is a check that does not
    # exist, which is the failure this repository spends most of its
    # time hunting in other people's code.
    #
    # Both are pure re-analysis of the PINNED hardware artifact: zero
    # QPU, zero network, ~50 s each.
    #
    # The fragility lane's headline is a REFUSAL, and that is exactly
    # why it must run: it re-measures that relabelling detectors
    # `i -> N-1-i` leaves the certified total at 2952.828 while moving
    # 4.61% of it between rounds. If a future change made the
    # attribution look stable, that would be the thing worth catching.
    results.append(run("certified fragility map (envelope over labellings)",
                       [PY, "experiments/dual_observable/"
                            "fragility_lane.py"],
                       env=env))
    # The device-law lane induces falsifiable claims about the hardware
    # and kills them on one certified counterexample. It also re-proves
    # the shortcut that makes it affordable: 793 decodes instead of
    # 39,168, because the certified correction is itself a witness that
    # an unused edge was avoidable.
    results.append(run("device laws induced from certified shots",
                       [PY, "experiments/dual_observable/"
                            "device_laws_lane.py"],
                       env=env))
    # *** THE EXACT DUAL ENVELOPE, AS A REGRESSION. *** The full run over
    # 576 marrakesh shots takes ~566 s and is the theorem: 0 shots with a
    # unique dual, exact ranges 20.20x wider than two labellings, no
    # detector's floor clearing any rival's ceiling -- a PROVEN
    # impossibility for dual-based localization on that data. The wall
    # runs a 100-shot prefix (~85 s, artifact marked incomplete by the
    # lane itself) for the two properties that must never regress: the
    # exact range must CONTAIN the two-labelling band on every detector
    # (a violation is rc 6), and the widening factor must stay >= 1.
    # A change that made the attribution look determined would read as
    # an improvement everywhere else; here it reads as a red.
    results.append(run("exact dual envelope contains the two-labelling band",
                       [PY, "experiments/dual_observable/"
                            "exact_envelope_lane.py", "--shot-cap", "100",
                        "--json", "evidence/exact_envelope_probe.json"],
                       env=env))
    # *** THE ANNIHILATOR OF THE FACE, AS A REGRESSION. *** The full run
    # over 576 marrakesh shots (~18 min) is the theorem's other half: the
    # face is a segment on 438 shots and on every one of them only the
    # fired pair floats, with its sum fixed; loc_free grows with |fired|
    # to 27 of 28 at twelve. The wall runs a 100-shot prefix for the one
    # property that must never regress: on every shot the algebraic
    # verdict and the ranged extremes must AGREE on every detector (a
    # disagreement is rc 6). A change that made the dual look more
    # decided would read as an improvement elsewhere; here it is a red.
    results.append(run("face annihilator agrees with the envelope on every detector",
                       [PY, "experiments/dual_observable/"
                            "face_annihilator_lane.py", "--shot-cap", "100",
                        "--json", "evidence/face_annihilator_probe.json"],
                       env=env))
    # *** THE SCALE'S MECHANISM, AS A REGRESSION. *** R14-R18 measured that
    # scaling the DEM's time-like weights cuts logical errors 2-7x on
    # hardware; R19/R20 established WHY, and the two claims that must never
    # silently flip are both here. The synthetic CONTROL: on shots sampled
    # from the very model that decodes them the same correction must HURT
    # (it does, 872 -> 2269 errors) -- if it ever stops hurting there, the
    # gain is our decoder's and not the device's, and every downstream
    # claim changes. The DEGENERACY kill: scaling by logical-cut membership
    # must still destroy the decoder. rc 6 on either.
    results.append(run("the scale corrects the DEVICE, not the decoder (control + kill)",
                       [PY, "experiments/dual_observable/probes/"
                            "scale_mechanism.py",
                        "--json", "evidence/scale_mechanism.json"],
                       env=env))
    # *** THE MODEL READ OFF THE DETECTORS (R23). *** Every correction
    # before this one had a constant in it that somebody chose. This one
    # has none: the two-point estimator reads each edge probability from
    # the detector statistics, using no calibration data and never seeing a
    # logical outcome, and it then beats the device's own DEM 8.47x on
    # HELD-OUT shots (381 -> 45 over four runs and two devices) -- better
    # held out than in sample, which is what an estimated model does and a
    # fitted one does not. The lane refuses (rc 6) the moment that stops
    # being true. 5 seconds for all four runs, so it runs at full strength.
    results.append(run("the model measured from detectors beats the device's own, held out",
                       [PY, "experiments/dual_observable/measured_model_lane.py",
                        "--json", "evidence/measured_model.json"],
                       env=env))
    # *** SITE-TO-SITE NOISE CORRELATION, AND A SELECTOR THAT DECLINES. ***
    # A decoherence-free memory pays only above a crossover correlation --
    # rho = 0.500 with ideal extra operations, 0.700 with four at 0.1%, and
    # NEVER at 0.5%. Measured on ibm_fez d=9, the only committed run with
    # non-adjacent sites, the excess over the structural control is +0.017 at
    # most: ~40x short. The lane exists to keep that comparison honest, and it
    # reds (rc 6) if the structural control ever stops showing the
    # code-induced correlation it is there to subtract -- because then the
    # excess is measuring the circuit and not the device.
    results.append(run("noise correlation is measured against its structural control",
                       [PY, "experiments/dual_observable/noise_correlation_lane.py",
                        "--json", "evidence/noise_correlation.json"],
                       env=env))
    # *** THE IBM FINDING'S SCOPE, ON PUBLIC DATA (R29). *** Google's shipped
    # SI1000 prior against Google's Willow shots by error family, with the
    # pair prediction dark-checked on DEM-sampled shots (hyperedges
    # included). The gate holds two things: the dark check reads 1, and a
    # second vendor's model-derived prior does NOT overstate the space-like
    # family -- so the 7-38x on IBM is IBM's pipeline. 36 s, twelve
    # configurations, zero QPU.
    results.append(run("the per-family overstatement is IBM's, not model-derived priors' (Willow)",
                       [PY, "experiments/dual_observable/willow_family_ratio.py",
                        "--json", "evidence/willow_family_ratio.json"],
                       env=env))
    # *** THE CAMPAIGN AS ONE TABLE (OPEN_WORK Sec 0++ item 5). *** R14-R18's
    # fitted scales, R21's burst price and R23's estimate had never been run
    # against each other on the same held-out shots. Now they are, on every
    # committed run, with exact intervals and paired discordant counts, and
    # the gate holds the ordering: the measured model has the fewest pooled
    # errors or the decoder story has moved. ~15 s.
    results.append(run("the campaign as one table: scales vs burst vs measured, held out",
                       [PY, "experiments/dual_observable/campaign_table_lane.py",
                        "--json", "evidence/campaign_table.json"],
                       env=env))
    # *** THE A/E/B/Y LEDGER ON DEVICE SHOTS (campaign 2.1). *** The four-term
    # budget, its calibrated Z_M null and the enclosure join were built,
    # mutation-tested and never pointed at a QPU. Now every committed run's
    # held-out shots go through A (pymatching on the DEM), E (exact
    # T-join), B/q (coset enclosure) and Y (the device's readout), with the
    # DEM-sampled control beside each. 300 shots per arm (1,000 ran past ten
    # minutes on the d=5 arm's enclosure), the same 300 the committed
    # artifact was produced at, so the wall's regeneration is the artifact
    # and not a smaller cousin of it.
    results.append(run("the error budget (A/E/B/Y) on device shots, with its sampled control",
                       [PY, "experiments/dual_observable/error_budget_device_lane.py",
                        "--shots", "300", "--json", "evidence/error_budget_device.json"],
                       env=env))
    # *** EVERY LANE IS ON THE WALL OR IN THE REGISTRY (campaign 3.4/3.5). ***
    # 274 experiment scripts and evidence-writing tools; 65 on the wall; the
    # other 209 ran once by hand and were named by nothing. `docs/
    # LANE_REGISTRY.md` names each with a status and a date, and the gate
    # holds it in both directions with its sensitivity proven.
    results.append(run("lane registry: every script is on the wall or listed with a reason",
                       [PY, "tools/lane_registry_gate.py"], env=env))
    # THE CEILING BRACKET THAT WAS OWED (campaign 3.6). g4a_replay waived its
    # oracle ceiling and named this script as the debtor; the script existed,
    # ran in 1.6 s, and was on no wall. Conditional-DEM matching against
    # EXACT ML on the four configurations where exact ML is affordable:
    # matching captures 0.78-0.85 of the optimum. The waiver in g4a_replay
    # now cites it and stays a waiver, because the replay's configuration
    # is not one of the four.
    results.append(run("ceiling bracket: conditional matching vs exact ML, matched configurations",
                       [PY, "experiments/g4_hardware/ceiling_bracket.py"], env=env))
    # *** FTQC_CONCEPTS RANK 8, THE ONE-DAY ENUMERATION, KILLED IN 8.5 s
    # (campaign 2.4, R31). *** Every signed pattern of weight <= 2 over the
    # 1,344-bit register, sorted by (residues, parity): 33,968 colliding
    # pairs survive the parity ancilla, 2(n-2) of them the integer identity
    # 2^i + 2^(i+1) = 2^(i+2) - 2^i that no modulus set can separate. The
    # gate pins the count and the two predictions the document made.
    results.append(run("RNS outer code: weight <= 4 aliasing enumerated exactly (Rank 8)",
                       [PY, "experiments/rns_aliasing/enumerate_aliasing.py",
                        "--json", "evidence/rns_aliasing.json"], env=env))
    # FTQC_CONCEPTS RANK 5 (2026-09-08, campaign 2.5): the deferred
    # logical Pauli's conjugated support against deferral depth, by
    # exact CHP propagation (oneq.pauli_defer, checked against stim's
    # tableau). The document's kill rule -- median support at depth 100
    # above 20 patches -- is read on Gidney's 20-qubit adder, a chain
    # of them, and modular exponentiation; the blocked fraction is
    # reported beside the median because a median over the deferrable
    # boundaries alone would flatter a circuit that blocks early. ~1 s.
    results.append(run("deferred Pauli support vs depth (Rank 5)",
                       [PY, "experiments/transparency/rank5_support_curve.py",
                        "--json", "evidence/rank5_support_curve.json"], env=env))
    # B2's EXACT-CLASS SAT ENCODING, CONTROLLED (2026-09-08, campaign
    # 2.7). The gross-code questions run for hours by hand
    # (`--budget 1800 --json evidence/b2_class_feasibility.json`); the
    # wall runs the encoding's control on [[72,12,6]] in under a second:
    # the class of a known weight-6 logical is SAT at 6 with a
    # re-verified witness, UNSAT at 5, and a different class is UNSAT
    # at 5 -- so a wrong dual basis or a wrong XOR chain turns this red
    # long before it turns a gross-code NO into a wrong answer.
    results.append(run("B2 class-feasibility encoding control ([[72,12,6]])",
                       [PY, "experiments/b2_class_feasibility.py", "--control",
                        "--budget", "120"], env=env))
    # FTQC_ESTATE_MAP #7 (2026-09-09, campaign 2.9): the BP message
    # Jacobian's spectra against non-convergence, on the estate's own
    # sum-product decoder. The wall runs the REDUCED profile -- the full
    # sweep is six configurations and half an hour, and its artifact is
    # committed. What the wall protects is that the instrument still
    # runs end to end and still reports its confound control: every
    # spectral score is computed FROM the syndrome, so the number of
    # fired detectors is the baseline each one has to beat.
    if not a.fast:
        results.append(run("BP transient growth: sigma_max vs rho (estate map #7)",
                           [PY, "experiments/bp_transient/kreiss_vs_rho.py",
                            "--quick", "--kreiss-shots", "4"], env=env))
    # THE SATURATION WALL (2026-09-09, campaign 5.1). The blueprint's own
    # headline, computed a priori with zero Monte Carlo: strip a plan's
    # noise from the CIRCUIT, re-derive the model, and ask stim whether
    # any graphlike logical error survives. Six families in under two
    # seconds, and the control (remove every noise source -> CUT) is what
    # makes the rest mean anything.
    results.append(run("saturation wall: does a plan CUT or hit a wall (qsep)",
                       [PY, "experiments/qsep/saturation_wall.py",
                        "--json", "evidence/qsep_wall.json"], env=env))
    # THE CIRCUIT-DISTANCE AUDIT (2026-09-09, campaign 2.3). Permute the
    # four CX layers of a syndrome-extraction round and measure the
    # circuit distance of every order, exactly. 20 of 24 are not
    # schedules at all -- the detectors stop being deterministic -- and of
    # the four that are, HALF give ceil((d+1)/2): a bad order costs half
    # the error-suppression exponent, and the penalty grows with d.
    # Three seconds for five families.
    results.append(run("circuit-distance audit of CX schedules",
                       [PY, "experiments/circuit_distance/schedule_audit.py",
                        "--json", "evidence/schedule_audit.json"], env=env))
    # THE ENCLOSURE'S CONFIGURATION, ATTACKED (2026-09-09, campaign 5.2).
    # `enclosure_forgery` attacks the DOCUMENT and the checker refuses
    # every entry; this attacks the MECHANISM TABLE the producer reads,
    # which the document then binds faithfully to the wrong thing. The
    # certificate accepts every corrupted document -- that is the finding,
    # not a failure -- and the configuration check catches every
    # corruption that influenced the answer. Under a second.
    results.append(run("enclosure configuration under single-bit corruption",
                       [PY, "experiments/enclosure_config/configuration_attack.py",
                        "--syndromes", "5", "--per-field", "6",
                        "--json", "evidence/enclosure_config_attack.json"], env=env))
    # THE TWO-SIDED NOISE FLOOR (2026-09-09, campaign 5.6 / ESTATE_MAP #6).
    # Measure eps by inverting 2 eps (1 - eps) on two replicas of one
    # measurement, then judge a model against `1 - eps` FROM BOTH SIDES.
    # The arm that matters is the model that returns the replica it is
    # scored against: it beats the truth at every setting -- which is what
    # a one-sided "higher is better" gate rewards -- and this gate rejects
    # it. Four error rates over two orders of magnitude, three seeds each,
    # bands Bonferroni-corrected to the family, in a quarter of a second.
    # The FULL profile runs here; there is no reason to smoke a lane this
    # cheap.
    results.append(run("two-sided noise floor: a model must agree AT 1 - eps",
                       [PY, "experiments/noise_floor/two_sided_gate.py",
                        "--json", "evidence/noise_floor_gate.json"], env=env))
    # ESTATE_MAP #6's PRECONDITION (2026-09-09, campaign 5.6). The map says
    # to check first whether there is a residual to resample at all, and
    # that a bad answer kills the surrogate-null design. Measured against
    # the only floor that means anything -- two draws of the SAME model --
    # the circuit-vs-its-own-DEM residual IS the floor, and its maximising
    # element does not repeat across seeds. The positive control reads
    # ~50 sigma at exactly the pair it was planted on and grows as sqrt(N)
    # while the floor falls. 48 s.
    results.append(run("DEM residual: is there anything to resample?",
                       [PY, "experiments/dem_admissions/surrogate_null_fdr.py",
                        "--json", "evidence/dem_residual_precondition.json"],
                       env=env))
    # *** WHAT A STRANGER RUNS FIRST, AND WHAT NOTHING RAN (2026-09-09). ***
    # `verify_release.py` is the tool REPRODUCE.md tells an outsider to run
    # before believing anything: it checks the archive signature, the
    # issuer pinning, the in-toto subjects, every per-shot record against
    # the SIGNED archive, and both external checker pins. It was on no
    # gate and no test drove it against the real tree -- so it sat RED for
    # an unknown period on a CRLF/LF equivocation, and nobody could have
    # known. A verifier a gate does not run is a verifier that drifts.
    # 2 seconds.
    results.append(run("verify_release: what a stranger runs first",
                       [PY, "tools/verify_release.py"], env=env))
    # INJECTION-RECOVERY COMPLETENESS (2026-09-09, campaign 5.6 /
    # ESTATE_MAP #6 part 3). The map: "FDR alone bounds only false
    # positives -- AND A BROKEN SEARCHER HAS FDR 0 TRIVIALLY." So the
    # threshold comes from the null at a 5% budget and the completeness is
    # measured against it: 50% recovery at p = 0.00215 in ABSOLUTE units,
    # held-out false positives 2/40 = 5.0%. The control that matters is
    # that the detector DOES fire on a null -- one that never fires scores
    # a perfect FDR and finds nothing. 28 s.
    results.append(run("injection-recovery: the 50% completeness point",
                       [PY, "experiments/dem_admissions/injection_recovery.py",
                        "--json", "evidence/injection_recovery.json"],
                       env=env))
    # TRAP FAMILIES (2026-09-09, campaign 5.6 / ESTATE_MAP #6 part 1). Five
    # UNLABELLED arms, one of which -- a real mechanism planted BELOW the
    # shot-noise floor -- the map calls "the arm most estimators will fail",
    # because the honest answer there is REFUSE and "nothing found" is the
    # clean arm's answer to a different question. So a verdict is one of
    # three and NONE carries a burden: the detectable floor must reach below
    # the amplitude the caller asked about. The clean arm appears TWICE at
    # two declared amplitudes, which is what makes NONE reachable and the
    # declared amplitude load-bearing rather than decoration.
    #
    # Three things make it more than a self-agreement machine. Every FOUND
    # is re-checked with a DIFFERENT functional of the same 2x2 table -- the
    # log odds ratio, against its own measured null -- because re-running
    # the detector's covariance is the same check twice. The floor comes
    # from the WEAKEST of three probe elements, and T9 regrades every arm
    # under the strongest to show no verdict turns on that choice. And the
    # floor answers to a lane outside this one: 2.00e-3 here against the
    # injection-recovery 50%-completeness point of 2.15e-3, scaled by
    # sqrt(N) so it holds at any shot count -- without that anchor,
    # floor-relative amplitudes would grade correctly under ANY
    # calibration, however wrong. 9 s.
    results.append(run("trap families: the arm whose answer is REFUSE",
                       [PY, "experiments/dem_admissions/trap_family_benchmark.py",
                        "--json", "evidence/trap_families.json"], env=env))
    # E7 -- THE PRICING THEOREM COSET_ENCLOSURE §9 ASKED FOR (2026-09-09,
    # campaign 5.3). §9: "the pricing still ignores collateral damage. E6
    # localises each owed detector to the mechanisms that can touch it, but
    # allows ANY of them to serve, ignoring what that would do to the other
    # detectors [...] that is the next theorem, and it is the one that would
    # carry the tier past d=5."
    #
    # E7's answer is that the independent SET is the wrong object and the
    # connected COMPONENT is. A mechanism touching owed detectors in two
    # components would merge them, so the components' mechanism sets are
    # pairwise DISJOINT, the factorisation stays EXACT, and every owed
    # detector's parity is enforced instead of only a greedy independent
    # subset's. Measured: E7 EQUALS the exact relaxed sum in 200/200 oracle
    # models -- it is optimal within its constraint family, not merely a
    # tighter bound -- and the gap over E6 grows x2.99 per owed detector E6
    # abandons, reaching 7.3e5 median (2.2e7 max) at 12 owed detectors on the
    # real d=5 r=5 p=1e-3 model. That is the same order of effect §9 credits
    # for the tier deciding at all, again, on top of it.
    #
    # NOT WIRED INTO THE TIER, and the lane says so: `_price` still calls E6,
    # the rule lives in the producer AND the independent checkers which must
    # agree byte-for-byte, and 0.6 s per price at 12 owed against a 5.6 s/shot
    # tier makes the naive per-discard call unaffordable without caching. The
    # theorem is proved and measured; the integration is a separate row. 11 s.
    results.append(run("E7: price a discard by the residual syndrome",
                       [PY, "experiments/enclosure/e7_pricing.py",
                        "--json", "evidence/e7_pricing.json"], env=env))
    # AND WHAT THAT INTEGRATION WOULD COST, MEASURED (R51). The comment above
    # prices the blocker at "0.6 s per price at 12 owed" -- true, and about a
    # case the real d=5 tier reaches in 27 of 533,760 calls (0.005%). E7's
    # cost is not 2^|owed|: the owed set splits into conflict-graph
    # components and the DP runs per component, so the driver is
    # sum(2^|component|), median 16 states. This lane is HERE, in the ladder,
    # because `evidence/e7_cost_profile.json` is asserted by
    # tests/test_e7_cost_profile.py -- and an artifact whose producer is not
    # on the wall rots silently while its test keeps passing against the
    # stale copy. 14 s.
    results.append(run("E7 integration cost profile (real d=5 tier)",
                       [PY, "experiments/enclosure/e7_cost_profile.py",
                        "--json", "evidence/e7_cost_profile.json"], env=env))
    # ANCHORED PRE-REGISTRATION (2026-09-09, campaign 5.6 / ESTATE_MAP #6
    # part 5). A signed JSON file is easy; what this drives is that each of
    # ELEVEN ways a pre-registration stops constraining anything produces a
    # REFUSAL, against a real sealed document -- write-once, the criterion
    # under the signature (tampering one clause invalidates Ed25519), both
    # witness directions, three clock refusals, output binding, an unpinned
    # issuer, and the honest round trip where the criterion is sealed BEFORE
    # the data exists.
    #
    # The two that are not obvious. *** A CRITERION THAT CANNOT FAIL IS
    # REFUSED AT REGISTRATION *** -- nobody edits a criterion, what happens
    # is that it was never sharp enough to exclude what occurred, so every
    # prereg carries a falsifying witness and registration refuses if that
    # witness passes. And *** A RESULT THAT PREDATES ITS OWN REGISTRATION IS
    # REFUSED *** -- the seal verifies, the output matches, the expiry has
    # not passed, and the author had already seen the number.
    #
    # Precedence is reported UNPROVEN by construction: a signature proves
    # WHO, never WHEN, and the third-party Bitcoin anchor needs network. The
    # digest to submit covers the SIGNED CORE only, so attaching the anchor
    # cannot change the thing that was anchored. 3 s.
    results.append(run("pre-registration: eleven ways it stops constraining",
                       [PY, "experiments/dem_admissions/anchored_prereg.py",
                        "--json", "evidence/anchored_prereg.json"], env=env))
    # THE BLUEPRINT'S CRITICAL PATH, AGAINST THE TREE (2026-09-09, campaign
    # 0.5.7). Section 89 gives an eleven-node dependency chain; nothing
    # connected it to what has actually been built, so a node could be
    # worked on while the node it depends on was absent and nothing would
    # say so. The chain and the parallel-work EXEMPTION are both parsed
    # from the blueprint -- letting the map declare its own exemptions
    # would make them unfalsifiable -- and the map is in docs/ where it
    # can be argued with. Instant.
    results.append(run("blueprint critical path vs the tree",
                       [PY, "tools/critical_path_gate.py",
                        "--json", "evidence/critical_path.json"], env=env))
    # THE WINDOW DIFFERENTIAL (2026-09-09, campaign 5.5 / ESTATE_MAP #1).
    # Sliding-window decoding is validated in the literature by "no
    # noticeable increase in logical error rate" -- an error-COUNT proxy.
    # This decodes the same shots globally and windowed and compares the
    # two per shot. At the field's own n_buf = n_com = d the per-shot
    # disagreement is 6.8-8.9x the LER difference, floor 3.8-5.3x after
    # the 95% band. The mechanism is subtraction: dLER differences the two
    # discordant counts while the disagreement sums them. Two controls
    # carry it -- committing the whole history reproduces the reference
    # bit-for-bit, and the decoder's own tie-break degeneracy sits 10-28x
    # below the effect. 77 s.
    results.append(run("window differential: per-shot vs the field's LER proxy",
                       [PY, "experiments/window_differential/per_shot_differential.py",
                        "--json", "evidence/window_differential.json"], env=env))
    # GT-CT ON THE WALL AT A REDUCED PROFILE. The full campaign is
    # 153 s; the wall runs a smaller one so every node stays affordable,
    # and the SHOT COUNT is the only thing reduced -- all six controls
    # run at full strength, because a control that only runs in the big
    # campaign is a control nobody runs.
    results.append(run("code trajectory vs hostile baseline (GT-CT)",
                       [PY, "experiments/code_trajectory/gtct_gate.py",
                        "--shots", "600", "--cal-shots", "120000",
                        "--json", "evidence/gtct_wall_probe.json"],
                       env=env))
    # *** A RECORDED FAILURE MUST STAY RE-DERIVABLE. *** GA-3 ran and
    # FAILED, and that failure is the evidence. It was PINNED -- held by
    # bytes, never recomputed -- because the row is priced at 30 CPU-h.
    # It runs in 45 seconds. A world edit that turned 141 failures into
    # 200 would have left the pinned artifact and every sentence about
    # the finding intact; now it goes red here.
    results.append(run("GA-3 reproduces its recorded failure",
                       [PY, "experiments/voi/ga3_voi_gate.py",
                        "--check-reproduces"], env=env))
    # GA-11's FIRST CLAUSE IS A THEOREM AND A MODEL, NOT A DEVICE. The
    # measured half needs IBM meas_level=1 and is blocked; the floor half
    # reduces to one dimensionless group and is decided here in
    # milliseconds. It also carries the units correction: Ozawa's eps is
    # an RMS error, the quoted readout number is a probability, and
    # comparing them directly reads "unphysical everywhere".
    results.append(run("WAY readout floor (GA-11)",
                       [PY, "experiments/way/ga11_way_floor_gate.py"],
                       env=env))
    results.append(run("roadmap matches the tree",
                       [PY, "tools/roadmap_gate.py", "--check",
                        "--json", "evidence/roadmap.json"], env=env))
    results.append(run("claims provenance + bundle completeness",
                       [PY, "tools/claims_provenance_gate.py"], env=env))
    # THE RECEIPT CORPUS, COMPLETELY (external audit 2026-09-14, R-01/R-03).
    # Every receipt behind paper 2, against the pinned manifest and the
    # persisted instances: missing, duplicated, unlisted or altered fails.
    # The second gate rebuilds each persisted instance from its construction
    # and requires byte-for-byte agreement -- reproduction kept separate from
    # replay, so the replay never checks proofs against a rebuilt input.
    results.append(run("receipt corpus: complete pinned replay",
                       [PY, "tools/replay_receipts.py"], env=env))
    results.append(run("receipt instances rebuild exactly",
                       [PY, "tools/export_receipt_instances.py", "--verify"],
                       env=env))
    results.append(run("code standards + checker independence",
                       [PY, "tools/standards_gate.py"], env=env))
    # The audit-debt register must describe THIS tree. It had drifted --
    # three modules listed as queued-for-deletion were already deleted --
    # and nothing checked, though the file opens by stating the rule.
    # THE PREREGISTRATION SEALS. Added 2026-08-19 after finding that the
    # Phase 6 pin had been broken on 2026-08-17 by a documentation-hygiene
    # commit and NOTHING WENT RED for two days: the preflight computed the
    # comparison but only used it to pick a beacon branch. A seal with no
    # gate is a note. Passes on the original pin or on a written,
    # reviewed acknowledgement; reds on anything else, and on an
    # acknowledgement that no longer matches any reachable state.
    results.append(run("preregistration seals (original pin or reviewed break)",
                       [PY, "tools/preregistration_gate.py", "--json",
                        "evidence/preregistration_seal.json"], env=env))
    results.append(run("audit-debt register matches the tree",
                       [PY, "tools/audit_debt_gate.py"], env=env))
    # H-4: the overhead program's headline 2.31x, re-derived from
    # published inputs by an implementation independent of the one that
    # produced it -- the exact-agreement discipline H-1 demands, applied
    # to the number every downstream estimate inherits. Fast and
    # deterministic, so it runs every time.
    results.append(run("H-4 heterogeneous ratio (independent re-derivation)",
                       [PY, "experiments/overhead/h4_hetero_ratio_verify.py"],
                       env=env))
    # B-1: the bias ceiling, from offline calibration snapshots. Also
    # fast and deterministic, and it is the other overhead number that
    # existed only in the spec.
    results.append(run("B-1 bias ceiling (offline, ABSTAIN 6/6)",
                       [PY, "experiments/overhead/b1_bias_ceiling.py"],
                       env=env))
    # H-1: the spec's named highest-risk gate -- exhaustive single-Pauli
    # enumeration adjudicated by TWO independent references, exact
    # agreement rather than a tolerance. "the whole pass is built on
    # sand" is its stated failure meaning, so it runs every time.
    results.append(run("H-1 sensitivity exact (0 discrepancies, 2 refs)",
                       [PY, "experiments/overhead/h1_sensitivity_exact.py"],
                       env=env))
    results.append(run("H-5 Gidney layout cross-check (exact)",
                       [PY, "experiments/overhead/h5_gidney_crosscheck.py"],
                       env=env))
    # L2' in RTL, verified against the Python detector across a language
    # boundary. Skipped with a NOTICE if iverilog is absent -- a silent
    # skip would be the dark gate this repo keeps finding.
    import shutil as _sh
    if _sh.which("iverilog"):
        results.append(run("L2' RTL == Python detector (bit-exact)",
                           [PY, "experiments/overhead/l2prime_rtl_gate.py",
                            "--vectors", "1500"], env=env))
    else:
        print("\n=== L2' RTL: NOT RUN (iverilog absent) ===")
        print("  the RTL detector is NOT being checked in this run")
    # RECEIPT V2 ACROSS A LANGUAGE BOUNDARY. Phase 1's exit gate reads "JS
    # agrees on every receipt"; it did not, because verify-certificate.mjs
    # checks the ORIGINAL certificate and knows nothing of a logical coset.
    # A second implementation in the same language proves the code
    # reproduces; a second LANGUAGE proves the SPEC is unambiguous, which
    # is what "a stranger can check this" actually requires.
    import shutil as _sh2
    if _sh2.which("node"):
        results.append(run("receipt v2: Python == JavaScript",
                           [PY, "tools/receipt_conformance_gate.py",
                            "--json", "evidence/receipt_conformance.json"],
                           env=env))
        if not a.fast:
            results.append(run("receipt conformance re-proves it can FAIL",
                               [PY, "tools/receipt_conformance_gate.py",
                                "--prove-sensitive"], env=env))
        # THE PASSPORT ACROSS THE SAME BOUNDARY, AT GT-A1's OWN NUMBERS
        # (2026-09-08, campaign 4.10). 15 hand-written cases, 1,000 seeded
        # passports across every failure category with the verdict each
        # category expects, and 10,000 whose integers span +-2^63. The
        # torture corpus found, the day it was written, that JSON.parse
        # rounds integers beyond 2^53: the JavaScript rejected genuine
        # passports at 2^53+1 and ACCEPTED a +1 forgery at 2^53. Three
        # seconds, one node process per corpus (`--batch`).
        results.append(run("passport: Python == JavaScript on 15 + 1,000 + 10,000",
                           [PY, "tools/conformance.py",
                            "--json", "evidence/conformance.json"], env=env))
        # THE OISC SUPPORT PROOF, ACROSS THE SAME LANGUAGE BOUNDARY [A8].
        # The release's lower bounds have no succinct proof object: the
        # LOCAL sweeps genuinely cost their enumeration. The GLOBAL
        # covering step does not, and this checks that the portable
        # document a stranger receives is decided identically by an
        # implementation that shares no code with the producer --
        # including the problem digest, which is what stops a valid
        # proof for an easy instance being replayed onto a hard one.
        results.append(run("OISC support proof: Python == JavaScript",
                           [PY, "-m", "pytest",
                            "tests/test_oisc_proof_conformance.py",
                            "-q", "-p", "no:randomly"], env=env))
        # THE ENCLOSURE TIER, ACROSS THE SAME BOUNDARY. Its certificate
        # is a SCHEDULE rather than a value, so "two languages agree" is
        # a stronger statement here than usual: both must reproduce the
        # same replay, the same exact integers, and the same refusals --
        # and 40 of the 82 vectors ARE refusals, because a corpus of
        # documents that should be accepted cannot tell a working
        # checker from one that says yes to everything.
        results.append(run("coset enclosure: Python == JavaScript",
                           [PY, "-m", "pytest",
                            "tests/test_enclosure_conformance.py",
                            "-q", "-p", "no:randomly", "-m", ""], env=env))
    else:
        # DELIBERATELY RED, not skipped. tests/test_cert_conformance.py
        # uses pytest.skipif for a missing node, and the L2' RTL gate
        # prints a notice -- both reasonable for a secondary detector.
        # This one DISCHARGES A NAMED EXIT GATE ("JS agrees on every
        # receipt"), so its absence is not a neutral fact about the
        # toolchain: it means Phase 1 is unproven in this run, and a wall
        # that stays green while saying so would be the dark gate this
        # project keeps finding. Install node or read the red honestly.
        #
        # *** THIS `else` SPENT WEEKS BOUND TO THE WRONG `if`. *** It was
        # written against `which("node")` in c04cd77. A later commit
        # inserted the OISC conformance gate and an
        # `if oisc_d8_attempt.json.exists():` block BETWEEN the body and
        # the `else`, and Python silently re-bound the `else` to the new
        # `if`. From then on the wall asked whether a JSON FILE existed
        # and printed "node absent" about the answer:
        #
        #   node ABSENT, that file present  ->  GREEN, Phase 1's JS exit
        #                                       gate silently unproven --
        #                                       precisely the dark gate
        #                                       this comment forbids
        #   node PRESENT, file absent       ->  a RED that names a cause
        #                                       that is not true
        #
        # On this machine both were true, so the wall was right by
        # coincidence and no run could have revealed it. Found by
        # `tools/gate_roster.py` on its first execution, from the guard
        # text alone -- a gate whose printed reason and whose condition
        # disagree is visible statically and invisible at runtime.
        print("\n=== receipt v2 conformance: NOT RUN (node absent) ===")
        print("  receipt v2 is NOT being checked across a language boundary "
              "in this run; Phase 1's exit gate is unproven here")
        results.append(("receipt v2 conformance (node absent)", False))
    # EVERY RECORDED CLOSURE, RE-DECIDED BY AN INDEPENDENT SOLVER.
    #
    # A closure is the load-bearing claim -- it turns "we swept some
    # sets" into "the published distance is reached" -- and it was
    # asserted by ONE program (ClosureEngine) for as long as it existed.
    # The hand-written branch-and-bound could not re-decide it (111M
    # nodes, 3000s, twice), so the check was not merely missing, it was
    # believed to be impossible. HiGHS does it in 78 seconds.
    #
    # Cheap enough to be standing rather than occasional, which is the
    # difference between a property and an anecdote.
    # THE INTEGER PROOFS, RE-CHECKED EVERY RUN, IN BOTH LANGUAGES.
    #
    # These are the only machine-checkable objects for the published d=8
    # closures -- the LP dual cannot certify them, because the gain OISC
    # exploits IS the integrality gap. A proof object nobody re-checks is
    # a file.
    if list((ROOT / "evidence" / "phase7").glob("branch_cert_*.json")):
        # *** THIS GATE RE-RAN WORK THE SUITE HAD ALREADY DONE. ***
        # Measured: the two files hold 16 tests, and the suite collects
        # 15 of them -- only the `slow_exhaustive` one is deselected
        # there. So this gate existed to add ONE test and was re-running
        # the other fifteen.
        #
        #     the 1 the suite CANNOT run   9.4 s
        #     the 15 it already ran      179.1 s
        #
        # 127 s of a 446 s wall, for 9 s of unique coverage. It now runs
        # exactly what the suite is unable to: `-m slow_exhaustive`.
        # `-p no:randomly` stays, because the point of this gate is a
        # deterministic exhaustive pass.
        results.append(run("d=8 branching certificates (exhaustive lane)",
                           [PY, "-m", "pytest",
                            "tests/test_branch_certificate.py",
                            "tests/test_branch_certificate_conformance.py",
                            "-q", "-p", "no:randomly",
                            "-m", "slow_exhaustive"], env=env))
    # THE CAPABILITY MANIFEST MUST STAY UNABLE TO ADMIT ANYTHING. The
    # shipped manifest is built entirely from the client library, and the
    # defect class it guards is a value drifting from sdk_expressible to
    # hardware_executable without a job behind it -- which is exactly how
    # `run_g4.py` came to report Qiskit's opinion as a provider
    # capability. Free and deterministic, so it runs every time.
    results.append(run("capability manifest admits nothing (SDK-sourced)",
                       [PY, "tools/capability_probe_run.py", "--gate"],
                       env=env))
    # DEM-CERT. The detector error model is the shared ancestor of every
    # decoder-assurance leg, so a drift here silently invalidates
    # everything downstream of it. The gate REBUILDS each circuit from
    # its recorded parameters and recomputes the root -- comparing a
    # stored root against itself would be the self-referential species
    # `tools/gate_lint.py` exists to flag. Under a second at d<=7.
    results.append(run("DEM-CERT roots re-derived (every fault signature)",
                       [PY, "tools/dem_cert_run.py", "--gate"], env=env))
    # GT-A0. The evidence algebra's bound, attacked with FRESH random
    # DAGs on a seed the shipped artifact never used -- so this is a new
    # falsification attempt each run rather than a replay of the one that
    # passed. A reduced budget (1e7 draws, ~2 s) because the full 5e9-draw
    # sweep is a three-minute job; the big one is pinned by the test
    # suite, this one is the live attempt.
    results.append(run("GT-A0 evidence algebra (fresh DAGs, new seed)",
                       [PY, "experiments/overhead/gt_a0_evidence_algebra.py",
                        "--dags", "400", "--trials", "25000",
                        "--seed", "917331",
                        "--json", "evidence/gt_a0_wall_sweep.json"], env=env))
    # AND THE SAME SWEEP PROVING IT CAN FAIL, on the SAME DAGs and the
    # SAME seed -- only the composition rule changes. Frechet finds 0
    # unsound there; the naive product finds hundreds here. A one-armed
    # sweep reporting "0 violations" is a number nobody has watched move.
    # L0 configuration integrity: the class `H x_hat = s` provably cannot
    # see. Both arms run -- with and without the dedicated mask replica --
    # because 100% detection alone is satisfiable by a check that always
    # refuses, and the difference between the arms IS the mechanism.
    results.append(run("L0 ABFT config integrity (paired mask-replica arms)",
                       [PY, "experiments/g3_abft/l0_config_integrity.py"],
                       env=env))
    results.append(run("GT-A0 negative control (naive OR must be caught)",
                       [PY, "experiments/overhead/gt_a0_evidence_algebra.py",
                        "--dags", "400", "--trials", "25000",
                        "--seed", "917331", "--negative-control",
                        "--json", "evidence/gt_a0_negative_control.json"],
                       env=env))
    # (The HiGHS re-decision is registered before the launch point, far
    # above. An empty `if ...oisc_d8_attempt.json.exists(): pass` used to
    # stand here as its residue -- and it was that vacuous `if` the
    # receipt-conformance `else` had silently attached itself to.)
    # DIFFERENTIAL EQUIVALENCE against the pre-refactor bytes. These are
    # not one-off migration checks: tools/frozen/ is a permanent
    # behavioural baseline, so any FUTURE edit to a checker or producer
    # that moves a verdict, a reason string, a bound or a tier goes red
    # here rather than being discovered in a receipt.
    # CONSERVATISM. `checker_equivalence_gate` pins VERDICTS; this pins
    # VALUES, because a bound that got WEAKER changes no verdict and is
    # therefore invisible to every other gate here. Measured twice in two
    # days: a discarded composed frontier and a dropped leg, both sound,
    # both silently costing capability. Fast and deterministic.
    # THE INFERENCE ENGINE, RE-CALIBRATED ON FRESH INSTANCES. The suite
    # pins specific instances so a failure is reproducible; this draws
    # new ones every run, so the calibration is not quietly conditioned
    # on the handful somebody wrote down. It is the only standing check
    # that the two exact engines still agree and that BP is still exact
    # where theory says it must be.
    # A RESEARCH THEOREM CANDIDATE, RE-ATTACKED ON FRESH INSTANCES. It
    # has survived 2,507 instances across seven adversarial families --
    # which is evidence and not a proof, so the attack is standing rather
    # than historical. The suite pins instances for reproducibility; this
    # draws new ones so the evidence is not conditioned on the handful
    # somebody wrote down.
    # L0 IN SILICON, OR IT STAYS A MODEL CLAIM. The layer's whole value
    # is a ONE-CYCLE verdict at the read, which is a statement about a
    # circuit. Bit-exact against the Python on the comparison, the sticky
    # bit, the validity rule, the mask replica and every refusal -- and
    # explicitly NOT about the hash datapath, which is not in the RTL.
    if _sh.which("iverilog"):
        results.append(run("L0 ABFT guard: RTL == Python (bit-exact)",
                           [PY, "tools/abft_l0_rtl_gate.py",
                            "--json", "evidence/abft_l0_rtl.json"],
                           env=env))
    else:
        print("\n=== L0 ABFT guard RTL: NOT RUN (iverilog absent) ===")
        print("  one-cycle configuration integrity remains a MODEL claim "
              "in this run, not a hardware one")
    # SIMULATION, REHEARSAL AND LIVE SPEAK ONE EVIDENCE LANGUAGE. The v2
    # wrapper grades a rehearsal receipt with the SAME soundness_budget
    # the live lane uses, inherits every refusal it makes, and adds the
    # one thing that actually differs: claim_kind. A second tier
    # vocabulary would let a simulated PROVEN be compared against a live
    # PROVEN by readers who never learn they were graded differently.
    # B2, REDUCED. 24 open directions became 3 yes/no questions the
    # moment the closure's 22 were used as a SUBSPACE rather than as 22
    # separate results. The gate re-derives the reduction and verifies
    # it; it does not answer the questions, and every one comes out
    # marked OPEN.
    # A DISTINCTIVENESS WORD MUST TRACE TO A REGISTER ROW. B4 said
    # "not-found != absent" for months, which is correct and stops
    # nothing: a sentence in a status document cannot prevent the next
    # README from saying "novel". The register is machine-read and the
    # prose is checked against it.
    # THE ENCLOSURE TIER'S OWN GATE. Fresh instances each run, brute-forced
    # in exact rationals, plus the hostile-schedule arm that exercises
    # E1/E2's generality rather than this module's beam. Its
    # --prove-sensitive form deletes the discard price and measures the
    # containment failures, so the gate has been watched to go red for
    # the right reason.
    results.append(run("coset enclosure brackets the truth (fresh instances)",
                       [PY, "tools/enclosure_gate.py",
                        "--json", "evidence/enclosure_gate.json"], env=env))
    if not a.fast:
        results.append(run("enclosure gate re-proves it can FAIL",
                           [PY, "tools/enclosure_gate.py",
                            "--prove-sensitive"], env=env))
    results.append(run("enclosure conformance corpus matches the checker",
                       [PY, "tools/build_enclosure_vectors.py", "--check"],
                       env=env))
    # THE LIVE LANE'S OWN does_not_claim, ANSWERED. `rep_code_cert.py`
    # ends its artifact saying delta_O, R_I and delta_M need P(Y=1|s),
    # "and bayes_coset refuses above kernel dimension 22 while this
    # graph's kernel is far larger". On the job's own configuration that
    # kernel is 39, and the enclosure brackets P(Y=1|s) per shot at
    # 2 ms. This runs the MIRROR -- no hardware, no credits -- because
    # the point of the gate is that the tier is ready for the owner's
    # one --fetch, not that it has already seen a device.
    results.append(run("enclosure closes the live lane's open budget terms",
                       [PY, "experiments/ibm_live/enclosure_on_the_live_lane.py"],
                       env=env))
    # V_E WITH A CERTIFICATE, AND ITS OWN CONTROL. The gate's criterion is
    # not the ratio -- it is that a SCRAMBLED herald must not beat no
    # herald. A clean, quotable V_E that survives a scrambled herald is a
    # ratio of an artifact, which is exactly how the estate's earlier
    # V_E = 1.000 came to be quoted at all.
    results.append(run("certified V_E, and a scrambled herald must not help",
                       [PY, "experiments/enclosure/certified_erasure_value.py"],
                       env=env))
    results.append(run("prior-art register: every novelty claim traces",
                       [PY, "tools/prior_art_gate.py",
                        "--json", "evidence/prior_art_gate.json"], env=env))
    # THE BARCODE LEVEL ENGINE, ON A CODE WHOSE ANSWER IS KNOWN.
    # [[72,12,6]] must come back flat at d=6 across all 24 levels, in
    # under a second. It is the only standing check that the whole-level
    # formulation still answers the question it claims -- three silent
    # convention bugs were found in this area in one session, and every
    # one of them produced a plausible wrong answer.
    results.append(run("barcode level engine (known code, flat at d)",
                       [PY, "tools/b3_distance_portfolio.py", "--barcode",
                        "--budget", "60",
                        "--json", "evidence/b3_barcode.json"], env=env))
    results.append(run("B2 quotient reduction (24 directions -> 3 questions)",
                       [PY, "tools/b2_quotient_questions.py",
                        "--json", "evidence/b2_questions.json"], env=env))
    results.append(run("rehearsal v2 speaks the live evidence language",
                       [PY, "tools/rehearsal_v2_mint.py",
                        "release/phase4-rehearsal",
                        "--claim-kind", "rehearsal",
                        "--json", "evidence/rehearsal_v2.json"], env=env))
    # THE PUBLISHED COUNTEREXAMPLE TO THE DUAL FORM, RUN EVERY WALL.
    # `matching_cert_strict` landed on main; the forgery that motivated
    # it stayed in an unmerged 2026-08-17 draft branch, so nothing here
    # would have noticed if the refusal regressed. It carries a POSITIVE
    # control: a checker stuck on "no" refuses every forgery and would
    # otherwise pass a forgery-only test.
    # THE PILOT'S INSTRUMENT, ON WORLDS WHOSE FAULT IS KNOWN. The
    # identity adding up is arithmetic; what earns the hardware spend is
    # that each term LOCALIZES its own fault. Five worlds, one broken
    # mechanism each (one of them LOCAL to a patch), every term checked
    # against a prediction derived from the world parameters, and the
    # local fault must be louder in its own slice than globally.
    results.append(run("error budget localizes the fault (5 known worlds)",
                       [PY, "tools/error_budget_report.py", "--worlds",
                        "--shots", "4000",
                        "--json", "evidence/error_budget.json"], env=env))
    # D2: HOW WRONG COULD THE DECLARED WEIGHTS BE? Every receipt here
    # certifies against weights we DECLARE uniform, and the robustness
    # tool had only ever run on its own demo. This scores real radii
    # against a MEASURED sigma -- `ibm_marrakesh`'s own pinned
    # calibration, minus the declaration, both scale-normalised so the
    # log-likelihood units cannot masquerade as disagreement.
    results.append(run("certificate robustness on a measured sigma (D2)",
                       [PY, "tools/certificate_robustness_corpus.py",
                        # 1200 so the ARTIFACT backs the README's
                        # headline. At 300 the tightest radius has not
                        # converged and the doc would cite a number the
                        # evidence file does not contain -- 2.7 s.
                        "--shots", "1200",
                        "--json", "evidence/robustness_corpus.json"],
                       env=env))
    # THE LIVE LANE'S OBSERVABLE, WHICH IT ADVERTISED AND DID NOT HAVE.
    # `--validate` is the offline stim mirror -- no network, no credits.
    # It now scores the decoder's logical class against stim's own
    # observable on BOTH boundaries: the declared cut must track it and
    # the opposite one must measurably not, or the parity rule is not
    # reading a logical class at all. The first version of that cut
    # named the wrong boundary and only this check could see it.
    # No availability guard: stim is a hard dependency of this repository
    # and a missing one must go RED with its own ImportError, not skip.
    results.append(run("live lane observable (stim mirror, both cuts)",
                       [PY, "experiments/ibm_live/rep_code_cert.py",
                        "--mode", "validate", "--distance", "9",
                        "--rounds", "5", "--shots", "400",
                        "--p", "0.01"], env=env))
    # THE CEILING THE EXACT BAYES TIER IMPOSED, REMOVED. d=9 r=5 is the
    # LIVE lane's own configuration and its kernel dimension is 46:
    # `bayes_coset` refuses at 22 -- "an enclosure tier is required at
    # this size" -- and `coset_enclosure` is that tier. This runs the
    # full four-term budget there, which `HARDWARE_PREFLIGHT_GATES.md`
    # had recorded as out of reach regardless.
    results.append(run("error budget past the exact tier (d9r5, kernel 46)",
                       [PY, "tools/error_budget_ledger.py",
                        "--shots", "60", "--enclosure", "40",
                        "--json", "evidence/shot_ledger_enclosure.json"],
                       env=env))
    # AND THE SAME BUDGET ON SHOTS NOBODY CONSTRUCTED. The worlds above
    # declare their own fault; this runs A (pymatching), E (the certified
    # exact tier) and B (exact coset masses) over a repetition-code lane
    # with planted truth, and plants a REAL wrong-DEM in ONE (d, r) cell:
    # that cell is generated at 3x the rate the decoder and the Bayes
    # tier are told about. The fault must leave delta_M's band in that
    # cell, stay inside it everywhere else, and never touch delta_S or
    # delta_O -- a model bug billed to the decoder is the failure the
    # four signed terms exist to prevent.
    results.append(run("error budget localizes a REAL wrong-DEM (A/E/B/Y)",
                       [PY, "tools/error_budget_ledger.py",
                        "--shots", "250", "--mismatch", "d5r2",
                        "--out", "evidence/shot_ledger.jsonl",
                        "--json", "evidence/shot_ledger.json"], env=env))
    results.append(run("matching dual forgery refused, real cut accepted",
                       [PY, "challenge/matching_mixed_dual_forgery.py",
                        "--json", "evidence/matching_forgery.json"],
                       env=env))
    results.append(run("robustness theorem re-attacked (fresh instances)",
                       [PY, "tools/robustness_theorem_gate.py",
                        "--json", "evidence/robustness_theorem.json"],
                       env=env))
    results.append(run("latent-parity estimator calibration (fresh draws)",
                       [PY, "tools/latent_parity_gate.py",
                        "--json", "evidence/latent_parity_gate.json"],
                       env=env))
    results.append(run("conservatism ratchet (a weaker bound is a defect)",
                       [PY, "tools/conservatism_gate.py"], env=env))
    # THE CAMPAIGN ARMS, RUN. Phase 3's exit gate is "all six arms green
    # WITH their sensitivity probes green", and until 2026-08-20 nothing
    # ran them: no gate, no test, no CI, no evidence artifact. The exit
    # gate for a campaign that spends QPU credits had never been
    # mechanically verified. They cost ~6 s in total.
    results.append(run("campaign arms + their sensitivity probes",
                       [PY, "tools/campaign_arms_gate.py"], env=env))
    # EVERY HASH PIN, AGAINST THE OBJECT DATABASE -- what a stranger's
    # clone receives, not what this worktree holds. The Phase 6 seal
    # broke on 2026-08-17 and nothing went red for two days because the
    # only reader used the comparison to pick a beacon branch. This gate
    # is that class, closed: it found 17 pins in the closure bundle that
    # never matched since publication and 5 in an unsealed V2 binding
    # labelled READY TO SEAL, both now written up and counted exactly.
    results.append(run("hash pins verify from a clean clone",
                       [PY, "tools/hash_pin_audit.py"], env=env))
    results.append(run("checker equivalence vs frozen bytes",
                       [PY, "tools/checker_equivalence_gate.py",
                        "--cases", "400" if a.fast else "1500"], env=env))
    results.append(run("producer equivalence (yield + tiers + trees)",
                       [PY, "tools/producer_equivalence_gate.py",
                        "--cases", "40" if a.fast else "120"], env=env))
    # THE ARBITERS MUST RE-PROVE THEMSELVES. Their sensitivity was
    # established once, by hand, and nothing re-established it -- so a
    # corpus that rotted (a generator edited, a producer signature
    # changed, a leg going dark) would leave both gates comparing
    # nothing and printing EQUIVALENT forever. That is the exact failure
    # class they exist to catch, and they were not immune to it.
    # Expensive, so full runs only; --fast skips it and says so.
    if not a.fast:
        pass   # deferred: registered before the launch point

    else:
        print("\n=== arbiter sensitivity: SKIPPED (--fast) ===")
        print("  the equivalence gates are not re-proving they can fail "
              "in this run; a full gate_all does that")
    # estate vacuity audit (ATLAS LAW: reuse, don't rebuild). 3 known
    # EMPTY_PARAMS remain: the TINY parametrize is guarded by a module-level
    # `assert TINY` the static tool cannot see -- threshold 3, may only fall.
    vac = ROOT.parent / "Coherence" / "coherence_lang" / "scripts" / \
        "gate_vacuity_audit.py"
    if vac.exists():
        print("\n=== test vacuity (estate tool) ===", flush=True)
        r = subprocess.run([PY, str(vac), "tests/"], cwd=ROOT, env=env,
                           capture_output=True, encoding="utf-8",
                           errors="replace")
        tail = (r.stdout or "").strip().splitlines()
        n_find = 0
        for ln in reversed(tail):
            if "finding(s)" in ln:
                n_find = int(ln.split()[0])
                break
        print(f"  {n_find} finding(s); threshold 3 (module-level-assert "
              f"guards the static tool cannot see)")
        results.append(("test vacuity (estate tool)", n_find <= 3))
    else:
        # A GATE THAT EXISTS ON ONE WORKSTATION IS NOT A GATE. This
        # tool lives OUTSIDE the repository, so on every other machine
        # -- including CI -- the `if` simply fell through: nothing was
        # appended to `results`, so the summary neither reported it nor
        # counted it as skipped. It read as green by omission.
        #
        # Recorded as RED with a reason, the way the node-absent branch
        # already does. An unrunnable gate is an unmet gate; vendoring
        # the tool or dropping it are the only ways to clear this.
        print("\n=== test vacuity (estate tool) ===", flush=True)
        print(f"  UNAVAILABLE: {vac} is outside this repository and is "
              f"not present. Recorded as RED rather than skipped -- a "
              f"gate that runs on one workstation protects one "
              f"workstation.")
        results.append(("test vacuity (estate tool) -- UNAVAILABLE",
                        False))
    results.append(run("ratchet + absolute soundness sweep",
                       [PY, "tools/ratchet_gate.py"]
                       + (["--fast"] if a.fast else []), env=env))
    results.append(run("bench ledger (judged only when stable)",
                       [PY, "tools/bench_gate.py"], env=env))

    results.extend(_collect_deferred(_started))
    _publish_staged()

    print("\n" + "=" * 46)
    # Results are (name, ok) from the inline gates and (name, ok, seconds)
    # from `run`. Normalised here rather than at every append site.
    rows = [(r[0], r[1], r[2] if len(r) > 2 else None) for r in results]
    rows += _roster_reconciliation(rows, fast=a.fast)
    bad = [n for n, ok, _ in rows if not ok]
    for n, ok, dt in rows:
        cost = f"{dt:7.1f}s" if dt is not None else "       -"
        print(f"  {'PASS' if ok else 'RED '} {cost}  {n}")

    # *** WHERE THE TIME GOES, EVERY RUN. *** This wall took 13 to 18
    # minutes and recorded NOTHING about its own cost, so "it is too
    # slow" could not be turned into "this gate is why". An optimisation
    # proposed without this table is a guess.
    timed = sorted([r for r in rows if r[2] is not None], key=lambda r: -r[2])
    total = sum(r[2] for r in timed)
    if timed:
        print(f"\n  {total:.0f}s across {len(timed)} timed gates. The five "
              f"most expensive:")
        for n, _ok, dt in timed[:5]:
            share = 100.0 * dt / total if total else 0.0
            print(f"    {dt:7.1f}s  {share:4.1f}%  {n}")

    if bad:
        print(f"\nGATE_ALL: RED ({', '.join(bad)})")
        return 1
    print("\nGATE_ALL: GREEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
