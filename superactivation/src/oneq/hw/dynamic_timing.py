"""The feed-forward's timing on ibm_kingston, MEASURED from the server's executed schedule (device model v2).

R166 (jobs dba0ds8c77vs73824c3g, dba0dufi0auc7391ap5g; twirling on) and R173 (job dba0jsvi0auc7391b5gg; twirling off)
read the same feed-forward circuit's schedule, on the same qubits and calibration. Both report the same timing to
4 ns, so it is the control system's, not twirling's:
- the read occupies 1.200 us (four rows: 256 + 432 + 256 + 256 ns), against the vendor's meas_duration of 2.18 us;
- the Hub broadcast starts 0.484 us into the read and lasts 2.104 us, then a 28 ns receive;
- the conditional gate starts 2.704 us after the read starts, 1.504 us after it ends.
A qubit the conditional does not touch is NOT stalled by it (R166): only the conditioned qubits wait.

tests/test_dynamic_timing.py recomputes every number below from the committed flight records with
measured_from_schedule, so a constant cannot drift from its evidence."""
from __future__ import annotations

READ_S = 1.200e-6
BROADCAST_OFFSET_S = 0.484e-6
BROADCAST_S = 2.104e-6
RECEIVE_S = 28e-9
READ_START_TO_CONDITIONAL_S = 2.704e-6
SOURCES = ("evidence/superactivation/r166_executed_schedule/flight_dba0ds8c77vs73824c3g.json",
           "evidence/superactivation/r166_executed_schedule/flight_dba0dufi0auc7391ap5g.json",
           "evidence/superactivation/r173_twirl_off/flight_dba0jsvi0auc7391b5gg.json")


def hold_s() -> float:
    """how long a conditioned qubit waits from its read's start until the conditional gate can act: the stand-in hold
    of a mid-circuit read with feed-forward (hw_dynamic used the vendor read + an ASSUMED 0.65 us = 2.83 us)"""
    return READ_START_TO_CONDITIONAL_S


SLOT_S = 64e-9            # R166: a twirled single-qubit run plays as ONE two-sx template slot
CZ_S = 68e-9              # R166: the CZ itself; a twirled CZ layer is a slot + the CZ = 132 ns
SX_S = 32e-9              # untwirled: each sx / x pulse (rz is virtual)


def conditional_waits(qc, twirled: bool = True, policy: str = "asap") -> list:
    """how long each CONDITIONED qubit idles at each if_test, on an as-soon-as-possible timeline of what the server
    plays (R166): a twirled single-qubit run is one SLOT_S slot and every twirled CZ gets a slot before it (layer
    SLOT_S + CZ_S); untwirled, each sx/x costs SX_S and rz nothing; a read takes READ_S; a barrier aligns its qubits;
    an if_test's body starts READ_START_TO_CONDITIONAL_S after the START of the latest read its condition uses, and a
    conditioned qubit ready before then WAITS. Returns, per if_test in circuit order, {qubit index: wait in s}.

    This is the early key's wait in R147's early schedule: the barrier releases the key only when the shield reads
    END, the key's preparation takes a few layers, and its conditional Z still cannot act until 1.504 us after the
    read ended -- the stall the stand-ins never charged it (R171's copy bit implies about 1.6 us).

    FALSIFIED as a statement of where the server puts the wait (R183, 2026-10-19, job dbap0i1gv2cc73e76mb0;
    `executed_key_wait` on all 130 R147 units with a conditional). The server holds the barrier before the key's
    preparation until the feed-forward has been RECEIVED, so the key idles in |0> (median 3.0 us, where dephasing
    cannot act). It then prepares and meets its conditional 0.04-0.18 us later (median 0.11 us), not 2.57 us.
    The stall falls instead on every qubit holding a quantum state across that barrier: `quantum_idle` reads 1.0-5.2 us
    more per circuit (median 2.9) than the frame-tracked twins. This model is kept as the ASAP bound it computes; a
    claim about the executed schedule reads the schedule.

    policy="alap" is the server's own rule, MEASURED (R194, job dbbfkevi0auc7393harg; R196, job dbbfmnpgv2cc73e8c6q0):
    an operation is played as late as what depends on it allows. A key with no barrier still met its conditional
    0.11 us after its last play (0.04-0.27 over 130 units; ASAP said 2.572), and a read nothing waits on was deferred
    to the end. So a conditioned qubit can only be made to wait in a quantum state by work the condition's read
    DEPENDS ON: its operations that are ancestors of that read are pinned at their ASAP times, everything after them
    slides late, and the wait is cond_t - (end of its last pinned operation) - (the time its unpinned operations
    take). With no pinned operation the qubit is prepared just in time: wait 0. (R196's magic data qubit, entangled
    INTO the read by a CX, is pinned and waits the full feed-forward: 2.58 us on Fez, as played.) The constants are
    ibm_kingston's (R166); the RULE is the server's."""
    from qiskit.circuit import ClassicalRegister, Clbit
    if policy not in ("asap", "alap"):
        raise ValueError(f"policy {policy!r}: 'asap' (the bound) or 'alap' (the server's rule)")
    t: dict = {}
    run: dict = {}
    read_start: dict = {}
    out = []
    # for policy="alap": per instruction, its transitive dependencies; per qubit, its instructions and its busy time
    anc: list = []
    last_on: dict = {}
    last_write: dict = {}
    ops_on: dict = {}
    busy: dict = {}
    busy_at: dict = {}
    end_at: dict = {}

    def close(q):
        if run.get(q):
            cost = SLOT_S if twirled else run[q]
            t[q] = t.get(q, 0.0) + cost
            busy[q] = busy.get(q, 0.0) + cost
        run[q] = 0.0 if not twirled else False

    def pending(q) -> float:
        """the time a still-open single-qubit run on q will cost when it closes"""
        if not run.get(q):
            return 0.0
        return SLOT_S if twirled else run[q]

    def bits_of(cond) -> list:
        if isinstance(cond, tuple):
            c = cond[0]
            return [c] if isinstance(c, Clbit) else list(c)
        found = []

        def walk(node):
            v = getattr(node, "var", None)
            if isinstance(v, Clbit):
                found.append(v)
            elif isinstance(v, ClassicalRegister):
                # a register iterates through __getitem__ and has no __iter__: tested by type, not by duck typing
                # (the first version asked hasattr(__iter__) and so refused every condition on a whole register)
                found.extend(list(v))
            for k in ("left", "right", "operand"):
                if hasattr(node, k):
                    walk(getattr(node, k))
        walk(cond)
        return found

    for i, inst in enumerate(qc.data):
        op = inst.operation
        qs = [qc.find_bit(q).index for q in inst.qubits]
        name = op.name
        deps = {last_on[q] for q in qs if q in last_on}
        used: list = []
        if getattr(op, "blocks", None):
            used = [qc.find_bit(c).index for c in bits_of(op.condition)]
            deps |= {last_write[c] for c in used if c in last_write}
        mine = set(deps)
        for d in deps:
            mine |= anc[d]
        anc.append(mine)
        if name in ("rz", "sx", "x", "u", "u1", "u2", "u3", "h", "s", "sdg", "y", "z", "id"):
            q = qs[0]
            if twirled:
                run[q] = True
            else:
                run[q] = (run.get(q) or 0.0) + (SX_S if name in ("sx", "x") else 0.0)
        elif name == "cz":
            a, b = qs
            if twirled:
                # a run before the CZ IS its layer's slot (R166: slot + CZ per layer), and the layer leaves a closing
                # slot owed on both qubits -- absorbed by a next CZ, charged at a read, barrier, conditional or delay
                start = max(t.get(a, 0.0), t.get(b, 0.0))
                t[a] = t[b] = start + SLOT_S + CZ_S
                run[a] = run[b] = True
                for q in (a, b):
                    busy[q] = busy.get(q, 0.0) + SLOT_S + CZ_S
            else:
                close(a)
                close(b)
                start = max(t.get(a, 0.0), t.get(b, 0.0))
                t[a] = t[b] = start + CZ_S
                for q in (a, b):
                    busy[q] = busy.get(q, 0.0) + CZ_S
        elif name == "measure":
            q = qs[0]
            close(q)
            read_start[qc.find_bit(inst.clbits[0]).index] = t.get(q, 0.0)
            t[q] = t.get(q, 0.0) + READ_S
            busy[q] = busy.get(q, 0.0) + READ_S
            last_write[qc.find_bit(inst.clbits[0]).index] = i
        elif name == "barrier":
            for q in qs:
                close(q)
            m = max((t.get(q, 0.0) for q in qs), default=0.0)
            for q in qs:
                t[q] = m
        elif name == "delay":
            q = qs[0]
            close(q)
            d = float(op.duration) * (1e-9 if op.unit == "ns" else 1.0 if op.unit == "s" else 0.0)
            t[q] = t.get(q, 0.0) + d
            busy[q] = busy.get(q, 0.0) + d
        elif getattr(op, "blocks", None):
            if not used or any(c not in read_start for c in used):
                raise ValueError(f"{name}: its condition reads no earlier measurement")
            cond_t = max(read_start[c] for c in used) + READ_START_TO_CONDITIONAL_S
            reads = {last_write[c] for c in used}
            pinned = set(reads)
            for r in reads:
                pinned |= anc[r]
            waits = {}
            for q in qs:
                close(q)
                if policy == "asap":
                    waits[q] = max(0.0, cond_t - t.get(q, 0.0))
                else:
                    pins = [j for j in ops_on.get(q, []) if j in pinned]
                    if not pins:
                        waits[q] = 0.0                     # nothing the read needs: prepared just in time
                    else:
                        last = pins[-1]
                        slid = busy.get(q, 0.0) - busy_at[(last, q)]
                        waits[q] = max(0.0, cond_t - end_at[(last, q)] - slid)
                t[q] = max(t.get(q, 0.0), cond_t) + SLOT_S
                busy[q] = busy.get(q, 0.0) + SLOT_S
            out.append(waits)
        elif name in ("reset",):
            q = qs[0]
            close(q)
            t[q] = t.get(q, 0.0) + READ_S
            busy[q] = busy.get(q, 0.0) + READ_S
        else:
            raise ValueError(f"{name}: no timing for this instruction")
        for q in qs:
            last_on[q] = i
            ops_on.setdefault(q, []).append(i)
            end_at[(i, q)] = t.get(q, 0.0) + pending(q)
            busy_at[(i, q)] = busy.get(q, 0.0) + pending(q)
    return out


def measured_from_schedule(rows: list, read_qubit: int, dt: float) -> dict:
    """the feed-forward's timing from one circuit's executed schedule (oneq.hw.timing rows): the read of read_qubit
    (its first measurement), the broadcast, the receive and the first conditional (branch 'then') play"""
    plays = sorted((r for r in rows if r.qubit == read_qubit and r.gate == "measure"), key=lambda r: r.t0)
    if not plays:
        raise ValueError(f"no measurement of qubit {read_qubit} in the schedule")
    start = plays[0].t0
    first_block = [r for r in plays if r.t0 < start + 4096]           # the read's rows, before any later read
    end = max(r.t0 + r.duration for r in first_block)
    b = next((r for r in rows if r.gate == "broadcast"), None)
    rc = next((r for r in rows if r.gate == "receive"), None)
    cond = min((r for r in rows if r.branch == "then" and r.pulse == "play"), key=lambda r: r.t0, default=None)
    if b is None or rc is None or cond is None:
        raise ValueError("the schedule has no broadcast, receive or conditional play")
    return dict(read_s=(end - start) * dt, broadcast_offset_s=(b.t0 - start) * dt, broadcast_s=b.duration * dt,
                receive_s=rc.duration * dt, read_start_to_conditional_s=(cond.t0 - start) * dt)


# ================================================================== waits READ from an executed schedule (R183)
_FRAMING = ("INIT", "FINI")          # the server's own per-shot preparation and teardown blocks


def _working(rows: list, q: int) -> list:
    """qubit q's physical plays in the main branch, in time order, the INIT / FINI blocks left out"""
    return sorted((r for r in rows if r.qubit == q and r.branch == "main" and r.pulse == "play"
                   and r.gate not in _FRAMING), key=lambda r: r.t0)


def executed_key_wait(rows: list, key: int, dt: float) -> dict:
    """the conditioned qubit's timing AS EXECUTED (oneq.hw.timing rows):
      wait_s   -- from the end of its last physical play before its first conditional ('then' row) to that row;
      ground_s -- from the end of its INIT block to its first working play (it idles in |0>, out of dephasing's reach);
      start_s  -- its first working play's time.
    Refused (ValueError) when the key has no conditional in the schedule or plays nothing before it."""
    then = sorted((r for r in rows if r.qubit == key and r.branch == "then"), key=lambda r: r.t0)
    if not then:
        raise ValueError(f"qubit {key} has no conditional in the schedule")
    work = _working(rows, key)
    before = [r.end for r in work if r.end <= then[0].t0]
    if not before:
        raise ValueError(f"qubit {key} plays nothing before its conditional")
    inits = [r.end for r in rows if r.qubit == key and r.gate == "INIT" and r.pulse == "play"]
    return dict(wait_s=(then[0].t0 - max(before)) * dt, start_s=work[0].t0 * dt,
                ground_s=((work[0].t0 - max(inits)) * dt) if inits else None)


def quantum_idle(rows: list, dt: float) -> dict:
    """per qubit, the idle AS EXECUTED split by what the qubit holds:
      quantum   -- gaps between consecutive working plays whose earlier play is not a read (the qubit carries a state
                   dephasing and decay act on);
      post_read -- gaps after a mid-circuit read (a collapsed bit: only T1 acts, on a 1);
    and their totals. The time before a qubit's first working play (in |0>) is neither."""
    out: dict = {"per_qubit": {}, "quantum_s": 0.0, "post_read_s": 0.0}
    for q in sorted({r.qubit for r in rows if r.qubit is not None}):
        work = _working(rows, q)
        qi = pr = 0
        for a, b in zip(work, work[1:]):
            gap = max(0, b.t0 - a.end)
            if a.gate == "measure":
                pr += gap
            else:
                qi += gap
        out["per_qubit"][q] = dict(quantum_s=qi * dt, post_read_s=pr * dt)
        out["quantum_s"] += qi * dt
        out["post_read_s"] += pr * dt
    return out
