r"""wu_model.py -- WINDOW_PROOF.md sections 2-4 made mechanical.

Machine(kind, basis): the blocks P, F, B, E, R of the template, the labelling rule (LBL), the two finite matrix
identities FIR(h) and END(e) on the body block's maps, and the EXACT model of C_T computed block by block from the
frame rule (a backward pass at block granularity: resp_i(x) = what a frame x entering block position i flips).

Detector index of coordinate (k, type, s): s * 2 n2 + type * n2 + k.  Symptom = (detector mask, observable mask).
"""
from __future__ import annotations

import sys

import wu_core as wc  # noqa: E402

bits = wc.bits


class Machine:
    def __init__(self, kind: str, basis: str, blocks: dict | None = None):
        self.kind, self.basis = kind, basis
        self.code = wc.Code(basis)
        self.tpl = wc.Template(kind, basis)
        n2, nq = self.code.n2, self.code.nq
        self.n2, self.nq, self.W = n2, nq, 2 * n2           # W = detector bits per round
        if blocks is None:
            blocks = {nm: wc.Block(getattr(self.tpl, nm), nq, nm) for nm in "PFBER"}
        self.blk = blocks
        self.FR = 2 * nq
        self.fmask = (1 << self.FR) - 1
        self.memmask_round = ((1 << n2) - 1) << (self.code.bi * n2)

    # ------------------------------------------------------------------ LBL: the labelling rule
    def lbl(self) -> dict:
        """B measures every ancilla exactly once; F a subset once each; E exactly the rest once each; R every data
        qubit once; P measures nothing. Then the measurement of ancilla q in block position j >= 1 is round
        j + delta_q with delta_q = 0 if q is measured in F else -1, F's are round 0, E's are round T-1."""
        c = self.code
        anc = set(range(c.nd, c.nq))
        mq = {nm: [q for q, _ in self.blk[nm].meas] for nm in "PFBER"}
        ok = dict(P_measures_nothing=not mq["P"],
                  B_each_ancilla_once=sorted(mq["B"]) == sorted(anc),
                  F_subset_once=len(set(mq["F"])) == len(mq["F"]) and set(mq["F"]) <= anc,
                  E_is_the_rest_once=sorted(mq["E"]) == sorted(anc - set(mq["F"])),
                  R_each_data_once=sorted(mq["R"]) == list(range(c.nd)))
        self.delta = {q: (0 if q in set(mq["F"]) else -1) for q in anc}
        nd0 = sum(1 for q in anc if self.delta[q] == 0)
        return dict(holds=all(ok.values()), **ok, delta0=nd0, delta_minus1=len(anc) - nd0)

    # ------------------------------------------------------------------ labels and detector masks for C_T
    def det_index(self, k: int, ti: int, s: int) -> int:
        return s * self.W + ti * self.n2 + k

    def label_masks(self, T: int, cf=None):
        """{record label: (detector mask, observable mask)} from the closed form of C_T (or a doctored one)."""
        dets, obs = cf if cf is not None else wc.closed_form(self.code, T)
        out: dict = {}
        for (k, ti, s), labs in dets.items():
            bit = 1 << self.det_index(k, ti, s)
            for lab in labs:
                d, o = out.get(lab, (0, 0))
                out[lab] = (d ^ bit, o)
        for j, labs in obs.items():
            for lab in labs:
                d, o = out.get(lab, (0, 0))
                out[lab] = (d, o ^ (1 << j))
        return out

    def positions(self, T: int):
        """[(position, block name, [label of each measurement slot])] for C_T: P at -1, F at 0, B at 1..T-1, E at
        T, R at T+1 (the labels follow LBL)."""
        out = [(-1, "P", [])]
        for j in range(T):
            nm = "F" if j == 0 else "B"
            labs = [(q, 0) if nm == "F" else (q, j + self.delta[q]) for q, _ in self.blk[nm].meas]
            out.append((j, nm, labs))
        out.append((T, "E", [(q, T - 1) for q, _ in self.blk["E"].meas]))
        out.append((T + 1, "R", [(q, 0) for q, _ in self.blk["R"].meas]))
        return out

    # ------------------------------------------------------------------ exact model by block iteration
    def responses(self, T: int, cf=None, lm=None):
        """resp[pos] = list over the 2 NQ frame basis bits of (det mask, obs mask) caused by that unit frame
        entering block position pos; slotm[pos] = per measurement slot (det mask, obs mask). `lm` replaces the
        label -> (mask, mask) map (record-level responses use one bit per record)."""
        if lm is None:
            lm = self.label_masks(T, cf)
        pos = self.positions(T)
        slotm = {p: [lm.get(lab, (0, 0)) for lab in labs] for p, nm, labs in pos}
        resp: dict = {T + 2: [(0, 0)] * self.FR}
        for p, nm, labs in reversed(pos):
            b = self.blk[nm]
            nxt = resp[p + 1]
            row = []
            for col in b.col:
                d = o = 0
                for mi in bits(col >> self.FR):
                    sd, so = slotm[p][mi]
                    d ^= sd
                    o ^= so
                for fb in bits(col & self.fmask):
                    rd, ro = nxt[fb]
                    d ^= rd
                    o ^= ro
                row.append((d, o))
            resp[p] = row
        return resp, slotm, pos

    def symptom(self, mask: int, p: int, resp, slotm):
        """(det mask, obs mask) of a direct effect `mask` (frame out | meas flips << FR) located in position p."""
        d = o = 0
        for mi in bits(mask >> self.FR):
            sd, so = slotm[p][mi]
            d ^= sd
            o ^= so
        nxt = resp[p + 1]
        for fb in bits(mask & self.fmask):
            rd, ro = nxt[fb]
            d ^= rd
            o ^= ro
        return d, o

    def proj(self, d: int, T: int) -> int:
        """Keep the memory-type detector bits of a mask over rounds 0..T."""
        m = 0
        for s in range(T + 1):
            m |= self.memmask_round << (s * self.W)
        return d & m

    def all_symptoms(self, T: int, what: str = "comps", cf=None, lm=None):
        """[(position, block name, index in the block's list, det mask, obs mask)] for every noise component
        ('comps') or every gauge ('gauges') of C_T. Gauges include the initial-state gauges Z_q (every qubit
        starts in |0>), entered as frames into P."""
        resp, slotm, pos = self.responses(T, cf, lm)
        out = []
        for p, nm, _ in pos:
            lst = self.blk[nm].comps if what == "comps" else self.blk[nm].gauges
            for i, item in enumerate(lst):
                d, o = self.symptom(item[-1], p, resp, slotm)
                out.append((p, nm, i, d, o))
        if what == "gauges":
            P = self.blk["P"]
            for q in range(self.nq):
                d, o = self.symptom(P.col[self.nq + q], -1, resp, slotm)
                out.append((-2, "init", q, d, o))
        return out

    def record_labels(self, T: int, circuit=None) -> list:
        c = circuit if circuit is not None else wc.build(self.kind, self.basis, T)
        labels, _, _, _ = wc.records(wc.flat(c))
        return labels

    def gauge_rank(self, T: int, drop=None) -> dict:
        """(G) instrument check: the GF(2) rank of the gauges' record-flip vectors of C_T against stim's tableau
        count of random measurement outcomes (tools_cde._measurement_counts). `drop(position, block, index)`
        removes gauges (negative control)."""
        import tools_cde as tc
        c = wc.build(self.kind, self.basis, T)
        labels = self.record_labels(T, c)
        lm = {lab: (1 << i, 0) for i, lab in enumerate(labels)}
        vecs = [d for p, nm, i, d, o in self.all_symptoms(T, "gauges", lm=lm) if not (drop and drop(p, nm, i))]
        piv: dict = {}
        for v in vecs:
            while v:
                hb = v.bit_length() - 1
                if hb not in piv:
                    piv[hb] = v
                    break
                v ^= piv[hb]
        n, rnd = tc._measurement_counts(c)
        return dict(T=T, measurements=n, gauge_rank=len(piv), tableau_random=rnd, equal=len(piv) == rnd)

    def model(self, T: int, projected: bool, cf=None) -> set:
        """The set of distinct nonzero symptoms (the DEM's mechanisms), full or same-type projected."""
        out = set()
        for p, nm, i, d, o in self.all_symptoms(T, "comps", cf):
            if projected:
                d = self.proj(d, T)
            if d or o:
                out.add((d, o))
        return out

    # ------------------------------------------------------------------ the finite identities FIR and END
    def _rows(self, kind: str) -> int:
        """Mask over B's measurement slots: 'mem' = memory-type checks, 'all' = every check."""
        m = 0
        for i, (q, _) in enumerate(self.blk["B"].meas):
            ti, _k = self.code.check_of(q)
            if kind == "all" or ti == self.code.bi:
                m |= 1 << i
        return m

    def A(self, x: int) -> int:
        return self.blk["B"].A(x)

    def Apow(self, x: int, h: int) -> int:
        for _ in range(h):
            x = self.A(x)
        return x

    def fir(self, h: int, rows: str = "all", A=None, C=None) -> dict:
        """FIR(h): C (A + I) A^h = 0 on every frame basis vector (rows restricted to `rows`)."""
        A = A or self.A
        C = C or self.blk["B"].C
        rm = self._rows(rows)
        bad = 0
        for b in range(self.FR):
            v = 1 << b
            for _ in range(h):
                v = A(v)
            if (C(A(v) ^ v)) & rm:
                bad += 1
        return dict(h=h, rows=rows, holds=bad == 0, violating_basis_vectors=bad)

    def terminal(self, x: int, rows: str = "all"):
        """N(x) and N_obs(x) for a frame x entering the LAST body block (position T-1), as small relative
        vectors: the detectors that read a tail / readout record or the final layer:
          delta = -1 checks:  D(k, T-1) = m_{T-1} (in E) + m_{T-2} (in the last B)
          memory type:        D(b, k, T) = d(S_k) + m_{T-1}  (m_{T-1} in E if delta = -1, else in the last B)
          observables:        L_j . d
        Returned as (tuple of bits for the D(.,T-1) rows, tuple for the D(.,T) rows, obs bits)."""
        c = self.code
        B, E, R = self.blk["B"], self.blk["E"], self.blk["R"]
        yB = B.C(x)
        xa = B.A(x)
        yE = E.C(xa)
        z = R.C(E.A(xa))
        posB = {q: i for i, (q, _) in enumerate(B.meas)}
        posE = {q: i for i, (q, _) in enumerate(E.meas)}
        dflip = {q: (z >> i) & 1 for i, (q, _) in enumerate(R.meas)}
        r1, r2 = [], []
        for q in range(c.nd, c.nq):
            ti, k = c.check_of(q)
            if rows == "mem" and ti != c.bi:
                continue
            if self.delta[q] == -1:
                r1.append(((yE >> posE[q]) & 1) ^ ((yB >> posB[q]) & 1))
            if ti == c.bi:
                par = 0
                for qq in c.H[c.bi][k]:
                    par ^= dflip[qq]
                last = (yE >> posE[q]) & 1 if self.delta[q] == -1 else (yB >> posB[q]) & 1
                r2.append(par ^ last)
        ob = 0
        for j, Lj in enumerate(c.L):
            par = 0
            for qq in Lj:
                par ^= dflip[qq]
            ob |= par << j
        return tuple(r1), tuple(r2), ob

    def end(self, e: int, rows: str = "all") -> dict:
        """END(e): N A^e = 0 and N_obs (A + I) A^e = 0 on every frame basis vector."""
        bad_d = bad_o = 0
        for b in range(self.FR):
            v = self.Apow(1 << b, e)
            r1, r2, ob = self.terminal(v, rows)
            if any(r1) or any(r2):
                bad_d += 1
            _, _, ob2 = self.terminal(self.A(v), rows)
            if ob ^ ob2:
                bad_o += 1
        return dict(e=e, rows=rows, holds=bad_d == 0 and bad_o == 0, detector_violations=bad_d,
                    observable_violations=bad_o)

    def minimal(self, fn, rows: str, hmax: int = 8) -> dict:
        """Smallest h in 0..hmax for which fn(h, rows) holds, with the failing h - 1 shown (the identity can
        fail: a check that passes at every h would be dark)."""
        prev = None
        for h in range(hmax + 1):
            r = fn(h, rows)
            if r["holds"]:
                return dict(minimal=h, at_minimal=r, just_below=prev)
            prev = r
        return dict(minimal=None, at_minimal=None, just_below=prev)
