# Anchored preregistrations

How this project's preregistrations are fixed in time from 2026-11-07 on, what that proves, and what it does not.
The tool is `tools/prereg_anchor.py`; its two pure parts are `src/oneq/prereg_seeds.py` (`derive_prereg_seeds`,
`round_time`) and `src/oneq/ots_proof.py` (an OpenTimestamps proof executed to each attested block's merkle root).

## The problem it answers

The papers' earlier preregistrations were committed to the project's development repository before their runs.
That repository is private, and a git commit's date is whatever its author's clock said, so a reader has only the
author's word for the order of a preregistration and its run (external audit 4, X3). Publishing the history would
not change that: dates in a public history are still set by the author. For simulated data, or data already public,
no timestamp on the plan alone can rule out "ran first, wrote the plan after".

## The protocol

1. **Write and commit the preregistration** (`docs/PREREG_<name>.md`): hypotheses, endpoints, the analysis, the
   stopping rule, and how the run turns seeds into data. It must not contain the seeds.
2. **Declare** (`python tools/prereg_anchor.py declare docs/PREREG_<name>.md`). The tool refuses a preregistration
   with uncommitted changes, then writes `docs/prereg_anchors/PREREG_<name>.anchor.json`: the preregistration's
   sha256 (of the bytes git stores) and a round of the public drand beacon (mainnet "quicknet", one round every 3
   seconds, each round's value unpredictable before it is published) about 48 hours ahead. It stamps the anchor
   with OpenTimestamps, which the public calendars commit to Bitcoin within hours.
3. **Commit and publish the anchor and its proof** before anything else happens: commit both, and push them to the
   `preregistrations` branch of the public repository, coherence-energy-labs/oneq-certificates, which only ever
   grows; the next release bundle carries them too.
4. **Wait** for both clocks: the proof's Bitcoin attestation, and the declared round.
5. **Redeem** (`python tools/prereg_anchor.py redeem docs/prereg_anchors/PREREG_<name>.anchor.json`). The tool
   upgrades a copy of the proof, checks every link below, and writes `PREREG_<name>.redeemed.json` with the round's
   randomness. The run's seeds are `oneq.prereg_seeds.derive_prereg_seeds(randomness, prereg_sha256=..., n=...)`
   (`... seeds ... --n N` prints them), and the run uses nothing else.
6. **Run**, then report against the preregistration. Deviations are recorded in a separate file, as before.

## What a reader checks

`python tools/prereg_anchor.py verify docs/prereg_anchors/PREREG_<name>.anchor.json`, which re-derives from public
sources, and refuses on any break:

- the preregistration in the tree has the sha256 the anchor names;
- the anchor's OpenTimestamps proof covers the anchor's bytes, and executing its operations reaches the merkle root
  of the Bitcoin block it names, as two block explorers report that block (`blockstream.info`, `mempool.space`);
- that block's timestamp is at least `margin_hours` (6 by default) before the declared round's publication time,
  which drand's specification fixes as genesis time plus (round - 1) times the period;
- two drand relays (`api.drand.sh`, `drand.cloudflare.com`) give the same randomness for that round;
- the seeds follow from that randomness and the preregistration's bytes, and the redemption record says the same.

Together: the preregistration existed before the block; the block came before the round; the seeds did not exist
before the round. Data that depends on the seeds was therefore drawn after the preregistration was fixed, by clocks
nobody involved controls.

## The margin

A Bitcoin block's timestamp is set by its miner. Consensus only requires it to exceed the median of the previous
eleven blocks' times, and nodes reject a block more than two hours ahead of their clocks, so a timestamp can sit
about two hours from real time in either direction. Six hours of margin covers that with room, and the 48-hour lead
leaves time for the attestation, which usually arrives within a few hours. If the attestation comes too late, the
anchor is void and says so: declare again with a longer lead. Nothing is adjusted after the fact.

## What it does not prove

- **That no other anchor exists.** Someone could stamp several preregistrations privately and reveal the one whose
  seeds suited them. Publishing every anchor before its run (step 3) is the defence; the release bundles carry every
  anchor ever declared, including abandoned ones, so a reader can see them all.
- **That the run used the seeds for everything that matters.** The preregistration must say how the seeds turn
  into data; a run with a second, undeclared source of randomness escapes the protocol.
- **That drand or the explorers are honest.** Two independent sources must agree, and a reader can consult others
  (any drand relay; a Bitcoin node with `ots verify`). drand's BLS signature on the round is recorded and not
  checked by this tool.
- **Anything about preregistrations made before 2026-11-07.** Those rest on the author's record, as the papers say.

## Earlier mechanism this reuses

`src/oneq/beacon.py` was built in August 2026 for the hardware readiness program's blind injection schedule (declare
a future drand round, derive the schedule from it); `tools/fetch_beacon.py` is its fetch step. The preregistration
protocol adds the OpenTimestamps link that puts the plan before the round, and the seed derivation bound to the
plan's bytes.
