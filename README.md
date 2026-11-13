# Preregistration anchors

This branch only ever grows. Each file under `anchors/` binds the SHA-256 of a committed preregistration to a future
round of the public drand beacon (mainnet quicknet), and carries an OpenTimestamps proof of when the anchor existed.

An anchor is valid only if its proof is attested in a Bitcoin block whose timestamp is at least `margin_hours`
before the declared round was published. The run's random seeds are then derived from that round's randomness and
the preregistration's digest, and from nothing else. So the run's analysis plan provably predates both its seeds and
its data.

To verify one: check the `.ots` proof with any OpenTimestamps client against Bitcoin, read the round's randomness
from a drand relay, and compare the preregistration's SHA-256 (of the bytes git stores) with `prereg_sha256` once
the preregistration is published with its result.
