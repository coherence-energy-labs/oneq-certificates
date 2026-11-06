# n-use and all-to-all projections (R147)

Files written by `experiments/superactivation/shield_family/hw_nuse.py`.

The four `ion_helios_typ_*` files were produced before `predict_ion` took a `--device` argument, under the names
`ion_n*_*.json`, and were renamed to the names the current code writes. Their CONTENTS are as produced: their
`device` field reads `"all-to-all, Helios data-sheet typical"` and their parameters are under `helios`, where a
re-run today writes `device: "helios_typ"` and `device_params`. The numbers are the same model (the Helios data
sheet v1.00 typical values, `hw_levers.HELIOS` = `hw_nuse.DEVICES["helios_typ"]`).
