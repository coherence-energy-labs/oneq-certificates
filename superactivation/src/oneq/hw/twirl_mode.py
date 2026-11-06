"""ONE knob for twirling, used by BOTH the submission and the prediction (the device model v2, 2026-10-17).

What the server plays depends on how a job is submitted, and R166/R173 MEASURED it from the executed schedule:
  server-active-accum  the runtime's default. Every single-qubit run is played as a two-sx template, every CZ layer
                       takes 132 ns, and idle touched qubits get a template slot per layer.
                       -> NoiseParams(twirl_template=True, twirl_template_dress=True): the dressing AS PULSES,
                       pulse for pulse the server's schedule on R166's ladders. Held out on R152 with nothing fitted
                       it misses R146 by +0.003 and R147 by +0.001 (evidence/hw/model_regression.json). The earlier
                       first-order form (the dressing as error ops at factor 2: -0.006 / -0.028) is still recognized
                       as this mode in old records.
  server-active        the template and the layer slot, no idle dressing -> NoiseParams(twirl_template=True).
  client               server twirling OFF, our own Pauli frames merged into the compiled gates
                       (oneq.hw.client_twirl): the compiled pulses, plus an x on some empty slots, and a Pauli-twirled
                       CZ -> NoiseParams() with the engine's Pauli CZ channel, which the twirl makes exact on average.
  off                  server twirling OFF, no frames: the compiled gates as compiled (R173). The CZ's coherent
                       error is NOT randomized, so the engine's Pauli CZ channel is an APPROXIMATION
                       (prediction_is_exact is False).
A prediction made for one mode and a job submitted in another predicts a circuit that never runs -- R152's twins
were one circuit twice for the same kind of reason. preflight_refusals() makes that a refusal, not a footnote."""
from __future__ import annotations

MODES = ("server-active-accum", "server-active", "client", "off")


class ModeRefused(ValueError):
    """the twirling mode is unknown, or the prediction and the submission disagree on it"""


def noise_kwargs(mode: str) -> dict:
    """the NoiseParams fields that predict what `mode` PLAYS (R166, R173)"""
    if mode == "server-active-accum":
        return dict(twirl_template=True, twirl_template_dress=True)
    if mode == "server-active":
        return dict(twirl_template=True)
    if mode in ("client", "off"):
        return {}
    raise ModeRefused(f"unknown twirl mode {mode!r}; expected one of {MODES}")


def prediction_is_exact(mode: str) -> bool:
    """whether the engine's Pauli CZ channel is the average channel the device runs in this mode"""
    noise_kwargs(mode)
    return mode != "off"


def plan_fields(mode: str, randomizations: int = 8, client_instances: int = 8) -> dict:
    """the oneq.hw.submit.Plan fields that submit `mode`"""
    if mode == "server-active-accum":
        return dict(twirl_randomizations=randomizations, twirl_strategy="active-accum", client_twirl_instances=None)
    if mode == "server-active":
        return dict(twirl_randomizations=randomizations, twirl_strategy="active", client_twirl_instances=None)
    if mode == "client":
        if not (isinstance(client_instances, int) and client_instances >= 1):
            raise ModeRefused(f"client twirling needs a positive instance count: {client_instances!r}")
        return dict(twirl_randomizations=None, twirl_strategy=None, client_twirl_instances=client_instances)
    if mode == "off":
        return dict(twirl_randomizations=None, twirl_strategy=None, client_twirl_instances=None)
    raise ModeRefused(f"unknown twirl mode {mode!r}; expected one of {MODES}")


def mode_of_plan(plan) -> str:
    """the mode a Plan submits (the runtime's default strategy is active-accum)"""
    if plan.twirl_randomizations:
        if getattr(plan, "client_twirl_instances", None):
            raise ModeRefused("a plan cannot twirl on the server AND client-side")
        return "server-active" if plan.twirl_strategy == "active" else "server-active-accum"
    if plan.twirl_strategy is not None:
        raise ModeRefused(f"strategy {plan.twirl_strategy!r} with server twirling off")
    return "client" if getattr(plan, "client_twirl_instances", None) else "off"


def mode_of_noise(described: dict) -> str:
    """the mode a prediction's NoiseParams.describe() was made for (the inverse of noise_kwargs)"""
    tmpl = bool(described.get("twirl_template", False))
    pulses = bool(described.get("twirl_template_dress", False))
    twirl = described.get("twirl", "off")
    dress = described.get("dressing_factor", 1.0)
    if tmpl and pulses and twirl == "off":
        return "server-active-accum"                   # v2: the dressing as pulses
    if tmpl and not pulses and twirl == "active-accum" and dress == 2.0:
        return "server-active-accum"                   # v1: the dressing as error ops (older records)
    if tmpl and not pulses and twirl == "off":
        return "server-active"
    if not tmpl and twirl == "off":
        return "compiled"                              # "client" or "off": the same engine model
    raise ModeRefused(f"the prediction's noise {described} is none of the measured modes")


def preflight_refusals(plan, frozen: dict) -> list:
    """refusals when the frozen record DECLARES the model its prediction used (frozen['prediction_noise'], a
    NoiseParams.describe()) and the plan submits a different mode; a record that declares none is not checked here"""
    pn = frozen.get("prediction_noise")
    if pn is None:
        return []
    try:
        submitted = mode_of_plan(plan)
        predicted = mode_of_noise(pn)
    except ModeRefused as e:
        return [f"TWIRL MODE: {e}"]
    if predicted == "compiled" and submitted in ("client", "off"):
        bad = []
        if submitted == "client" and not frozen.get("client_twirl"):
            bad.append("TWIRL MODE: a client-twirled plan needs a frozen record built by client_twirl.expand")
        return bad
    if predicted != submitted:
        return [f"TWIRL MODE: the frozen prediction models {predicted!r} but the plan submits {submitted!r}: it would "
                "predict a circuit the server never plays"]
    return []
