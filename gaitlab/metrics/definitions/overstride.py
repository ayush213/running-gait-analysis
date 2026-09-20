"""Overstride — how far ahead of the hip the foot lands at contact, side view.

MEASUREMENT CONTRACT:
- Modeled quantity: Sagittal horizontal (anterior-posterior) displacement from proximal
  landmark (hip) to distal landmark (ankle) at initial contact, normalized by leg length.
- Formula: ((ankle_x - hip_x) * ctx.facing) / ctx.leg * 100.0  [%leg]

DECISIONS & PROVENANCE:
- Proximal landmark [heuristic]: 2D pose `hip` (joint center). Literature (e.g. Lieberman
  et al. 2010, DOI 10.1038/nature08723) marks the greater trochanter. 2D pose models place
  the hip keypoint at the joint center, which is slightly superior and medial to the
  greater trochanter.
- Distal landmark [heuristic]: 2D pose `ankle` (lateral malleolus). Selected as headline
  landmark because it is the most stable foot keypoint across shoe geometry and pitch
  angles. Alternative distal options considered:
    * Option A (headline / active): `ankle`. Stable; does not shift with shoe sole thickness
      or foot-strike angle. Note: true contact point sits ~10-15 cm behind/below at heel strike.
    * Option B (alternative): `foot centre` (heel <-> big-toe midpoint). Closer to colloquial
      meaning and Lieberman's marker set (calcaneus + metatarsal heads), but translates with
      foot pitch, confounding overstride with foot-strike angle.
    * Option C (alternative): `heel`. Actual contact point for rearfoot strikers, but meaningless
      for forefoot strikers (meaning shifts between runners).
- Event anchor [heuristic]: `ctx.ev.strikes[side]` at initial ground contact. Derived from
  the leading edge of the stance plateau in `gaitlab/core/events.py` (LIFT_FRACTION = 0.15).
  Temporal calibration of touchdown is tracked as an uncalibrated heuristic.
- Denominator [heuristic]: `ctx.leg` = median(thigh + shank) in px. Lieberman normalizes by
  trochanter-to-floor height, which includes ankle-to-floor height (~7-10 cm). `ctx.leg` sums
  thigh + shank segments, so `%leg` represents segment-normalized proportion.
  * Note: Hip->ankle inclination angle from vertical (atan2(dx, dy)) removes the denominator
    entirely and can be verified with a protractor on a still frame without pixel calibration.
- Sign convention [convention]: Multiplied by `ctx.facing` (+1 moving image-right, -1 left).
  Positive = foot lands ahead of hip (anterior); negative = foot lands behind hip.
- Units [convention]: `%leg` (percentage of segmented leg length, thigh + shank).
- Aggregation [heuristic]: Per-side median across strides to reject tracking noise. Bilateral
  aggregation is `worst_high` (max of left and right) to surface whichever side has greater
  braking force / injury risk.
- Good / warn bands [heuristic]: `good=(None, 8)`, `warn=(None, 15)`. Literature (Lieberman
  2010, Heiderscheit 2011, DOI 10.1249/MSS.0b013e31820a40b5) demonstrates that landing ahead
  of the center of mass increases braking impulse and tibial shock. The specific cutoffs
  (8% / 15% leg) are project heuristics pending dedicated threshold calibration.

PROVENANCE VOCABULARY:
- [literature]: Direct citation from peer-reviewed biomechanics studies.
- [calibrated]: Validated or tuned against ground-truth video/fixtures.
- [heuristic]: Domain-informed engineering decision or convention.
- [uncalibrated]: Working assumption requiring experimental validation.
"""

from __future__ import annotations

from ..ctx import median
from ..keys import MetricKey
from ..spec import MetricDef, register


def _compute(ctx, side):
    """Compute per-strike overstride (%leg) for one side, aggregated by median."""
    vals = []
    for s in ctx.ev.strikes[side]:
        ankle = ctx.seq.xy(s, f"{side}_ankle")
        hip = ctx.seq.xy(s, f"{side}_hip")
        # Positive = distal landmark ahead of proximal landmark in running direction
        vals.append(((ankle[0] - hip[0]) * ctx.facing) / ctx.leg * 100.0)
    return median(vals)


def _trigger(defn, value, values, targets):
    if value != value:
        return None
    t = targets.get(defn.key, defn)
    st = t.status(value)
    if st == "good":
        return None
    return "high", ("high" if st == "bad" else "med")


register(MetricDef(
    key=MetricKey.OVERSTRIDE,
    label="Overstride",
    unit="%leg",
    good=(None, 8),     # [heuristic] <8% leg considered low braking risk
    warn=(None, 15),    # [heuristic] >15% leg indicates excessive reach / braking
    note="Foot should land close to under your hips. Landing far ahead (>~8% of leg length) brakes you.",
    confidence="high",
    views=("side",),
    scored=True,
    per_side=True,
    asym_direction="higher_worse",
    compute=_compute,
    per_side_compute=True,
    aggregate="worst_high",  # [heuristic] Max of left and right sides
    keypoints=("l_hip", "l_ankle", "r_hip", "r_ankle"),
    anchor_frame="l_strike",
    card_per_side_key="overstride",
    trigger_fn=_trigger,
    finding_text={
        "high": {
            "title": "You're overstriding",
            "detail": (
                "Your foot lands about {value:.0f}% of a leg-length ahead of your hips. Landing "
                "that far out in front creates a braking force on every step and raises impact loading."
            ),
            "cue": "Let your foot land closer to under your hips, and lean slightly from the ankles — not the waist.",
            "drill": "High-cadence strides: 6×20s focusing on quick feet landing beneath you.",
        },
    },
    exercises=[
        {"name": "High-cadence strides",
         "why": "Pulls the foot-strike back under your hips.",
         "dose": "6×20s focusing on landing beneath you",
         "progression": "Blend into tempo running."},
        {"name": "Falling-start runs",
         "why": "Teaches leaning from the ankles, not reaching.",
         "dose": "6×20m from a tall lean",
         "progression": "Carry the lean into a relaxed cruise."},
        {"name": "Wall posture drill",
         "why": "Builds the tall, forward-from-the-ankle position.",
         "dose": "3×30s holds",
         "progression": "Add a marching knee-drive."},
    ],
))
