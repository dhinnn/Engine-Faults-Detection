"""Static advice mapping for detected engine faults.

This module provides a simple, deterministic mapping from fault labels to
actionable advice text (Markdown). It's intentionally small and offline so the
app can provide immediate, reliable guidance without external APIs.
"""

ADVICE = {
    "Misfire": (
        "### Misfire\n"
        "**What it means:** One or more cylinders are not firing consistently.\n\n"
        "**Immediate actions:** Stop driving at high speed; avoid hard acceleration; check for rough idle or engine warning lights.\n\n"
        "**Typical causes:** Faulty spark plugs or ignition coils, clogged fuel injector, vacuum leak, or low compression.\n\n"
        "**Next steps:** Inspect spark plugs and coils, run an OBD-II scan for P030x codes, check fuel delivery and compression; consult a mechanic if the issue persists."
    ),

    "Normal": (
        "### Normal\n"
        "**What it means:** No notable fault detected.\n\n"
        "**Immediate actions:** None required — continue normal operation.\n\n"
        "**Next steps:** Maintain routine maintenance and keep recordings for future comparison."
    ),

    "Rodknock": (
        "### Rodknock\n"
        "**What it means:** A deep knocking from the engine bottom end indicating possible bearing wear.\n\n"
        "**Immediate actions:** Minimize engine load; avoid high RPMs; get the vehicle to a mechanic promptly.\n\n"
        "**Typical causes:** Worn rod bearings, low oil pressure, or severe engine wear.\n\n"
        "**Next steps:** Check oil level and pressure; avoid long trips; have a mechanic inspect engine internals."
    ),

    "Timing Chain": (
        "### Timing Chain\n"
        "**What it means:** Problems with timing chain tension or alignment.\n\n"
        "**Immediate actions:** Avoid high RPM and heavy loads.\n\n"
        "**Typical causes:** Worn chain/tensioner, stretched chain, or failed guides.\n\n"
        "**Next steps:** Inspect timing chain components; replacement of chain/tensioner/guides is commonly required if confirmed."
    ),

    "Clicking": (
        "### Clicking\n"
        "**What it means:** A light clicking noise which can originate from injectors, valvetrain, or accessories.\n\n"
        "**Immediate actions:** Note when it occurs (idle, acceleration) and check oil level.\n\n"
        "**Typical causes:** Injector noise, valve lifter/tappet issues, loose accessory belt or pulley.\n\n"
        "**Next steps:** Check oil and belts; if persistent, run compression/leakdown tests or consult a mechanic."
    ),

    "Knocking": (
        "### Knocking\n"
        "**What it means:** Severe detonation-like knocking indicating combustion issues.\n\n"
        "**Immediate actions:** Reduce load immediately; avoid high RPMs.\n\n"
        "**Typical causes:** Pre-ignition/detonation, incorrect timing, poor fuel quality, or carbon buildup.\n\n"
        "**Next steps:** Check ignition timing and engine management codes; use recommended fuel; consult a professional if knocking persists."
    ),
}


def get_advice(label: str) -> str:
    """Return advice Markdown for the given label.

    If the label is unknown, returns a generic guidance message.
    """
    if not label:
        return "No label provided. Please get a prediction first."

    return ADVICE.get(label, f"### No advice available for '{label}'.\nPlease consult a mechanic or provide a valid fault label.")


def available_labels():
    """Return a sorted list of supported labels."""
    return sorted(ADVICE.keys())
