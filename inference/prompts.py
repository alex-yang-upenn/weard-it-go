"""What the vision-language model is asked, and the JSON shape it must answer in."""

from __future__ import annotations

from datetime import datetime, timezone

SYSTEM_PROMPT = """\
You index frames from a person's smart glasses so they can later ask "where did I leave my X?".
For each frame you get the image, why it was captured, and the objects a detector found (with boxes).
Describe where each object is, relative to nearby furniture, surfaces and landmarks a person would
recognize when they come back ("on the kitchen counter, left of the sink"). The detector only knows a
fixed list and misses things: also include in "objects" any small personal items you can see that
people misplace (keys, wallet, phone, glasses, earbuds, remote, cards, bags, bottles...).
Be concrete and brief. If a detection looks wrong, say what it actually is. Do not invent details
you cannot see. If no personal items are visible, describe the setting and return an empty list."""

# Why the frame was saved (scene "kind", set by live_agent.SightingTracker).
KIND_CONTEXT = {
    "arrived": "These objects just came into view.",
    "left": ("This is the LAST frame these objects were seen in before they left view. "
             "Where they are here is most likely where they were left; be precise about it."),
    "snapshot": "Periodic snapshot of the view, saved whatever the detector found.",
}

# Enforced by vLLM structured outputs, so the reply always parses.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {
            "type": "string",
            "description": "One or two sentences: the setting and where the detected objects are.",
        },
        "place": {
            "type": "string",
            "description": "Short guess at the room or setting, e.g. 'kitchen', 'office desk', 'car'.",
        },
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "where": {"type": "string", "description": "Where it is, relative to its surroundings."},
                },
                "required": ["label", "where"],
            },
        },
    },
    "required": ["caption", "place", "objects"],
}


def _region(box: list[float]) -> str:
    """Normalized [x, y, w, h] -> rough position words, e.g. 'bottom left'."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    col = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
    row = "top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else "middle"
    return "center" if (row, col) == ("middle", "center") else f"{row} {col}"


def build_context(scene: dict) -> str:
    """The text that goes next to the image: detections and when it was taken."""
    lines = [KIND_CONTEXT.get(scene.get("kind", "arrived"), "")]
    if ts_ms := scene.get("ts_ms"):
        taken = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        lines.append(f"Captured: {taken:%Y-%m-%d %H:%M} UTC")
    objects = scene.get("objects") or []
    if objects:
        lines.append("Detected objects:")
        for o in objects:
            lines.append(f"- {o['label']} ({o['confidence']:.0%} confidence, {_region(o['box'])} of frame)")
    else:
        lines.append("Detected objects: none")
    lines.append("Describe where the objects are.")
    return "\n".join(line for line in lines if line)


def search_text(description: dict) -> str:
    """One string to embed and full-text index: caption + place + per-object placement."""
    parts = [description["caption"], f"Place: {description['place']}."]
    parts += [f"{o['label']}: {o['where']}." for o in description["objects"]]
    return " ".join(parts)
