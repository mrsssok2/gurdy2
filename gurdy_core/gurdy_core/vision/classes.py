"""Object classes of the inspection-robot detector (Phase 4): ids, names, Hinglish aliases, size priors."""
from __future__ import annotations

import re

CLASSES = ("box", "barrel", "fire_extinguisher", "traffic_cone", "toolbox", "mannequin", "chair")
CLASS_ID = {n: i for i, n in enumerate(CLASSES)}
# Gazebo label-system id = class id + 1 (label 0 is "background / unlabelled" in the boundingbox camera).
GZ_LABEL_OFFSET = 1

# find_object("laal dabba") style lookup: English + Hinglish words -> class
ALIASES = {
    "box": ("box", "boxes", "carton", "cardboard", "dabba", "dabbe", "dibba", "peti"),
    "barrel": ("barrel", "drum", "pipa", "tanki"),
    "fire_extinguisher": ("extinguisher", "fire", "agni", "aag"),
    "traffic_cone": ("cone", "traffic", "coan"),
    "toolbox": ("toolbox", "tool", "auzaar", "aujar"),
    "mannequin": ("mannequin", "person", "putla", "insaan", "aadmi", "admi", "human", "dummy"),
    "chair": ("chair", "kursi", "stool"),
}
# approximate real size (w, d, h) metres, the radius used to push a front-surface depth point to the object centre
SIZE_PRIOR = {
    "box": (0.40, 0.40, 0.35), "barrel": (0.55, 0.55, 0.90), "fire_extinguisher": (0.16, 0.16, 0.50),
    "traffic_cone": (0.30, 0.30, 0.70), "toolbox": (0.45, 0.20, 0.20), "mannequin": (0.45, 0.25, 1.70),
    "chair": (0.45, 0.45, 0.90),
}
COLORS = {  # RGB 0..1 for RViz markers
    "box": (0.65, 0.45, 0.20), "barrel": (0.10, 0.30, 0.85), "fire_extinguisher": (0.90, 0.10, 0.10),
    "traffic_cone": (1.00, 0.45, 0.00), "toolbox": (0.95, 0.80, 0.10), "mannequin": (0.80, 0.30, 0.80),
    "chair": (0.20, 0.70, 0.45),
}


def radius_prior(name: str) -> float:
    """Half of the smaller horizontal extent: how far behind the visible surface the object centre is."""
    w, d, _ = SIZE_PRIOR[name]
    return 0.5 * min(w, d)


def resolve_class(text: str) -> str | None:
    """'laal dabba dhundo' -> 'box'; 'fire extinguisher' -> 'fire_extinguisher'; unknown -> None."""
    words = re.findall(r"[a-z]+", text.lower())
    hits = [c for c, al in ALIASES.items() if any(w in al for w in words)]
    return hits[0] if len(hits) == 1 else None
