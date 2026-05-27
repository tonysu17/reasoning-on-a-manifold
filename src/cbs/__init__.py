"""
src/cbs/ — Content-Beyond-Source (CBS) knowledge-creation extension.

Subpackage scaffolded under P0.3 of the synthesis plan
(`empirical_plan_synthesis.md`). Subsequent milestones fill in the modules:

    schemas.py    : Data classes / TypedDicts (P0.3, locked)
    annotation.py : CBS-tier + cross-domain annotator                       (M1)
    geometry.py   : Per-sentence geometric tests                            (M2)
    trajectory.py : Chain-as-curve trajectory module                        (M3)
    matching.py   : Jaccard matched-pair construction                       (M4)
    ablation.py   : CBS steering ablation                                   (M5)
    comparison.py : Cross-model comparison                                  (M6)
"""

__all__ = [
    "schemas",
    "annotation",
    "geometry",
    "trajectory",
    "matching",
    "ablation",
    "comparison",
]
