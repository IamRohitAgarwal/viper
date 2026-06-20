"""PIVOT visual prompt — annotate the frame with candidates (SPEC section 8.2).

Delegates all drawing to ``visualization/draw.py`` and packages the result
as an ``AnnotatedProposal``.
"""

from PIL import Image

from src.models import AnnotatedProposal, Trajectory
from src.visualization import draw


def draw_candidates(image: Image.Image, candidates: list[Trajectory]) -> AnnotatedProposal:
    """Label each candidate T1..Tn on the frame; return image + candidates."""
    annotated = draw.draw_candidates(image, candidates)
    return AnnotatedProposal(image=annotated, candidates=candidates)
