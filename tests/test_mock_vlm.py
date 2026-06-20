"""Phase 2 — MockVLM determinism + interface conformance (SPEC section 13 Phase 2)."""

from PIL import Image

from src.models import (
    DebateArtifact,
    SceneUnderstanding,
    SelectedCandidates,
    Trajectory,
    Point,
)
from src.vlm.interface import VLMInterface
from src.vlm.mock_vlm import MockVLM, parse_verdict


def _img():
    return Image.new("RGB", (64, 64), (0, 0, 0))


def _candidates(n=5):
    # endpoints fan out across the image so distances differ
    return [
        Trajectory(id=i + 1, points=[Point(0.1, 0.1), Point(0.1 + 0.15 * i, 0.5)])
        for i in range(n)
    ]


def test_is_vlm_interface():
    assert isinstance(MockVLM(), VLMInterface)


def test_understand_scene_deterministic():
    a = MockVLM(seed=42).understand_scene(_img(), "from A to B")
    b = MockVLM(seed=42).understand_scene(_img(), "from A to B")
    assert a == b
    assert isinstance(a, SceneUnderstanding)
    assert 2 <= len(a.objects) <= 5
    assert any(o.role == "target" for o in a.objects)
    assert any(o.role == "goal" for o in a.objects)


def test_understand_scene_seed_changes_output():
    a = MockVLM(seed=42).understand_scene(_img(), "from A to B")
    b = MockVLM(seed=7).understand_scene(_img(), "from A to B")
    assert a != b


def test_obstacle_added_on_keyword():
    scene = MockVLM(seed=1).understand_scene(_img(), "from A to B avoiding the pallet")
    assert any(o.role == "obstacle" for o in scene.objects)


def test_select_candidates_returns_top_k():
    from config import SELECT_TOP_K

    scene = MockVLM(seed=42).understand_scene(_img(), "from A to B")
    sel = MockVLM(seed=42).select_candidates(_img(), "from A to B", _candidates(5), scene)
    assert isinstance(sel, SelectedCandidates)
    assert 1 <= len(sel.ids) <= SELECT_TOP_K
    assert all(isinstance(i, int) for i in sel.ids)


def test_select_candidates_deterministic():
    scene = MockVLM(seed=42).understand_scene(_img(), "g")
    s1 = MockVLM(seed=42).select_candidates(_img(), "g", _candidates(), scene)
    s2 = MockVLM(seed=42).select_candidates(_img(), "g", _candidates(), scene)
    assert s1 == s2


def test_verdict_from_seed():
    # seed % 3 -> endorse / amend / reject
    assert MockVLM(seed=42).critique_plan(_artifact(), None).verdict == "endorse"  # 42%3=0
    assert MockVLM(seed=43).critique_plan(_artifact(), None).verdict == "amend"    # 43%3=1
    assert MockVLM(seed=44).critique_plan(_artifact(), None).verdict == "reject"   # 44%3=2


def test_critique_reply_is_parseable():
    turn = MockVLM(seed=42, name="claude").critique_plan(_artifact(), None)
    assert turn.model == "claude"
    assert parse_verdict(turn.raw_reply) == turn.verdict


def test_concession_to_more_conservative():
    # endorser (seed 42) that concedes should move to 'reject' when shown a reject.
    conceder = MockVLM(seed=42, concede=True)
    other = "I think this fails.\nVERDICT: reject"
    turn = conceder.critique_plan(_artifact(), other)
    assert turn.verdict == "reject"


def test_no_concession_when_disabled():
    stubborn = MockVLM(seed=42, concede=False)
    other = "VERDICT: reject"
    assert stubborn.critique_plan(_artifact(), other).verdict == "endorse"


def test_no_concession_to_less_conservative():
    # a rejecter never downgrades to endorse, even if it concedes.
    conceder = MockVLM(seed=44, concede=True)  # base reject
    other = "VERDICT: endorse"
    assert conceder.critique_plan(_artifact(), other).verdict == "reject"


def _artifact():
    return DebateArtifact(
        selected_image=_img(),
        goal="from A to B",
        trajectory_summary="A->B direct",
        cost_summary="low cost",
        candidate_summary="others longer",
    )
