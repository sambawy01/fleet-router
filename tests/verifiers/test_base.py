from fleet.verifiers.base import Candidate, VerificationResult


def test_candidate_with_score_returns_new_instance():
    c = Candidate(model="m", sample_idx=0, text="hi")
    c2 = c.with_score(0.8, "great")
    assert c2 is not c
    assert c2.score == 0.8
    assert c2.notes == "great"
    # Original untouched
    assert c.score == 0.0


def test_candidate_with_score_keeps_existing_notes_when_blank():
    c = Candidate(model="m", sample_idx=0, text="hi", notes="prior")
    c2 = c.with_score(0.5)
    assert c2.notes == "prior"


def test_verification_result_winner_text():
    c = Candidate(model="m", sample_idx=0, text="answer")
    r = VerificationResult(winner=c, all_scored=[c])
    assert r.winner_text == "answer"

    r2 = VerificationResult(winner=None, all_scored=[], abstain=True)
    assert r2.winner_text is None
