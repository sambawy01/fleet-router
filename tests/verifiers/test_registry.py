from fleet.verifiers.heuristic import HeuristicVerifier
from fleet.verifiers.math import MathVerifier
from fleet.verifiers.registry import VerifierRegistry


def test_explicit_registration_wins():
    reg = VerifierRegistry()
    v = MathVerifier()
    reg.register(v)
    assert reg.for_tag("math") is v


def test_default_returned_when_no_explicit_registration():
    default = HeuristicVerifier(tag="general")
    reg = VerifierRegistry(default=default)
    assert reg.for_tag("creative") is default


def test_falls_back_to_heuristic_when_no_default():
    reg = VerifierRegistry()
    v = reg.for_tag("creative")
    assert isinstance(v, HeuristicVerifier)
    assert v.tag == "creative"


def test_has_and_tags():
    reg = VerifierRegistry()
    reg.register(MathVerifier())
    assert reg.has("math")
    assert not reg.has("code")
    assert reg.tags() == ["math"]
