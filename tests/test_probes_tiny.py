from qwenjev.probes import ALL_PROBES, run_all
from qwenjev.testing import build_tiny_engine


def test_probe_names_cover_the_essay_sections():
    assert set(ALL_PROBES) == {
        "visibility",
        "reference_card",
        "option_interaction",
        "option_order",
        "fake_option",
        "accounting",
        "latency",
    }


def test_quick_probes_run_end_to_end():
    engine = build_tiny_engine()
    results = run_all(engine, quick=True)
    names = {r.name for r in results}
    assert {"visibility", "accounting"} <= names
    for result in results:
        assert result.summary


def test_accounting_probe_is_additive():
    engine = build_tiny_engine()
    result = ALL_PROBES["accounting"](engine)
    assert result.summary["additive"] is True
    assert result.summary["billing_delta"] > 0


def test_fake_option_probe_keeps_the_slot_count():
    engine = build_tiny_engine()
    result = ALL_PROBES["fake_option"](engine)
    assert result.summary["slots"] == {"base": 3, "injected": 3}
