import pytest

from perception.tasks import registry
from perception.tasks.TaskPerceiver import TaskPerceiver


@pytest.fixture
def clean_registry():
    """Registration is a module-level global; isolate each test's writes to it."""
    saved = dict(registry._REGISTRY)
    registry._REGISTRY.clear()
    try:
        yield registry._REGISTRY
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved)


class _DummyPerceiver(TaskPerceiver):
    def analyze(self, frame, debug, slider_vals):
        return None


def test_register_and_get(clean_registry):
    registry.register_perceiver(task="dummy", algo="v1")(_DummyPerceiver)

    assert registry.get_perceiver("dummy", "v1") is _DummyPerceiver


def test_get_missing_raises_key_error(clean_registry):
    with pytest.raises(KeyError):
        registry.get_perceiver("nonexistent", "nonexistent")


def test_duplicate_registration_raises(clean_registry):
    registry.register_perceiver(task="dummy", algo="v1")(_DummyPerceiver)

    class _OtherPerceiver(TaskPerceiver):
        def analyze(self, frame, debug, slider_vals):
            return None

    with pytest.raises(ValueError):
        registry.register_perceiver(task="dummy", algo="v1")(_OtherPerceiver)


def test_reregistering_same_class_is_a_no_op(clean_registry):
    decorator = registry.register_perceiver(task="dummy", algo="v1")
    decorator(_DummyPerceiver)
    decorator(_DummyPerceiver)  # e.g. re-imported module; must not raise

    assert registry.get_perceiver("dummy", "v1") is _DummyPerceiver


def test_list_tasks_and_list_algos(clean_registry):
    registry.register_perceiver(task="dummy", algo="v1")(_DummyPerceiver)
    registry.register_perceiver(task="dummy", algo="v2")(_DummyPerceiver)
    registry.register_perceiver(task="other", algo="v1")(_DummyPerceiver)

    assert registry.list_tasks() == ["dummy", "other"]
    assert registry.list_algos("dummy") == ["v1", "v2"]
    assert registry.list_algos("nonexistent") == []


def test_discover_all_finds_real_perceivers():
    """Integration check: discovery over the real perception.tasks package finds
    algorithms that were ported/registered as part of this consolidation, with
    no hand-maintained import list required."""
    registry.discover_all()

    assert "slalom" in registry.list_tasks()
    assert "classical" in registry.list_algos("slalom")

    from perception.tasks.slalom.classical.perceiver import SlalomClassicalPerceiver

    assert registry.get_perceiver("slalom", "classical") is SlalomClassicalPerceiver

    assert "gate" in registry.list_tasks()
    assert set(registry.list_algos("gate")) >= {
        "center",
        "segmentation_a",
        "segmentation_b",
        "segmentation_c",
    }
