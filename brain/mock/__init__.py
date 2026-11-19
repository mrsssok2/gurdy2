from .backend import MockBackend
from .world import Rect, World, flat_lab

__all__ = ["MockBackend", "World", "Rect", "flat_lab", "gazebo_flat_lab"]


def gazebo_flat_lab() -> World:
    """The mock world generated from the real Gazebo flat_lab sources (see lab_world.py)."""
    from .lab_world import gazebo_flat_lab as build
    return build()
