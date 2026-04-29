from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GridNode:
    """A spatial node in the low-altitude grid graph."""

    grid_id: str
    height_layer: int

    @property
    def node_id(self) -> str:
        return make_node_id(self.grid_id, self.height_layer)


def make_node_id(grid_id: str, height_layer: int) -> str:
    return f"{grid_id}#H{int(height_layer)}"


def split_node_id(node_id: str) -> GridNode:
    try:
        grid_id, height = node_id.rsplit("#H", 1)
        return GridNode(grid_id=grid_id, height_layer=int(height))
    except ValueError as exc:
        raise ValueError(f"Invalid node_id: {node_id!r}") from exc


def make_cell_key(grid_id: str, height_layer: int, time_slot: str) -> str:
    return f"{make_node_id(grid_id, height_layer)}@{time_slot}"

