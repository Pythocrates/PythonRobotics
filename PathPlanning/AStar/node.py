''' This module contains a simple node class for path planning.
'''


class Node:
    def __init__(self, grid_position, cost=0, previous=None, parent=None):
        self._grid_position = grid_position
        self._cost = cost
        self._previous = previous
        self._parent = parent

    def update(self, cost, previous):
        self._cost = cost
        self._previous = previous

    @property
    def cost(self):
        return self._cost

    @property
    def previous(self):
        return self._previous

    @property
    def grid_position(self):
        return self._grid_position

    @property
    def world_position(self):
        return self._parent.obstacle_map.grid_to_world(self.grid_position)

    @property
    def is_ok(self):
        return self.grid_position in self._parent.obstacle_map

    def __str__(self):
        return f'{self.grid_position}, {self._cost}, {self._previous}'
