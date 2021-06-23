''' This module contains a simple ostacle map class for path planning.
'''

from collections import namedtuple

import math


Position = namedtuple('Position', 'x y')


class ObstacleMap:
    def __init__(self, obstacles, resolution, robot_radius):
        min_x = round(min(ox for ox, _ in obstacles))
        min_y = round(min(oy for _, oy in obstacles))
        max_x = round(max(ox for ox, _ in obstacles))
        max_y = round(max(oy for _, oy in obstacles))
        self._left_bottom = Position(min_x, min_y)
        self._right_top = Position(max_x, max_y)
        self._resolution = resolution
        self._grid = self._calc_obstacle_map(obstacles, robot_radius)

    def grid_to_world(self, grid_position):
        return Position(
            grid_position.x * self._resolution + self._left_bottom.x,
            grid_position.y * self._resolution + self._left_bottom.y
        )

    def world_to_grid(self, world_position):
        return Position(
            round((world_position.x - self._left_bottom.x) / self._resolution),
            round((world_position.y - self._left_bottom.y) / self._resolution)
        )

    def __contains__(self, grid_position):
        world_position = self.grid_to_world(grid_position)
        return (
            self._left_bottom.x <= world_position.x < self._right_top.x and
            self._left_bottom.y <= world_position.y < self._right_top.y and
            not self._grid[grid_position.x][grid_position.y]  # collision check
        )

    def _calc_obstacle_map(self, obstacles, robot_radius):
        width, height = self.world_to_grid(self._right_top)

        # obstacle map generation
        grid = [[False for _ in range(height)] for _ in range(width)]
        for x_g in range(width):
            for y_g in range(height):
                x_w, y_w = self.grid_to_world(Position(x_g, y_g))
                for x_og, y_og in obstacles:
                    distance = math.hypot(x_og - x_w, y_og - y_w)
                    if distance <= robot_radius:
                        grid[x_g][y_g] = True
                        break

        return grid
