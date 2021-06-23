"""

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""

from collections import namedtuple
from functools import partial
import math
import sys

import matplotlib.pyplot as plt


show_animation = True


Position = namedtuple('Position', 'x y')


class PPVisualizer:
    def __init__(self, start, goal, obstacles, *args, **kwargs):
        super().__init__(*args, **kwargs)
        plt.plot(*obstacles, ".k")
        plt.plot(*start, "og")
        plt.plot(*goal, "xb")
        plt.grid(True)
        plt.axis("equal")
        self._counter = 0

    def on_position_update(self, position):
        plt.plot(*position, "xc")
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [sys.exit(0) if event.key == 'escape' else None]
        )
        self._counter = (self._counter + 1) % 10
        if not self._counter:
            plt.pause(0.0001)

    @staticmethod
    def on_final_path(*path):
        plt.plot(*path, "-r")
        plt.pause(0.001)
        plt.show()


class Node:
    def __init__(self, grid_position, cost=0, previous=None, parent=None):
        self._grid_position = grid_position
        self.cost = cost
        self.previous = previous
        self._parent = parent

    @property
    def grid_position(self):
        return self._grid_position

    def __str__(self):
        return f'{self.grid_position}, {self.cost}, {self.previous}'

    @property
    def position(self):
        return self._parent.obstacle_map.grid2world(self.grid_position)

    @property
    def is_ok(self):
        return self.grid_position in self._parent.obstacle_map


class AStarPlanner:
    def __init__(self, ox, oy, resolution, robot_radius):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        robot_radius: robot radius[m]
        """

        self.obstacle_map = ObstacleMap(ox, oy, resolution, robot_radius)
        self.motion = self.get_motion_model()
        self._handlers = []

        self.Node = partial(Node, parent=self)
        self._all_nodes = dict()

    def add_handler(self, handler):
        self._handlers.append(handler)

    def node_at(self, world_position):
        grid_position = self.obstacle_map.world2grid(world_position)
        return self.node_at_grid(grid_position)

    def node_at_grid(self, grid_position):
        try:
            node = self._all_nodes[grid_position]
        except KeyError:
            node = self.Node(grid_position)
            self._all_nodes[grid_position] = node

        return node

    def planning(self, start_position, goal_position):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.node_at(start_position)
        goal_node = self.node_at(goal_position)

        open_set = {start_node}
        closed_set = set()

        while open_set:
            current = min(
                open_set,
                key=lambda o: o.cost + self.calc_heuristic(goal_node, o)
            )
            # Remove the item from the open set, and add it to the closed set
            open_set.remove(current)
            closed_set.add(current)

            # show graph
            for handler in self._handlers:
                handler.on_position_update(current.position)

            if current is goal_node:
                print("Goal found")
                break

            # expand_grid search grid based on motion model
            for motion in self.motion:
                new_cost = current.cost + motion[2]
                node = self.node_at_grid(
                    Position(
                        current.grid_position.x + motion[0],
                        current.grid_position.y + motion[1],
                    )
                )

                # If the node is not safe, do nothing
                if not node.is_ok:
                    continue

                if node in closed_set:
                    continue

                if node not in open_set:
                    open_set.add(node)  # discovered a new node
                    node.cost = new_cost
                    node.previous = current
                elif node.cost > new_cost:
                    # This path is the best until now. record it
                    node.cost = new_cost
                    node.previous = current

        path = self.calc_final_path(goal_node)

        for handler in self._handlers:
            handler.on_final_path(*path)

        return path

    def calc_final_path(self, goal_node):
        print(f'Calculating: {goal_node}')
        # generate final course
        rx, ry = list(), list()
        node = goal_node
        while True:
            rx.append(node.position[0])
            ry.append(node.position[1])
            node = node.previous
            if not node:
                break

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        pos_1 = n1.grid_position
        pos_2 = n2.grid_position
        d = w * math.hypot(pos_1.x - pos_2.x, pos_1.y - pos_2.y)
        return d

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion


class ObstacleMap:
    def __init__(self, ox, oy, resolution, robot_radius):
        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        self.left_bottom = Position(self.min_x, self.min_y)
        self.right_top = Position(self.max_x, self.max_y)
        self.resolution = resolution
        self._grid = self._calc_obstacle_map(ox, oy, robot_radius)

    def grid2world(self, grid_position):
        wx = grid_position.x * self.resolution + self.min_x
        wy = grid_position.y * self.resolution + self.min_y
        return Position(wx, wy)

    def world2grid(self, world_position):
        gx = round((world_position.x - self.min_x) / self.resolution)
        gy = round((world_position.y - self.min_y) / self.resolution)
        return Position(gx, gy)

    def __contains__(self, grid_position):
        px, py = self.grid2world(grid_position)
        if not self.min_x <= px < self.max_x:
            return False
        elif not self.min_y <= py < self.max_y:
            return False
        elif self._grid[grid_position.x][grid_position.y]:
            # collision check
            return False

        return True

    def _calc_obstacle_map(self, ox, oy, robot_radius):
        width, height = self.world2grid(self.right_top)

        # obstacle map generation
        grid = [[False for _ in range(height)] for _ in range(width)]
        for x_g in range(width):
            for y_g in range(height):
                x_w, y_w = self.grid2world(Position(x_g, y_g))
                for x_og, y_og in zip(ox, oy):
                    distance = math.hypot(x_og - x_w, y_og - y_w)
                    if distance <= robot_radius:
                        grid[x_g][y_g] = True
                        break

        return grid


def main():
    print(__file__ + " start!!")

    start_position = Position(10.0, 10.0)  # world coordinates in m
    goal_position = Position(50.0, 50.0)  # world coordinates im m
    grid_size = 2.0  # [m]
    robot_radius = 1.0  # [m]

    # set obstacle positions
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10.0)
    for i in range(-10, 60):
        ox.append(60.0)
        oy.append(i)
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60.0)
    for i in range(-10, 61):
        ox.append(-10.0)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20.0)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40.0)
        oy.append(60.0 - i)

    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    if show_animation:  # pragma: no cover
        a_star.add_handler(
            PPVisualizer(start=start_position, goal=goal_position, obstacles=(ox, oy)))

    print('Starting planning')
    a_star.planning(start_position, goal_position)


if __name__ == '__main__':
    main()
