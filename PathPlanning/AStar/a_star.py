"""

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""

from collections import namedtuple
import math
import sys

import matplotlib.pyplot as plt


SHOW_ANIMATION = True


Position = namedtuple('Position', 'x y')


class PPVisualizer:
    def __init__(self, start, goal, obstacles, *args, **kwargs):
        super().__init__(*args, **kwargs)
        plt.plot(*zip(*obstacles), ".k")
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
    def on_final_path(path):
        plt.plot(*zip(*path), "-r")
        plt.pause(0.001)
        plt.show()


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


class AStarPlanner:
    def __init__(self, obstacles, resolution, robot_radius):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        robot_radius: robot radius[m]
        """

        self.obstacle_map = ObstacleMap(obstacles, resolution, robot_radius)
        self.motion = self.get_motion_model()
        self._handlers = []
        self._all_nodes = dict()

    def _create_node(self, *args, **kwargs):
        return Node(*args, parent=self, **kwargs)

    def add_handler(self, handler):
        self._handlers.append(handler)

    def node_at(self, world_position):
        grid_position = self.obstacle_map.world_to_grid(world_position)
        return self.node_at_grid(grid_position)

    def node_at_grid(self, grid_position):
        try:
            node = self._all_nodes[grid_position]
        except KeyError:
            node = self._create_node(grid_position)
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
                handler.on_position_update(current.world_position)

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
                    node.update(cost=new_cost, previous=current)
                elif node.cost > new_cost:
                    # This path is the best until now. record it
                    node.update(cost=new_cost, previous=current)

        path = self.calc_final_path(goal_node)

        for handler in self._handlers:
            handler.on_final_path(path)

        return path

    @staticmethod
    def calc_final_path(goal_node):
        # generate final course
        result = list()
        node = goal_node
        while True:
            result.append(node.world_position)
            node = node.previous
            if not node:
                return result


    @staticmethod
    def calc_heuristic(node_1, node_2):
        weight = 1.0  # weight of heuristic
        pos_1 = node_1.grid_position
        pos_2 = node_2.grid_position
        return weight * math.hypot(pos_1.x - pos_2.x, pos_1.y - pos_2.y)

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


def main():
    print(__file__ + " start!!")

    start_position = Position(10.0, 10.0)  # world coordinates in m
    goal_position = Position(50.0, 50.0)  # world coordinates im m
    grid_size = 2.0  # [m]
    robot_radius = 1.0  # [m]

    # set obstacle positions
    obstacles = {
        *((i, -10.0) for i in range(-10, 60)),
        *((60.0, i) for i in range(-10, 60)),
        *((i, 60.0) for i in range(-10, 61)),
        *((-10.0, i) for i in range(-10, 61)),
        *((20.0, i) for i in range(-10, 40)),
        *((40.0, 60.0 - i) for i in range(0, 40)),
    }

    a_star = AStarPlanner(obstacles, grid_size, robot_radius)
    if SHOW_ANIMATION:  # pragma: no cover
        a_star.add_handler(
            PPVisualizer(start_position, goal_position, obstacles))

    print('Starting planning')
    a_star.planning(start_position, goal_position)


if __name__ == '__main__':
    main()
