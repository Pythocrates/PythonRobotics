"""

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""

from functools import partial
import math
import sys

import matplotlib.pyplot as plt

show_animation = True


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
    def __init__(self, x, y, cost=0, previous=None, parent=None):
        self.x = x  # index of grid
        self.y = y  # index of grid
        self.cost = cost
        self.previous = previous
        self._parent = parent

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(
            self.cost) + "," + str(self.previous)

    @property
    def position(self):
        x = self._parent.calc_grid_position(self.x, self._parent.min_x)
        y = self._parent.calc_grid_position(self.y, self._parent.min_y)
        return x, y


class AStarPlanner:
    def __init__(self, ox, oy, resolution, rr):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.calc_obstacle_map(ox, oy)
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self._handlers = []

        self.Node = partial(Node, parent=self)
        self._all_nodes = dict()

    def add_handler(self, handler):
        self._handlers.append(handler)

    def node_at(self, x, y):
        x_grid = self.calc_xy_index(x, self.min_x)
        y_grid = self.calc_xy_index(y, self.min_y)
        return self.node_at_grid(x_grid, y_grid)

    def node_at_grid(self, x, y):
        try:
            node = self._all_nodes[(x, y)]
        except KeyError:
            node = self.Node(x, y)
            self._all_nodes[(x, y)] = node

        return node

    def planning(self, sx, sy, gx, gy):
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

        start_node = self.node_at(sx, sy)
        goal_node = self.node_at(gx, gy)

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
                    current.x + motion[0],
                    current.y + motion[1],
                )

                # If the node is not safe, do nothing
                if not self.verify_node(node):
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
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px, py = node.position

        if not self.min_x <= px <= self.max_x:
            return False
        elif not self.min_y <= py <= self.max_y:
            return False
        elif self.obstacle_map[node.x][node.y]:
            # collision check
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

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


def main():
    print(__file__ + " start!!")

    # start and goal position
    sx = 10.0  # [m]
    sy = 10.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]
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
            PPVisualizer(start=(sx, sy), goal=(gx, gy), obstacles=(ox, oy)))

    print('Starting planning')
    a_star.planning(sx, sy, gx, gy)


if __name__ == '__main__':
    main()
