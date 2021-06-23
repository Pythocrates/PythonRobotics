"""

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""

import math

from node import Node
from obstacle_map import Position


class AStarPlanner:
    def __init__(self, obstacle_map):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        robot_radius: robot radius[m]
        """

        self.obstacle_map = obstacle_map
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

    def plan(self, start_position, goal_position):
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
