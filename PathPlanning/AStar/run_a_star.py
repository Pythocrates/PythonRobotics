#! /usr/bin/env python3

''' Run a standard A* planner.
'''

from a_star import AStarPlanner
from obstacle_map import ObstacleMap, Position
from pp_visualizer import PPVisualizer


SHOW_ANIMATION = True


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
    obstacle_map = ObstacleMap(obstacles, grid_size, robot_radius)
    a_star = AStarPlanner(obstacle_map)

    if SHOW_ANIMATION:  # pragma: no cover
        a_star.add_handler(
            PPVisualizer(start_position, goal_position, obstacles))

    print('Starting planning')
    a_star.plan(start_position, goal_position)


if __name__ == '__main__':
    main()
