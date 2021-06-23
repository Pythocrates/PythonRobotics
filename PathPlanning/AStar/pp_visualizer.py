''' This module contains a common visualizer class for path planning.
'''

import sys

import matplotlib.pyplot as plt


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
