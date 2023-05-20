import numpy as np

from rrt import SingleRRT


if __name__ == "__main__":
    N = 2
    start = 3 * (np.random.rand(2) - 0.5)
    goal = 3 * (np.random.rand(2) - 0.5)

    r = SingleRRT(N, start, goal)
    r.run()

    if N == 2:
        r.plot()
