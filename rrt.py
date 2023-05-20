import numpy as np
import matplotlib.pyplot as plt


class BaseRRT:
    def __init__(self, N, start, goal, stepsize=0.1, beta=0.5):
        """Constructor.

        Args:
            N (int): Number of dimensions.
            start (ndarray): the start configuration. Defaults to None.
            goal (ndarray): the target configuration. Defaults to None.
            stepsize (float, optional): the length of the edges. Defaults to 0.1.
            beta (float, optional): The mixing parameter. Defaults to 0.5.
        """
        # PARAMETERS
        self.N = N
        self.start = start
        assert (
            self.start.size == self.N
        ), f"Start configuration should be of size {N} (current size={self.start.size})"
        self.goal = goal
        assert (
            self.goal.size == self.N
        ), f"Goal configuration should be of size {N} (current size={self.goal.size})"
        self.minBounds = np.min((self.start, self.goal), axis=0) - 1
        self.maxBounds = np.max((self.start, self.goal), axis=0) + 1
        self.obstacles = self._generateObstacles(nb_obstacles=20)

        # HYPERPARAMETERS
        self.stepsize = stepsize
        self.beta = beta
        assert (
            self.beta >= 0 and self.beta <= 1
        ), f"Beta should be in [0,1] (current beta={self.beta})."

        # TREE & PATH
        self.tree = np.array([]).reshape(0, self.N + 1)  # store x1,x2,...,xn,parent_id
        self.path = np.array([]).reshape(0, self.N)  # store x1,x2,...,xn

    def _getRandomTarget(self):
        """Pick a random point within the min-max bounds.

        Returns:
            ndarray: (1 x self.N) the random point.
        """
        return np.random.uniform(self.minBounds, self.maxBounds)

    def _generateObstacles(self, nb_obstacles):
        """Generate circular obstacles.

        Args:
            nb_obstacles (int): the number of obstacles to generate.

        Returns:
            ndarray: (nb_obstacles x self.N+1) the array containing the
                     circular obstacles stored as x1,...,xn,r
        """
        obstacles = np.zeros((nb_obstacles, self.N + 1))

        for i in range(nb_obstacles):
            isCollision = True
            while isCollision:
                pos = self._getRandomTarget()
                r = np.random.uniform(0.1, 0.4)

                c1 = np.linalg.norm(pos - self.start) > r
                c2 = np.linalg.norm(pos - self.goal) > r
                if c1 and c2:
                    isCollision = False

            obstacles[i, :] = np.array([*pos, r])
        return obstacles

    def _getProposalTowards(self, temporary_goal):
        """Given a temporary goal, compute a new edge connecting
        the temporary goal and its closest point in the tree.

        Args:
            temporary_goal (ndarray): the temporary goal = the direction
            in which to grow.

        Returns:
            ndarray, int: the new position and the index of its closest point
            in the tree.
        """
        id = np.linalg.norm(self.tree[:, : self.N] - temporary_goal, axis=1).argmin()
        dir = temporary_goal - self.tree[id, : self.N]
        return self.tree[id, : self.N] + self.stepsize * (dir / np.linalg.norm(dir)), id

    def _isCollision(self, position):
        """Verify if a position is colliding with obsacles.

        Args:
            position (ndarray): the position to check for collision.

        Returns:
            boolean: a flag indicating a collision
        """
        obstacles_pos = self.obstacles[:, : self.N]
        obstacles_r = self.obstacles[:, -1]
        collisions = np.linalg.norm(obstacles_pos - position, axis=1) < obstacles_r
        return collisions.any()

    def _findPath(self):
        """Retrieve the RRT path from the tree."""
        self.path = np.vstack((self.path, self.tree[-1, : self.N]))

        id = -1
        pathFound = False
        while not pathFound:
            id = int(self.tree[id, -1])
            self.path = np.vstack((self.path, self.tree[id, : self.N]))
            pathFound = id == 0


class SingleRRT(BaseRRT):
    def __init__(self, N, start, goal, stepsize=0.1, beta=0.5):
        super().__init__(N, start, goal, stepsize, beta)

    def run(self):
        """Run the RRT planner."""
        self.tree = np.vstack((self.tree, np.array([*self.start, 0])))

        success = False
        while not success:
            # SAMPLING
            temporary_goal = (
                self.goal if np.random.rand() < self.beta else self._getRandomTarget()
            )

            # GROWTH
            position, id = self._getProposalTowards(temporary_goal)
            isCollision = self._isCollision(position)
            if not isCollision:
                self.tree = np.vstack((self.tree, np.array([*position, id])))

                # EXIT CONDITION
                if np.linalg.norm(self.tree[-1, : self.N] - self.goal) < self.stepsize:
                    success = True
                    self.tree = np.vstack(
                        (self.tree, np.array([*self.goal, self.tree.shape[0] - 1]))
                    )

        self._findPath()

    def plot(self):
        """Plot the RRT path and tree."""
        assert self.N == 2, "Plotting is only possible for N=2."

        _, ax = plt.subplots()

        ### START & GOAL
        ax.scatter(self.start[0], self.start[1], label="start", c="red")
        ax.scatter(self.goal[0], self.goal[1], label="goal", c="green")

        ### PATH & NODES

        for i in range(1, self.tree.shape[0]):
            x1, y1, id1 = self.tree[i, :]
            x2, y2, _ = self.tree[int(id1), :]
            ax.plot([x1, x2], [y1, y2], c="k")
        ax.scatter(self.tree[:, 0], self.tree[:, 1], c="k", marker=".", label="tree")
        ax.plot(self.path[:, 0], self.path[:, 1], c="b", label="RRT path")

        ### OBSTACLES
        for o in self.obstacles:
            x, y, r = o
            c = plt.Circle((x, y), r, color="gray", alpha=0.5)
            ax.add_patch(c)

        ax.set_title("Rapidly-exploring Random Tree")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.legend()
        ax.set_xlim([self.minBounds[0], self.maxBounds[0]])
        ax.set_ylim([self.minBounds[1], self.maxBounds[1]])

        ax.set_aspect("equal", "box")
        plt.show()
        plt.close()
