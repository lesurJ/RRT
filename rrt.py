""" This script is a python implementation of the Rapidly-exploring Random Tree algorithm. """
""" Authored : Jean Lesur, 2021 """

import numpy as np
import matplotlib.pyplot as plt
import math

class RRTNode():
    """ This class helps representing a node in the RRT."""
    def __init__(self, id, pos, previous):
        self.id = id #id of current node
        self.node = pos
        self.previous_id = previous #id of the node at the other end of the edge

class RRT():
    """ This class constructs a RRT in a N-dimensional space given a start and a goal node.
    """
    def __init__(self, N, bounds=None, start=None, goal=None, stepsize=0.1, beta=0.5):
        # PROBLEM PARAMETERS
        self.N = N # dimension of the problem
        self.bounds = bounds if bounds is not None else np.ones(self.N) # the bounds are assumed to be symmetric
        self.start = start if start is not None else self.getRandomTarget()
        self.goal = goal if goal is not None else self.getRandomTarget()

        # OBSTACLES
        self.nb_obstacles = 15
        self.obstacles = self.generateObstacles()

        # VERTICES & EDGES
        self.path = []
        self.visited_nodes = []
        self.optimized_path = []
        self.edges = []

        # HYPERPARAMETERS
        self.stepsize = stepsize
        self.beta = beta

        # SMOOTHING
        self.smooth_enable = True
        self.smoothed_path = []
        self.smoothing_order = 1
        self.smoothing_order_max = 7
        self.smoothing_resolution = 200
        self.spacing = np.linspace(0,1,self.smoothing_resolution)

    def runRRT(self, mode='dual'):
        """
        Run the RRT algorithm, find executable path and data.
        """
        if mode == 'single':
            self.visited_nodes, self.path = self.runRRT_single()
        elif mode == 'dual':
            self.visited_nodes, self.path = self.runRRT_dual()

        self.optimized_path = [item.node for item in self.optimizePath(self.path)]
        if self.smooth_enable:
            self.smoothed_path = self.smoothPath(self.path)

    def runRRT_single(self):
        """Construct a single RRT tree, find its path and optimize it."""
        tree = []
        path = []

        idCounter = 1
        success = False
        new = RRTNode(0, self.start, None)
        tree.append(new)
        while not success:
            # SAMPLING : find the direction of the growth (target or random)
            if np.random.rand() < self.beta :
                rrt_goal = self.goal
            else:
                rrt_goal = self.getRandomTarget()

            # GROWTH : grow the tree in the direction of the current sample.
            # If this is not possible due to constraint violation, grow the
            # tree towards a random sample
            q, prev, isProblematic = self.getProposalTowards(rrt_goal, tree)
            while isProblematic:
                q, prev, isProblematic = self.getProposalTowards(self.getRandomTarget(), tree)
            idCounter = self.add(tree, q, idCounter, prev.id)
            self.edges.append([prev.node, q])

            # EXIT CONDITION : are we close enough to the target ?
            if np.linalg.norm(q - self.goal) < self.stepsize:
                success = True

        # Add the target to the tree
        self.edges.append([tree[-1].node, self.goal])
        idCounter = self.add(tree, self.goal, idCounter, idCounter-1)

        # PATH RETRIEVAL : finish RRT
        path = self.findPath(tree)
        visited_nodes = [list(item.node) for item in tree]

        return visited_nodes, path

    def runRRT_dual(self):
        """
        Construct 2 RRTs and make them grow towards each other. Find the total
        path and optimize it.
        """
        tree0 = [] # 0 is the tree growing form the start configuration
        tree1 = [] # 1 is the tree growing from the target configuration

        idCounter0 = 1
        idCounter1 = 1
        success = False
        init0 = RRTNode(0, self.start, None)
        tree0.append(init0)
        init1 = RRTNode(0, self.goal, None)
        tree1.append(init1)
        while not success:
            # SAMPLING : find the direction of the growth (target or random) for the two trees
            if np.random.rand() < self.beta :
                nearest1 = self.getNearest(tree0[-1].node, tree1)
                nearest0 = self.getNearest(tree1[-1].node, tree0)
                rrt0_goal = tree1[nearest1].node
                rrt1_goal = tree0[nearest0].node
            else:
                rrt0_goal = self.getRandomTarget()
                rrt1_goal = self.getRandomTarget()

            # GROWTH : grow the first tree in the direction of the current sample.
            # If this is not possible due to constraint violation, grow the
            # first tree towards a random sample
            q0, prev, isProblematic = self.getProposalTowards(rrt0_goal, tree0)
            while isProblematic:
                q0, prev, isProblematic = self.getProposalTowards(self.getRandomTarget(), tree0)
            idCounter0 = self.add(tree0, q0, idCounter0, prev.id)
            self.edges.append([prev.node, q0]) # keep track of edges in the tree for plotting

            # Verify if the newly appended node is close enough the the other tree.
            nearest1 = self.getNearest(q0, tree1)
            if np.linalg.norm(tree1[nearest1].node - q0) < self.stepsize:
                self.add(tree1, q0, idCounter1, tree1[nearest1].id)
                break

            # GROWTH : grow the second tree in the direction of the current sample.
            # If this is not possible due to constraint violation, grow the
            # second tree towards a random sample
            q1, prev, isProblematic = self.getProposalTowards(rrt1_goal, tree1)
            while isProblematic:
                q1, prev, isProblematic = self.getProposalTowards(self.getRandomTarget(), tree1)
            idCounter1 = self.add(tree1, q1, idCounter1, prev.id)
            self.edges.append([prev.node, q1])

            # Verify if the newly appended node is close enough the the other tree.
            nearest0 = self.getNearest(q1, tree0)
            if np.linalg.norm(tree0[nearest0].node - q1) < self.stepsize:
                self.add(tree0, q1, idCounter0, tree0[nearest0].id)
                break

        # PATH RETRIEVAL : the total path is the concatenation of the paths of the two trees.
        path = self.findPath(tree0)
        path += self.findPath(tree1, reverse=False)

        visited_nodes = [list(item.node) for item in tree0]
        visited_nodes += [list(item.node) for item in tree1]

        return visited_nodes, path

    def getNearest(self, x_goal, tree):
        """ Find the index of the closest to x_goal in a RRT tree.
            params:
                x_goal : the point we want to find the closest neighbor
                tree   : the tree in which to find the closest neighbor
        """
        array = np.array([item.node for item in tree])
        nearest = (np.linalg.norm(array - x_goal, axis=1)).argmin()
        return nearest

    def getProposalTowards(self, x_goal, tree):
        """ Given a desired target (that can either be goal or random), find the
            closest point to it in the tree and compute a new branch in this
            direction.
            params:
                x_goal : the point in search space that we want to reach.
                tree   : rrt tree in which to find the closest point.
        """

        nearest = self.getNearest(x_goal, tree)
        diff = x_goal - tree[nearest].node
        dist = np.linalg.norm(diff)
        x_new = tree[nearest].node + self.stepsize/dist * diff

        isOutsideRange = self.checkBounds(x_new) # does the new configuration respect constraints ?
        isCollision = self.checkCollision(x_new) # does the new configuration trigger a collision ?
        flag = isOutsideRange or isCollision
        return x_new, tree[nearest], flag

    def getRandomTarget(self):
        """ Compute a reachable random target. """
        output = []
        for k in self.bounds:
            output.append(np.random.uniform(-k, k))
        return np.array(output)

    def add(self, tree, q, id, prev_id):
        """ Adds a new node object in the tree.
            params:
                tree    : the rrt tree in which one adds a point
                q       : point in search space of the new node
                id      : id of the new node
                prev_id : id of the previous node (the parent node)
        """
        new = RRTNode(id, q, prev_id)
        tree.append(new)
        return id+1

    def findPath(self, tree, reverse=True):
        """ Retrieve the path from start to goal in the tree.
            params:
                tree    : rrt tree in which to find the path
                reverse : flag to specify if the path must be reverted, useful for bidirectional version
        """
        path = []
        path.append(tree[-1])
        for n in reversed(tree):
            if (n.id == path[-1].previous_id):
                path.append(n)
        if reverse:
            path.reverse()
        return path

    def checkBounds(self, q):
        """ Checks whether a point in search space is reachable.
            params:
                q : point in search space to be checked.
        """
        if (np.abs(q) >= self.bounds).any():
            return True
        return False

    def generateObstacles(self):
        """ Compute random positions and sizes of the obstacles"""
        obstacles = []
        for _ in range(self.nb_obstacles):
            pos = self.getRandomTarget()
            r = np.random.uniform(0.1,0.3)
            if np.linalg.norm(pos - self.start) < r or np.linalg.norm(pos - self.goal) < r:
                continue
            else:
                obstacles.append([pos, r])
        return obstacles

    def checkCollision(self, q):
        """ Checks if configuration q triggers a collision with obstacles.
        """
        
        for o in self.obstacles:
            pos, r = o
            if np.linalg.norm(pos - q) < r:
                return True

        return False

        # # dummy implementation, change it to your convenience


        # a check conducted in simulation should be preferred (e.g with Pybullet)
        # return simulation.verify_collision(q)

    def freePath(self, a, b):
        """ Checks whether the path connecting a to b in search space is free in
            task space.
            params:
                a : start point
                b : end point
        """
        a = np.array(a)
        b = np.array(b)
        size = int(np.linalg.norm(a - b)/self.stepsize)
        for s in range(size):
            point = a + s/size * (b - a)
            is_collision = self.checkCollision(point)
            if is_collision:
                return False
        return True

    def optimizePath(self, path):
        """ Optimize the path found by removing unnecessary points.
            params:
                path : the path to optimize
        """

        optimized_path = []
        optimized_path.append(path[0])
        last = optimized_path[-1]

        cnt = 1
        size = len(path)
        finished = False
        while not finished:
            for i in range(cnt, size):
                if not self.freePath(last.node, path[i].node):
                    optimized_path.append(path[i-1])
                    last = optimized_path[-1]
                    cnt = i-1
                    break

                if i == size - 1:
                    optimized_path.append(path[-1])
                    finished = True
        return optimized_path

    def smoothPath(self, path):
        """ Smooth the path found by RRT using a Bézier curve. The order 
            of smoothness (number of appearance of each point along the curve)
            will increase if there is a collision between the smoothed path and
            the obstacles.
        """
        smoothed_path = []
        isValid = False
        while not isValid:
            if self.smoothing_order >= self.smoothing_order_max:
                smoothed_path = []
                print("Sorry : unable to use a Bézier curve for smoothing")
                break
            flag = True
            smoothed_path = self.BezierCurve([item.node for item in path for _ in range(self.smoothing_order)])
            for p in smoothed_path:
                if self.checkCollision(p):
                    flag = False
                    self.smoothing_order += 1
                    break
            
            if flag:
                isValid = True
        return smoothed_path
        
    def BernsteinPolynomial(self, n, i):
        """ Compute the Bernstein polynomial """
        return lambda t: math.comb(n,i) * t**i * (1-t)**(n-i)

    def BezierCurve(self, vectors):
        """ Given a collection of control points, find the parametric Bezier curve"""
        n = len(vectors)
        points = []
        for t in self.spacing:
            polynomial = np.array([self.BernsteinPolynomial(n-1, i) for i in range(n)])
            points.append(np.array(vectors).T @ [p(t) for p in polynomial])
        return points

    def plot(self):
        """
        Plot the visited nodes, retrieved path as well as start and goal.
        """
        if self.N == 2:
            fig, (ax1, ax2) = plt.subplots(1,2)

            for e in self.edges:
                _from, _to = e
                ax1.plot([_from[0], _to[0]],[_from[1], _to[1]], c='black', linestyle='-')
            ax1.scatter([item[0] for item in self.visited_nodes], [item[1] for item in self.visited_nodes], label='visited nodes', c='black', marker='.')           
            ax1.plot([item.node[0] for item in self.path], [item.node[1] for item in self.path], label='path', c='blue')
            ax1.plot([item[0] for item in self.optimized_path], [item[1] for item in self.optimized_path], label='optimized path', c='red', linestyle='--')
            ax1.scatter(self.start[0], self.start[1], label='start', c='red')
            ax1.scatter(self.goal[0], self.goal[1], label='goal', c='green')

            ax2.plot([item.node[0] for item in self.path], [item.node[1] for item in self.path], label='path', c='blue')
            ax2.plot([p[0] for p in self.smoothed_path], [p[1] for p in self.smoothed_path], label=f"Bezier curve", c='green')
            ax2.scatter(self.start[0], self.start[1], label='start', c='red')
            ax2.scatter(self.goal[0], self.goal[1], label='goal', c='green')

            
            for o in self.obstacles:
                pos, r = o
                c = plt.Circle(pos, r, color='gray', alpha=0.5)
                ax1.add_patch(c)
                c = plt.Circle(pos, r, color='gray', alpha=0.5)
                ax2.add_patch(c)

            for a, title in zip((ax1, ax2), ['RRT', 'Smoothing']):
                a.set_xlim([-self.bounds[0],self.bounds[0]])
                a.set_ylim([-self.bounds[1],self.bounds[1]])
                a.legend()
                a.set_title(title)
                a.set_xlabel('dim. 1')
                a.set_ylabel('dim. 2')
                a.axis('square')
            fig.suptitle('Rapidly-exploring Random Tree')
            plt.show()

        elif self.N == 3:
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            for e in self.edges:
                _from, _to = e
                ax.plot3D([_from[0], _to[0]],[_from[1], _to[1]],[_from[2], _to[2]], c='black', linestyle='-')
            
            ax.scatter3D([item[0] for item in self.visited_nodes], [item[1] for item in self.visited_nodes], [item[2] for item in self.visited_nodes], label='visited nodes', c='black', marker='.')
            ax.plot3D([item.node[0] for item in self.path], [item.node[1] for item in self.path], [item.node[2] for item in self.path], label='path', c='blue')
            ax.plot3D([item[0] for item in self.optimized_path], [item[1] for item in self.optimized_path], [item[2] for item in self.optimized_path], label='optimized path', c='red', linestyle='--')
            ax.plot3D([item[0] for item in self.smoothed_path], [item[1] for item in self.smoothed_path], [item[2] for item in self.smoothed_path], label='path', c='green')

            ax.scatter3D(self.start[0], self.start[1], self.start[2], label='start', c='red')
            ax.scatter3D(self.goal[0], self.goal[1], self.goal[2], label='goal', c='green')

            for o in self.obstacles:
                pos, r = o
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = r*np.cos(u)*np.sin(v)
                y = r*np.sin(u)*np.sin(v)
                z = r*np.cos(v)
                ax.plot_wireframe(pos[0] + x, pos[1] + y, pos[2] + z, color="r", linewidth=0.25)



            ax.set_xlim3d([-1, 1])
            ax.set_ylim3d([-1, 1])
            ax.set_zlim3d([-1, 1])
            ax.set_xlabel('dimension 1')
            ax.set_ylabel('dimension 2')
            ax.set_zlabel('dimension 3')
            ax.legend()
            plt.show()
        else:
            print(f"[ERROR] the data with dimension {self.N} cannot be plotted! ")


if __name__=='__main__':
    # 1. Choose dimension of the problem.
    N = 2
    # 2. Instantiate the RRT object with N, you may indicate the start and the goal configurations.
    r = RRT(N, start=-0.75*np.ones(N), goal=0.75*np.ones(N))
    # 3. Run the RRT planner (the default one uses bidirectional search)
    r.runRRT(mode='dual')
    # 4. Plot the results
    r.plot()

# <END OF FILE>