""" This script is a python implementation of the Rapidly-exploring Random Tree algorithm. """
""" Authored : Jean Lesur, 2021 """

import numpy as np
import matplotlib.pyplot as plt

class RRTNode():
    """ This class helps representing a node in the RRT."""
    def __init__(self, id, pos, previous):
        self.id = id #id of current node
        self.node = pos
        self.previous_id = previous #id of the node at the other end of the edge

class RRT():
    """ This class constructs a RRT in a N-dimensional space given a start and a goal node.
    """
    def __init__(self, N, start, goal, stepsize=0.1, beta=0.5):
        self.N = N # dimension of the problem
        self.bounds = np.ones(self.N) # change this to any value you want; the bounds are assumed to be symmetric

        self.start = None
        self.goal = None    
        self.setStartGoal(start, goal)

        # VERTICES & EDGES
        self.path = []
        self.visited_nodes = []
        self.optimized_path = []
        self.edges = []

        # HYPERPARAMETERS
        self.stepsize = stepsize
        self.beta = beta

    def setStartGoal(self, start, goal):
        """
        Set start and goal positions of the RRT.
        params:
             start : start position in joint space
             goal  : goal position in joint space
        """
        if len(start) == self.N:
            self.start = start
        else:
            raise ValueError(f"[ERROR] : start (dim {len(start)}) does not have the proper dimensionality (dim {self.N})")

        if len(goal) == self.N:
            self.goal = goal
        else:
            raise ValueError(f"[ERROR] : goal (dim {len(goal)}) does not have the proper dimensionality (dim {self.N})")

    def setParameters(self, ui):
        """ a callback to be called when sliders in the GUI are changed
            params:
                ui : the user interface object
        """
        self.beta = ui.horizontalSlider_rrtBeta.value()/100
        self.stepsize = ui.horizontalSlider_rrtStepsize.value()*0.05

    def runRRT(self, mode='dual'):
        """
        Run the RRT algorithm, find executable path and data.
        """
        if mode == 'single':
            self.visited_nodes, self.path, self.optimized_path = self.runRRT_single()
        elif mode == 'dual':
            self.visited_nodes, self.path, self.optimized_path = self.runRRT_dual()
        self.optimized_path = [item.node for item in self.optimized_path]

        # Depending on your needs, you might want to reduce dimension of the RRT path !
        # Here you can use dimension reduction for example
        # self.reduceDimension(0)

    def runRRT_single(self):
        """Construct a single RRT tree, find its path and optimize it."""
        tree = []
        path = []

        idCounter = 1
        success = False
        new = RRTNode(0, self.start, None)
        tree.append(new)
        while not success:
            if np.random.rand() < self.beta :
                rrt_goal = self.goal
            else:
                rrt_goal = self.getRandomTarget()

            q, prev, isProblematic = self.getProposalTowards(rrt_goal, tree)
            while isProblematic:
                q, prev, isProblematic = self.getProposalTowards(self.getRandomTarget(), tree)
            idCounter = self.add(tree, q, idCounter, prev.id)
            self.edges.append([prev.node, q])

            if np.linalg.norm(q - self.goal) < self.stepsize:
                success = True

        self.edges.append([tree[-1].node, self.goal])
        idCounter = self.add(tree, self.goal, idCounter, idCounter-1)

        path = self.findPath(tree)
        optimized_path = self.optimizePath(path)
        visited_nodes = [list(item.node) for item in tree]

        return visited_nodes, path, optimized_path

    def runRRT_dual(self):
        """
        Construct 2 RRT and make them grow towards each other. Find the total
        path and optimize it.
        """
        tree0 = [] # 0 is start
        tree1 = [] # 1 is goal

        idCounter0 = 1
        idCounter1 = 1
        success = False
        init0 = RRTNode(0, self.start, None)
        tree0.append(init0)
        init1 = RRTNode(0, self.goal, None)
        tree1.append(init1)
        while not success:
            if np.random.rand() < self.beta :
                nearest1 = self.getNearest(tree0[-1].node, tree1)
                nearest0 = self.getNearest(tree1[-1].node, tree0)
                rrt0_goal = tree1[nearest1].node
                rrt1_goal = tree0[nearest0].node
            else:
                rrt0_goal = self.getRandomTarget()
                rrt1_goal = self.getRandomTarget()

            q0, prev, isProblematic = self.getProposalTowards(rrt0_goal, tree0)
            while isProblematic:
                q0, prev, isProblematic = self.getProposalTowards(self.getRandomTarget(), tree0)
            idCounter0 = self.add(tree0, q0, idCounter0, prev.id)
            self.edges.append([prev.node, q0])

            nearest1 = self.getNearest(q0, tree1)
            if np.linalg.norm(tree1[nearest1].node - q0) < self.stepsize:
                self.add(tree1, q0, idCounter1, tree1[nearest1].id)
                break

            q1, prev, isProblematic = self.getProposalTowards(rrt1_goal, tree1)
            while isProblematic:
                q1, prev, isProblematic = self.getProposalTowards(self.getRandomTarget(), tree1)
            idCounter1 = self.add(tree1, q1, idCounter1, prev.id)
            self.edges.append([prev.node, q1])

            nearest0 = self.getNearest(q1, tree0)
            if np.linalg.norm(tree0[nearest0].node - q1) < self.stepsize:
                self.add(tree0, q1, idCounter0, tree0[nearest0].id)
                break

        path = self.findPath(tree0)
        path += self.findPath(tree1, reverse=False)

        optimized_path = self.optimizePath(path)

        visited_nodes = [list(item.node) for item in tree0]
        visited_nodes += [list(item.node) for item in tree1]

        return visited_nodes, path, optimized_path

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
                x_goal : the point in joint space that we want to reach.
                tree   : rrt tree in which to find the closest point.
        """

        nearest = self.getNearest(x_goal, tree)
        diff = x_goal - tree[nearest].node
        dist = np.linalg.norm(diff)
        x_new = tree[nearest].node + self.stepsize/dist * diff

        isOutsideRange = self.checkBounds(x_new)
        isCollision = self.checkCollision(x_new)
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
                q       : point in joint space of the new node
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
        """ Checks whether a point in joint space is reachable.
            params:
                q : point in joint space to be checked.
        """
        if (np.abs(q) >= self.bounds).any():
            return True
        return False

    def checkCollision(self, q):
        """ Checks if configuration q triggers a collision with obstacles.
        """
        # dummy implementation, change it to your convenience
        if np.linalg.norm(np.zeros(N) - q) < 0.25: # avoid center of radius 0.25 centered at the origin
            return True
        else:
            return False

        # this should be preferred : 
        # return simulation.verify_collision(q)

    def freePath(self, a, b):
        """ Checks whether the path connecting a to b in joint space is free in
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

    def plot(self):
        """
        Plot the visited nodes, retrieved as well as start and
        goal.
        """
        if self.N == 2:
            fig, ax = plt.subplots()
            ax.scatter([item[0] for item in self.visited_nodes], [item[1] for item in self.visited_nodes], label='visited nodes', c='black', marker='o')           
            ax.plot([item.node[0] for item in self.path], [item.node[1] for item in self.path], label='path', c='blue')
            # ax.plot([item[0] for item in self.optimized_path], [item[1] for item in self.optimized_path], label='optimized', c='red')

            # for i in range(len(self.path)):
            #     ax.annotate(f"{i}", (self.path[i].node[0],self.path[i].node[1]))

            for e in self.edges:
                _from, _to = e
                ax.plot([_from[0], _to[0]],[_from[1], _to[1]], c='black', linestyle=':')

            ax.scatter(self.start[0], self.start[1], label='start', c='red')
            ax.scatter(self.goal[0], self.goal[1], label='goal', c='green')

            c1 = plt.Circle((0,0), 0.25, color='gray')
            ax.add_patch(c1)

            plt.xlim([-1,1])
            plt.ylim([-1,1])
            plt.legend()
            fig.suptitle("Rapidly-exploring Random Tree")
            plt.xlabel("dimension 1")
            plt.ylabel("dimension 2")
            plt.axis('equal')
            plt.show()

        elif self.N == 3:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            
            ax.scatter3D([item[0] for item in self.visited_nodes], [item[1] for item in self.visited_nodes], [item[2] for item in self.visited_nodes], label='visited nodes', c='black', marker='o')
            ax.plot3D([item.node[0] for item in self.path], [item.node[1] for item in self.path], [item.node[2] for item in self.path], label='path', c='blue')

            # ax.plot3D([item[0] for item in self.cart_optimized_path], [item[1] for item in self.cart_optimized_path], [item[2] for item in self.cart_optimized_path], label='cart. opti. path', c='magenta')

            for e in self.edges:
                _from, _to = e
                ax.plot3D([_from[0], _to[0]],[_from[1], _to[1]],[_from[2], _to[2]], c='black', linestyle=':')

            ax.scatter3D(self.start[0], self.start[1], self.start[2], label='start', c='red')
            ax.scatter3D(self.goal[0], self.goal[1], self.goal[2], label='goal', c='green')

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
    N = 2
    r = RRT(N, -0.5*np.ones(N), 0.5*np.ones(N))
    r.runRRT()
    r.plot()

# <END OF FILE>