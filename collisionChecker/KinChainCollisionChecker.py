from IPEnvironment import CollisionChecker
from shapely.geometry import LineString
import numpy as np
import copy

def interpolate_line(startPos, endPos, step_l):
    steps = []
    line = np.array(endPos) - np.array(startPos)
    line_l = np.linalg.norm(line)
    step = line / line_l * step_l
    n_steps = np.floor(line_l / step_l).astype(np.int32)
    c_step = np.array(startPos)
    for i in range(n_steps):
        steps.append(copy.deepcopy(c_step))
        c_step += step
    if not (c_step == np.array(endPos)).all():
        steps.append(np.array(endPos))
    return steps


class KinChainCollisionChecker(CollisionChecker):
    def __init__(self, kin_chain, scene, limits=[[-3.0, 3.0], [-3.0, 3.0]], statistic=None, fk_resolution=0.1):
        super(KinChainCollisionChecker, self).__init__(scene, limits, statistic)
        self.kin_chain = kin_chain
        self.fk_resolution = fk_resolution
        self.dim = self.kin_chain.dim
        self.collision_calls = 0

    def getDim(self):
        return self.dim


    def pointInCollision(self, pos):
        self.collision_calls += 1
        self.kin_chain.move(pos)
        joint_positions = self.kin_chain.get_transforms()
        self.dim = 2
        for i in range(1, len(joint_positions)):
            if self.segmentInCollision(joint_positions[i-1], joint_positions[i]):
                self.dim = self.kin_chain.dim
                return True
        self.dim = self.kin_chain.dim
        return False

    def lineInCollision(self, startPos, endPos):
        self.collision_calls += 1
        assert (len(startPos) == self.getDim())
        assert (len(endPos) == self.getDim())
        steps = interpolate_line(startPos, endPos, self.fk_resolution)
        for pos in steps:
            if self.pointInCollision(pos):
                return True
        return False

    def segmentInCollision(self, startPos, endPos):
        self.collision_calls += 1
        assert (len(startPos) == self.getDim())
        assert (len(endPos) == self.getDim())
        for key, value in self.scene.items():
            if value.intersects(LineString([(startPos[0], startPos[1]), (endPos[0], endPos[1])])):
                return True
        return False

    def drawObstacles(self, ax, inWorkspace=False):
        if inWorkspace:
            for key, value in self.scene.items():
                plotting.plot_polygon(value, add_points=False, color='red', ax=ax)

    def drawRobot(self, ax):
        joint_positions = self.kin_chain.get_transforms()
        for i in range(1, len(joint_positions)):
            xs = [joint_positions[i-1][0], joint_positions[i][0]]
            ys = [joint_positions[i-1][1], joint_positions[i][1]]
            ax.plot(xs, ys, color='g')

    def drawObstacles_patched(self, ax, inWorkspace=False):
        if inWorkspace:
            for key, value in self.scene.items():
                plotting.plot_polygon(value, add_points=False, color='red', ax=ax)