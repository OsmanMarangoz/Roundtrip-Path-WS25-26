import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math
import copy
from tqdm.notebook import tqdm
from shapely import plotting
from shapely.geometry import Polygon, Point, LineString
from shapely.affinity import translate, rotate
from IPPerfMonitor import IPPerfMonitor
from IPBenchmark import Benchmark

class CollisionChecker3D_2Robis:

    def __init__(self, scene, robot1_shape: Polygon, robot2_shape: Polygon, 
                 limits=[[0.0, 22.0], [0.0, 22.0], [0.0, 360.0], [0.0, 22.0], [0.0, 22.0], [0.0, 360.0]], 
                 statistic=None):
        self.scene = scene
        self.limits = limits

        r1_center = robot1_shape.centroid
        self.robot1_shape = translate(robot1_shape, -r1_center.x, -r1_center.y)

        r2_center = robot2_shape.centroid
        self.robot2_shape = translate(robot2_shape, -r2_center.x, -r2_center.y)

    def getDim(self):
        """ Return dimension of Environment """
        # 2 robots * 3 dimensions (x,y,theta)
        return 6

    def getEnvironmentLimits(self):
        """ Return limits of Environment"""
        return list(self.limits)

    @IPPerfMonitor
    def pointInCollision(self, pos):
        """
        pos = [x1, y1, a1,  x2, y2, a2]

        Returns True if ANY robot collides with environment
        OR robots collide with each other.
        """

        assert (len(pos) == self.getDim())
        
        robot1_rotated_shape = rotate(self.robot1_shape, pos[2]) 
        r1 = translate(robot1_rotated_shape, pos[0], pos[1])

        robot2_rotated_shape = rotate(self.robot2_shape, pos[5]) 
        r2 = translate(robot2_rotated_shape, pos[3], pos[4])

        # --- Check collision with environment ---
        for obs in self.scene.values():
            if obs.intersects(r1) or obs.intersects(r2):
                return True
        
        # --- Check robot-robot collision ---
        if r1.intersects(r2):
            return True

        return False

    @IPPerfMonitor
    def lineInCollision(self, startPos, endPos):
        """ Check whether a line from startPos to endPos is colliding"""
        assert (len(startPos) == self.getDim())
        assert (len(endPos) == self.getDim())
        
        p1 = np.array(startPos)
        p2 = np.array(endPos)
        delta = p2 - p1

        k = 40
        for i in range(k):
            alpha = (i + 1) / k
            interm = p1 + alpha * delta
            if self.pointInCollision(interm):
                return True

        return False
                

    def drawObstacles(self, ax, color='r'):
        for key, value in self.scene.items():
            plotting.plot_polygon(value, ax=ax, add_points=False, color=color)

    def drawRobots(self, ax, pos):
        """
        Draw both robots.
        """
        x1, y1, a1, x2, y2, a2 = pos

        r1 = translate(rotate(self.robot1_shape, a1), x1, y1)
        r2 = translate(rotate(self.robot2_shape, a2), x2, y2)

        plotting.plot_polygon(r1, ax=ax, color='k')
        plotting.plot_polygon(r2, ax=ax, color='b')   