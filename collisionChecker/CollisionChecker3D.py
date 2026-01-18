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

class CollisionChecker3D(object):

    def __init__(self, scene, robot_shape: Polygon, limits=[[0.0, 22.1], [0.0, 22.1], [0.0, 360.0]], statistic=None):
        self.scene = scene
        self.limits = limits

        self.robot_shape = robot_shape
        center = robot_shape.centroid
        self.robot_shape = translate(self.robot_shape, -center.x, -center.y)

    def getDim(self):
        """ Return dimension of Environment (Shapely should currently always be 2)"""
        return 3

    def getEnvironmentLimits(self):
        """ Return limits of Environment"""
        return list(self.limits)

    @IPPerfMonitor
    def pointInCollision(self, pos):
        """ Return whether a configuration is
        inCollision -> True
        Free -> False """

        assert (len(pos) == self.getDim())
        
        rotated_shape = rotate(self.robot_shape, pos[2]) 
        transformed_shape = translate(rotated_shape, pos[0], pos[1])

        for key, value in self.scene.items():
            if value.intersects(transformed_shape):
                return True
        return False

    @IPPerfMonitor
    def lineInCollision(self, startPos, endPos):
        """ Check whether a line from startPos to endPos is colliding"""
        assert (len(startPos) == self.getDim())
        assert (len(endPos) == self.getDim())
        
        p1 = np.array(startPos)
        p2 = np.array(endPos)
        p12 = p2-p1
        k = 40
        
        for i in range(k):
            testPos = p1 + (i+1)/k*p12
            if self.pointInCollision(testPos)==True:
                return True
        
        return False
                

    def drawObstacles(self, ax, color='r'):
        for key, value in self.scene.items():
            plotting.plot_polygon(value, ax=ax, add_points=False, color=color)

    def drawRobot(self, ax, pos):
        rotated_shape = rotate(self.robot_shape, pos[2])
        transformed_shape = translate(rotated_shape, pos[0], pos[1])
        plotting.plot_polygon(transformed_shape, ax, color='k')