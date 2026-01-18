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

class CollisionCheckerMultipleRobots:
    def __init__(self, scene, robot_shapes: list[Polygon], limits=None, statistic=None):
        """
        scene        : dict von Hindernissen (Polygon / LineString)
        robot_shapes : Liste von Polygonen, ein Polygon pro Roboter
        limits       : Liste von [xmin,xmax],[ymin,ymax],[theta_min,theta_max] pro Roboter
        """
        self.scene = scene
        self.robot_shapes = []
        self.num_robots = len(robot_shapes)

        # standardlimits
        if limits is None:
            limits = []
            for _ in range(self.num_robots):
                limits.extend([[0.0, 22.0], [0.0, 22.0], [0.0, 360.0]])
        self.limits = limits

        for poly in robot_shapes:
            c = poly.centroid
            self.robot_shapes.append(translate(poly, -c.x, -c.y))

    def getDim(self):
        """Return dimension of environment (3 per robot)"""
        return 3 * self.num_robots

    def getEnvironmentLimits(self):
        return list(self.limits)

    @IPPerfMonitor
    def pointInCollision(self, pos):
        """
        pos: [x1,y1,a1, x2,y2,a2, ...] 
        Prüft Kollision mit Hindernissen und gegenseitige Roboterkollision.
        """
        assert len(pos) == self.getDim()
        robots_polygons = []

        # transform robots
        for r in range(self.num_robots):
            idx = r*3
            x, y, a = pos[idx:idx+3]
            rotated = rotate(self.robot_shapes[r], a)
            transformed = translate(rotated, x, y)
            robots_polygons.append(transformed)

        # collision - scene
        for poly in robots_polygons:
            for obs in self.scene.values():
                if obs.intersects(poly):
                    return True

        # collision - robots
        for i in range(self.num_robots):
            for j in range(i+1, self.num_robots):
                if robots_polygons[i].intersects(robots_polygons[j]):
                    return True

        return False

    @IPPerfMonitor
    def lineInCollision(self, startPos, endPos, steps=40):
        """ Prüft Kollision auf einem geraden Segment """
        startPos = np.array(startPos)
        endPos = np.array(endPos)
        delta = endPos - startPos

        for i in range(steps):
            alpha = (i+1)/steps
            interm = startPos + alpha * delta
            if self.pointInCollision(interm):
                return True
        return False

    def drawObstacles(self, ax, color='r'):
        for obs in self.scene.values():
            plotting.plot_polygon(obs, ax=ax, add_points=False, color=color)

    def drawRobots(self, ax, pos, colors=None):
        if colors is None:
            colors = plt.cm.get_cmap('tab10', self.num_robots)

        for r in range(self.num_robots):
            idx = r*3
            x, y, a = pos[idx:idx+3]
            poly = translate(rotate(self.robot_shapes[r], a), x, y)
            col = colors(r) if callable(colors) else colors[r]
            plotting.plot_polygon(poly, ax=ax, color=col)
