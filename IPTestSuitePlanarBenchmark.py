import sys

from IPPlanarManipulator import PlanarRobot
sys.path.append("./collisionChecker")

from IPBenchmark import Benchmark
from IPEnvironment import CollisionChecker
from collisionChecker.KinChainCollisionChecker import KinChainCollisionChecker

from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
import math


benchList = list()

Spinner_with_Core3DoF = dict()
Spinner_with_Core3DoF["obs1"] = (
    Point(15, 15).buffer(10)
    .difference(Point(15, 15).buffer(8))
    .difference(box(11, 2, 19, 28))  # Breiterer vertikaler Ausschnitt (von 13-17 auf 11-19)
    .difference(box(2, 11, 28, 19))  # Breiterer horizontaler Ausschnitt (von 13-17 auf 11-19)
)
#Spinner_with_Core3DoF["obs2"] = Point(15, 15).buffer(3)


robot_arm_2 = PlanarRobot(n_joints=2, base_x=15, base_y=12.5)
limits_2dof = [[-3.14, 3.14], [-3.14, 3.14]]

benchList.append(Benchmark(
    "PlanarArm_2DoF",
    KinChainCollisionChecker(robot_arm_2, Spinner_with_Core3DoF, limits=limits_2dof),
    [[0.0, 0.0]],       # Start-Konfiguration (Alle Winkel 0)
    [
        [0, math.pi/4],
        [math.pi, -math.pi/4],      # weiteres Ziel
        [-math.pi/2, 0]
    ],   # weiteres Ziel], # Ziel-Winkel (nicht x/y Koordinaten!)
    "2-Gelenk Arm in Hindernisumgebung",
    3
))



robot_arm_3 = PlanarRobot(n_joints=3, base_x=15, base_y=15)
limits_3dof = [[-3.14, 3.14], [-3.14, 3.14], [-3.14, 3.14]]

benchList.append(Benchmark(
    "PlanarArm_3DoF",
    KinChainCollisionChecker(robot_arm_3, Spinner_with_Core3DoF, limits=limits_3dof),
    [[0.0, 0.0, 0.0]],  # Start
    [[0,0,math.pi/4],
      [math.pi, 0, -math.pi/4],
      [-math.pi/2, 0,0]], # Ziele (3 Winkel pro Ziel)
    "3-Gelenk Arm (Schlangenroboter)",
    4
))
