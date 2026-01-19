# coding: utf-8
"""
Benchmark Test Suite für Roundtrip-Planer - Aufgabe b)
Enthält 4 Benchmarkumgebungen:
- 2x 2-DoF Punktroboter (doubleBowl, arcade)
- 2x 3-DoF ShapeRoboter (spinner, grid)

Basiert auf der Struktur von IPTestSuite.py
"""

import sys

from IPPlanarManipulator import PlanarRobot
sys.path.append("./collisionChecker")

from IPBenchmark import Benchmark
from IPEnvironment import CollisionChecker
from CollisionChecker3D import CollisionChecker3D
from collisionChecker.KinChainCollisionChecker import KinChainCollisionChecker

from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

# ============================================================================
# BENCHMARK LISTE
# ============================================================================
benchList = list()

# ============================================================================
# 2-DoF PUNKTROBOTER BENCHMARKS (2 Stück)
# ============================================================================

# --- Benchmark 1: DoubleBowl (2-DoF Punkt) - LEICHT ---
doubleBowl2DoF = dict()
doubleBowl2DoF["obs1"] = box(5, 5, 18, 15).difference(box(7, 7, 16, 16))
doubleBowl2DoF["obs2"] = box(11, 12, 24, 22).difference(box(13, 9, 22, 20))
description_bowl = "Zwei U-förmige Hindernisse"
goalList_bowl = [[17, 17], [3,3], [25, 5], [15, 27]]
benchList.append(Benchmark(
    "doubleBowl_2DoF",
    CollisionChecker(doubleBowl2DoF, limits=([0, 30], [0, 30])),
    [[11, 9]],
    goalList_bowl,
    description_bowl,
    1
))

# --- Benchmark 2: Arcade Scene (2-DoF Punkt) - MITTEL ---
arcade_scene2DoF = dict()
pattern = [
    "   ##   ",
    "   ##   ",
    " ###### ",
    " ###### ",
    "########",
    "###  ###",
    "###  ###",
    "########",
    "########",
    "########",
    "########",
    " #    # ",
    "# #  # #",
    "# #  # #"
]
arcade_scene2DoF["invader"] = unary_union([
    box(10 + c, 22 - r, 11 + c, 23 - r)
    for r, row in enumerate(pattern)
    for c, char in enumerate(row) if char == "#"
])
arcade_scene2DoF["asteroid1"] = Polygon([(7, 23), (3, 25), (5, 27), (11, 26), (7, 25)])
arcade_scene2DoF["asteroid2"] = Polygon([(2, 6), (4, 7), (8, 3), (7, 2), (3, 1)])
arcade_scene2DoF["asteroid3"] = Polygon([(24, 15), (26, 13), (29, 16), (28, 21), (23, 23)])
description_arcade = "Space-Invader mit Asteroiden"
goalList_arcade = [[5, 20], [25, 27], [5,10], [10,5]]
benchList.append(Benchmark(
    "arcade_2DoF",
    CollisionChecker(arcade_scene2DoF, limits=([0, 30], [0, 30])),
    [[25, 5]],
    goalList_arcade,
    description_arcade,
    2
))

# ============================================================================
# 3-DoF SHAPE-ROBOTER BENCHMARKS (2 Stück)
# ============================================================================

# --- Benchmark 3: Spinner with Core (3-DoF Shape) - MITTEL ---
Spinner_with_Core3DoF = dict()
Spinner_with_Core3DoF["obs1"] = (
    Point(15, 15).buffer(10)
    .difference(Point(15, 15).buffer(8))
    .difference(box(11, 2, 19, 28))  # Breiterer vertikaler Ausschnitt (von 13-17 auf 11-19)
    .difference(box(2, 11, 28, 19))  # Breiterer horizontaler Ausschnitt (von 13-17 auf 11-19)
)
Spinner_with_Core3DoF["obs2"] = Point(15, 15).buffer(3)
description_spinner = "Ring-Struktur mit weiten Öffnungen - Rotation erforderlich"
robot_shape_spinner = (
    Point(1.5, 1.5).buffer(1.8)
    .difference(Point(1.5, 1.5).buffer(0.9))
    .difference(Point(3.0, 1.5).buffer(2))
)
goalList_spinner = [[5, 15, 270], [15, 25, 180], [25, 15, 90]]
benchList.append(Benchmark(
    "spinner_3DoF",
    CollisionChecker3D(Spinner_with_Core3DoF, robot_shape_spinner, limits=[[0, 30], [0, 30], [0, 360]]),
    [[15, 5, 0]],
    goalList_spinner,
    description_spinner,
    2
))

# --- Benchmark 4: Grid Scene (3-DoF Shape) - SCHWER ---
grid_scene3DoF = dict()
for r in range(4):  # Reduziert von 8 auf 4
    for c in range(4):  # Reduziert von 8 auf 4
        grid_scene3DoF[f"rect_{r}_{c}"] = box(c * 10, r * 10, c * 10 + 1, r * 10 + 1)  # Abstand 10 statt 5
description_grid = "Gitter mit weitem Abstand - Passagen für L-Roboter"
robot_shape_L = Polygon([(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)])
goalList_grid = [[27, 7, 180], [17, 28, 90], [7, 17, 270]]
benchList.append(Benchmark(
    "grid_3DoF",
    CollisionChecker3D(grid_scene3DoF, robot_shape_L, limits=[[0, 40], [0, 40], [0, 360]]),
    [[2.5, 2.5, 0]],
    goalList_grid,
    description_grid,
    3
))

# ============================================================================
# HILFSFUNKTIONEN
# ============================================================================

def getBenchmarkByName(name):
    """Gibt einen Benchmark anhand seines Namens zurück."""
    for bench in benchList:
        if bench.name == name:
            return bench
    return None

def getBenchmarks2DoF():
    """Gibt nur 2-DoF Benchmarks zurück."""
    return [b for b in benchList if "2DoF" in b.name]

def getBenchmarks3DoF():
    """Gibt nur 3-DoF Benchmarks zurück."""
    return [b for b in benchList if "3DoF" in b.name]

def printBenchmarkOverview():
    """Gibt eine Übersicht aller Benchmarks aus."""
    print("=" * 50)
    print("BENCHMARK ÜBERSICHT")
    print("=" * 50)
    for i, bench in enumerate(benchList, 1):
        dof = "3-DoF" if "3DoF" in bench.name else "2-DoF"
        print(f"{i}. {bench.name} ({dof}, Level {bench.level})")
        print(f"   Start: {bench.startList[0]}")
        print(f"   Goals: {bench.goalList}")
    print("=" * 50)


if __name__ == "__main__":
    printBenchmarkOverview()
