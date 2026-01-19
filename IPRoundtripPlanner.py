# coding: utf-8

"""
Roundtrip Path Planner für kollisionsfreie Pfade mit TSP-Optimierung
es wird kollisionsfreien Roundtrip-Pfad geplant, der bei der Startposition beginnt,
alle angegebenen Zielpositionen genau einmal besucht und wieder zur Startposition zurückkehrt.

TSP-Optimierung: NetworkX Christofides-Algorithmus
"""

import math
import networkx as nx
from IPPlanerBase import PlanerBase


class RoundtripPlanner(PlanerBase):
    """
    Roundtrip Planner mit wählbarem Basis-Planer

    Features:
    - Akzeptiert 1 Startposition + mehrere Endpositionen
    - Nutzt beliebigen Planer für kollisionsfreie Punkt-zu-Punkt Pfade
    - Gibt zurück: Vollständiger Roundtrip-Pfad als Liste von Positionen
    """

    def __init__(self, collisionChecker, pairwise_planner):
        """
        Konstruktor

        Args:
            collisionChecker: CollisionChecker Instanz für die Umgebung
            pairwise_planner: Planer-Instanz für paarweise Pfade zwischen Punkten
                             Kann sein: LazyPRM, VisPRM, BasicPRM, RRT, AStar, etc.

        """
        PlanerBase.__init__(self, collisionChecker)
        self._pairwise_planner = pairwise_planner

    def _euclidean_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _find_optimal_order_tsp(self, start, goals):
        """
        Findet optimale Besuchsreihenfolge mit NetworkX Christofides

        Args:
            start: Startposition [x, y]
            goals: Liste von Zielpositionen [[x1,y1], [x2,y2], ...
        Returns:
            list: Goals in optimaler Reihenfolge
        """
        # Alle Punkte: Index 0 = Start, Index 1..n = Goals
        all_points = [start] + goals
        n = len(all_points)

        # Vollständigen Graphen mit Distanzen erstellen
        G = nx.Graph()
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._euclidean_distance(all_points[i], all_points[j])
                G.add_edge(i, j, weight=dist)

        # Christofides TSP
        tsp_tour = nx.approximation.christofides(G, weight='weight')

        # Tour so rotieren, dass Start (Index 0) am Anfang steht
        start_idx = tsp_tour.index(0)
        rotated_tour = tsp_tour[start_idx:] + tsp_tour[1:start_idx + 1]

        # Goals in TSP-Reihenfolge extrahieren (ohne Start-Index 0)
        optimized_goals = [goals[i - 1] for i in rotated_tour[1:-1]]

        # Debug
        total_dist = sum(self._euclidean_distance(all_points[rotated_tour[i]],
                                                   all_points[rotated_tour[i+1]])
                        for i in range(len(rotated_tour) - 1))
        print(f"  TSP-Lösung (Christofides): Approximierte Distanz = {total_dist:.2f}")
        print(f"  Optimierte Reihenfolge: Start → {' → '.join(str(g) for g in optimized_goals)} → Start")

        return optimized_goals

    def planPath(self, startList, goalList, config):
        """
        Plant einen kollisionsfreien Roundtrip-Pfad mit ausgewähltem Planer und optimiert mit TSP

        Args:
            startList: Liste mit einer Startposition [[x, y]]
            goalList: Liste mit mehreren Endpositionen [[x1, y1], [x2, y2], ...]
            config: Dictionary mit Planer-Konfiguration
        """
        print(f"Path Planing gestartet mit {type(self._pairwise_planner).__name__}")
        print(f"\n{'='*60}")

        # Schritt 1: Start und Goals prüfen (kollisionsfrei?) mit planerbase parent methode
        checkedStartList, checkedGoalList = self._checkStartGoal(startList, goalList)

        print(f"Start: {checkedStartList[0]}")
        print(f"Goals: {len(checkedGoalList)} Ziele")
        for i, goal in enumerate(checkedGoalList):
            print(f"  - Goal {i+1}: {goal}")

        # Schritt 2: TSP-Optimierung für beste Reihenfolge
        optimized_goals = self._find_optimal_order_tsp(checkedStartList[0], checkedGoalList)

        # Schritt 3: Alle Punkte in optimierter Reihenfolge sammeln
        all_points = [checkedStartList[0]] + optimized_goals
        full_path = []

        # Config auslesen
        max_retries = config.get("maxRetries", 3)
        retry_until_found = config.get("retry_until_found", False)

        for i in range(len(all_points)):
            # Verbinde aktuellen Punkt mit nächstem (letzter → zurück zu Start)
            current_point = all_points[i]
            next_point = all_points[(i + 1) % len(all_points)] # modulo für Rückkehr zum Start

            print(f"\n→ Plane Segment {i+1}/{len(all_points)}: {current_point} → {next_point}")

            attempts = 0
            segment_path = []

            while True:
                segment_path = self._pairwise_planner.planPath(
                    [current_point],
                    [next_point],
                    config
                )

                if segment_path:
                    # Erfolg -> Raus aus der while-Schleife
                    break

                attempts += 1

                # Prüfen ob wir abbrechen müssen
                if (not retry_until_found) and attempts > max_retries:
                    print(f"FEHLER: Kein Pfad gefunden nach {attempts} Versuchen.")
                    return []

                print(f"  [Retry {attempts}] Kein Pfad gefunden. Starte Planer neu (gleiche Config) und versuche erneut...")

                try:
                    self._pairwise_planner = self._pairwise_planner.__class__(self._collisionChecker)
                except Exception as e:
                    print(f"  Warnung: Konnte Planer nicht resetten ({e}), nutze alte Instanz.")

            print(f"Pfad gefunden: {len(segment_path)} Punkte (im Versuch {attempts + 1})")

            # WICHTIG: Konvertiere SOFORT zu Koordinaten, bevor der Graph gelöscht wird!
            segment_coords = self._convertToCoordinates(segment_path)
            print(f"Konvertiert zu Koordinaten: {len(segment_coords)} Punkte")

            # Füge Segment zum Gesamtpfad hinzu (ohne Duplikate am Übergang)
            if i == 0:
                full_path.extend(segment_coords)
            else:
                # Überspringe ersten Punkt (ist gleich letztem Punkt vom vorherigen Segment)
                full_path.extend(segment_coords[1:])

        print(f"\n{'='*60}")
        print(f" Roundtrip erfolgreich geplant!")
        print(f"  - Gesamtpunkte im Pfad: {len(full_path)}")
        print(f"  - Besuchte Ziele: {len(checkedGoalList)}")
        print(f"  - Start = Ende: {full_path[0] == full_path[-1]}")
        print(f"{'='*60}\n")

        return full_path

    def _convertToCoordinates(self, path):
        """
        Konvertiert einen Pfad von Node-IDs zu Koordinaten
        den verschiedene Planer haben verschiedene Formate die sie zurückgeben.

        Funktioniert mit allen Planern:
        - LazyPRM, BasicPRM, VisPRM, RRT: Node-IDs (int) oder "start"/"goal" (str)
        - AStar: Tuple-Keys wie (0, 0), (1, 1)
        - Bereits Koordinaten: [x, y] Listen

        Args:
            path: Liste von Node-IDs, Tuples oder bereits Koordinaten

        Returns:
            list: Liste von [x, y] Koordinaten
        """
        coordinate_path = []

        for node in path:
            # Fall 1: Bereits eine Koordinate [x, y] oder (x, y)
            if isinstance(node, (list, tuple)) and len(node) == 2:
                # Prüfe ob es numerisch ist (Koordinate)
                try:
                    x, y = float(node[0]), float(node[1])
                    coordinate_path.append([x, y])
                except (ValueError, TypeError):
                    # Kein numerisches Tuple, versuche als Node-ID im Graph
                    if hasattr(self._pairwise_planner, 'graph') and node in self._pairwise_planner.graph.nodes:
                        pos = self._pairwise_planner.graph.nodes[node]['pos']
                        coordinate_path.append(list(pos))
                    else:
                        coordinate_path.append(list(node))

            # Fall 2: Node-ID (int oder string) - hole Position aus Graph
            elif isinstance(node, (int, str)):
                if hasattr(self._pairwise_planner, 'graph') and node in self._pairwise_planner.graph.nodes:
                    pos = self._pairwise_planner.graph.nodes[node]['pos']
                    coordinate_path.append(list(pos))
                else:
                    # Fallback: Wenn nicht im Graph, kann es nicht konvertiert werden
                    print(f"Warning: Node {node} nicht im Graph gefunden")
                    coordinate_path.append([0, 0])  # Dummy-Koordinate

            # Fall 3: Unbekanntes Format
            else:
                print(f"Warning: Unbekanntes Node-Format: {type(node)}")
                coordinate_path.append([0, 0])

        return coordinate_path