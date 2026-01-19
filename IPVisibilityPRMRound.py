from IPVisibilityPRM import VisibilityStatsHandler, VisPRM
import networkx as nx
from scipy.spatial import cKDTree
from IPPerfMonitor import IPPerfMonitor
import math
import numpy as np

class VisPRMRound(VisPRM):
    """
    Klasse zur Implementierung eines Roundtrip-Pfades auf Grundlage des VisPRMs.
    Erbt von VisPRM und nutzt die Roadmap-Erstellung der Ursprungsklasse.
    Kompatibel mit 2D (Punkt) und 3D (Shape) Robotern.
    """

    def __init__(self, _collChecker, _statsHandler=None):
        super(VisPRMRound, self).__init__(_collChecker)
        if self.graph is None:
            self.graph = nx.Graph()
        self.statsHandler = VisibilityStatsHandler()

    def _isVisible(self, pos, guardPos):
        return not self._collisionChecker.lineInCollision(pos, guardPos)

    def _getDist(self, pos1, pos2):
        return math.dist(pos1, pos2)

    @IPPerfMonitor
    def planPath(self, startList, goalsList, config):
        print("\n--- PLAN PATH ROUNDTRIP (VisPRM Optimized) ---")
        
        # 0. Reset & Dimension prüfen
        self.graph.clear()
        dim = self._collisionChecker.getDim() #Unterscheidung 2D/3D

        # 1. Start und Ziel prüfen
        checkedStartList, checkedGoalList = self._checkStartGoal(startList, goalsList)

        # 2. Graph erstellen (Original VisPRM Logik)
        self._learnRoadmap(config["ntry"])

        # 3. Gewichte berechnen
        for u, v in self.graph.edges():
            if 'weight' not in self.graph[u][v]:
                p1 = self.graph.nodes[u]['pos']
                p2 = self.graph.nodes[v]['pos']
                self.graph[u][v]['weight'] = self._getDist(p1, p2)

        # 4. Start/Ziele verbinden
        posList = nx.get_node_attributes(self.graph, 'pos')
        if not posList:
             print("DEBUG: Roadmap leer!")
             return []

        roadmap_keys = list(posList.keys())
        roadmap_values = list(posList.values())

        SCALE_FACTOR_THETA = 0.05
        
        # Skalieren wenn 3D (x,y,theta) (Winkel in Verhältnis setzen)
        if dim >= 3:
            roadmap_values_scaled = []
            for v in roadmap_values:
                # theta skalieren
                new_v = list(v)
                new_v[2] = new_v[2] * SCALE_FACTOR_THETA
                roadmap_values_scaled.append(new_v)
        else:
            # 2D Fall: Keine Skalierung nötig
            roadmap_values_scaled = roadmap_values

        kdTree = cKDTree(roadmap_values_scaled)

        def connect_node(pos_raw, name, node_type, color):
            # Query-Punkt an Dimension anpassen
            if dim >= 3:
                pos_query = list(pos_raw)
                pos_query[2] = pos_query[2] * SCALE_FACTOR_THETA
            else:
                pos_query = pos_raw

            dist, indices = kdTree.query(pos_query, k=min(30, len(roadmap_values)))
            if not isinstance(indices, (list, np.ndarray)): indices = [indices]

            connected = False
            for nodeIndex in indices:
                if nodeIndex >= len(roadmap_keys): continue
                targetNode = roadmap_keys[nodeIndex]
                targetPos = roadmap_values[nodeIndex]

                if not self._collisionChecker.lineInCollision(pos_raw, targetPos):
                    self.graph.add_node(name, pos=pos_raw, color=color, nodeType=node_type)
                    d = self._getDist(pos_raw, targetPos)
                    self.graph.add_edge(name, targetNode, weight=d)
                    connected = True
                    break
            return connected

        # Start verbinden
        if not connect_node(checkedStartList[0], "start", "Start", "lightgreen"):
            print("DEBUG: Start connect failed.")
            return []

        # Ziele verbinden
        connected_goals = []
        for i, goalPos in enumerate(checkedGoalList):
            goalName = f"goal_{i+1}"
            if connect_node(goalPos, goalName, "Goal", "blue"):
                connected_goals.append(goalName)
            else:
                print(f"DEBUG: {goalName} connect failed.")

        if not connected_goals:
            return []

        # 5. TSP Graph erstellen
        stations = ["start"] + connected_goals
        tsp_graph = nx.Graph()

        for startnode in stations:
            if startnode not in self.graph: continue
            try:
                # Dijkstra zu allen anderen Knoten
                lengths = nx.single_source_dijkstra_path_length(self.graph, startnode, weight='weight')
                for target_node in stations:
                    if startnode != target_node and target_node in lengths:
                          tsp_graph.add_edge(startnode, target_node, weight=lengths[target_node])
            except Exception: pass

        if tsp_graph.number_of_nodes() < len(stations):
            print("WARNUNG: Inseln im Graph (nicht alle Ziele erreichbar).")

        # 6. TSP Lösen
        if tsp_graph.number_of_nodes() > 1:
            try:
                best_order = nx.approximation.traveling_salesman_problem(tsp_graph, weight='weight', cycle=True)
                # Reihenfolge rotieren, sodass 'start' am Anfang steht
                if "start" in best_order:
                    idx = best_order.index("start")
                    best_order = best_order[idx:] + best_order[:idx]
                    best_order.append("start") # Zyklus schließen
            except:
                return []
        else:
            return []

        # 7. Pfad rekonstruieren (als Koordinaten-Liste zurückgeben!)
        full_path_nodes = []
        for i in range(len(best_order) - 1):
            try:
                segment = nx.shortest_path(self.graph, best_order[i], best_order[i + 1], weight='weight')
                if i > 0: full_path_nodes.extend(segment[1:])
                else: full_path_nodes.extend(segment)
            except nx.NetworkXNoPath: pass

        # Umwandeln von Knoten-IDs in [x, y, (theta)] Koordinaten
        full_path_coords = [self.graph.nodes[n]['pos'] for n in full_path_nodes]
        return full_path_coords