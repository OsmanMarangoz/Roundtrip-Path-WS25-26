
from IPVisibilityPRM import VisibilityStatsHandler, VisPRM
import networkx as nx
from scipy.spatial import cKDTree
from IPPerfMonitor import IPPerfMonitor
import itertools
import math

class VisPRMRound(VisPRM):

    def __init__(self, _collChecker, _statsHandler = None):
        super(VisPRMRound, self).__init__(_collChecker)
        self.graph = nx.Graph()
        self.statsHandler = VisibilityStatsHandler() # not yet fully customizable (s. parameters of constructors)
                
    def _isVisible(self, pos, guardPos):
        return not self._collisionChecker.lineInCollision(pos, guardPos)

    def _getDist(self, pos1, pos2):
        return math.dist(pos1, pos2)   

    @IPPerfMonitor
    def planPath(self, startList, goalsList, config):
        """
        Args:
            startList (array): start position in planning space
            goalsList (array) : goal positions in planning space
            config (dict): dictionary with configuration options
        """
        print("\n--- PLAN PATH DEBUG ---")
        # 0. Reset
        self.graph.clear()
        
        # 1. Start und Ziel prüfen
        checkedStartList, checkedGoalList = self._checkStartGoal(startList, goalsList)
        
        # 2. Roadmap lernen
        self._learnRoadmap(config["ntry"])

        # 3. Gewichte zu den Kanten der Roadmap hinzufügen
        for u, v in self.graph.edges():
            p1 = self.graph.nodes[u]['pos']
            p2 = self.graph.nodes[v]['pos']
            dist = self._getDist(p1, p2)
            self.graph[u][v]['weight'] = dist

        # 4. Verbindung von Start und Zielen zur Roadmap finden
        posList = nx.get_node_attributes(self.graph, 'pos')
        roadmap_keys = list(posList.keys())
        roadmap_values = list(posList.values())
        
        if not roadmap_values:
             print("DEBUG: Roadmap ist leer!")
             return []

        # --- SCALING (Wichtig für 3D) ---
        SCALE_FACTOR_THETA = 0.05 
        roadmap_values_scaled = []
        for val in roadmap_values:
            roadmap_values_scaled.append([val[0], val[1], val[2] * SCALE_FACTOR_THETA])
            
        kdTree = cKDTree(roadmap_values_scaled)
        
        def connect_node(pos_raw, name, node_type, color):
            pos_scaled = [pos_raw[0], pos_raw[1], pos_raw[2] * SCALE_FACTOR_THETA]
            # Suche mehr Nachbarn (k=30)
            result = kdTree.query(pos_scaled, k=30) 
            indices = result[1] if hasattr(result[1], '__iter__') else [result[1]]

            for nodeIndex in indices:
                if nodeIndex >= len(roadmap_keys): continue
                targetNode = roadmap_keys[nodeIndex]
                targetPos = roadmap_values[nodeIndex] 

                if not self._collisionChecker.lineInCollision(pos_raw, targetPos):
                    self.graph.add_node(name, pos=pos_raw, color=color, nodeType=node_type)
                    d = self._getDist(pos_raw, targetPos)
                    self.graph.add_edge(name, targetNode, weight=d)
                    return True
            return False

        # Start verbinden
        if not connect_node(checkedStartList[0], "start", "Start", "lightgreen"):
            print("DEBUG: Start konnte nicht verbunden werden.")
            return []

        # Ziele verbinden
        connected_goals = []
        for i, goalPos in enumerate(checkedGoalList):
            goalName = f"goal_{i}"
            if connect_node(goalPos, goalName, "Goal", "blue"):
                connected_goals.append(goalName)
            else:
                print(f"DEBUG: {goalName} konnte nicht verbunden werden.")

        if not connected_goals:
            print("DEBUG: Keine Ziele verbunden.")
            return []

        print(f"DEBUG: Verbunden: Start und {len(connected_goals)} Ziele.")

        # 5. Distanzmatrix erstellen
        try:
            dismatrix = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight='weight'))
        except Exception as e:
            print(f"DEBUG: Fehler bei Distanzmatrix: {e}")
            return []

        # 6. TSP-Graph erstellen
        stations = ["start"] + connected_goals
        tsp_graph = nx.Graph()

        for u, v in itertools.combinations(stations, 2):
            if u in dismatrix and v in dismatrix[u]:
                weight = dismatrix[u][v]
                tsp_graph.add_edge(u, v, weight=weight)
            else:
                print(f"DEBUG: Kein Pfad zwischen {u} und {v}")

        print(f"DEBUG: TSP Graph Knoten: {tsp_graph.number_of_nodes()}, Kanten: {tsp_graph.number_of_edges()}")

        # 7. TSP lösen
        best_order = []
        if tsp_graph.number_of_nodes() > 1:
            try:
                # Versuch 1: NetworkX Approximation
                best_order = nx.approximation.traveling_salesman_problem(tsp_graph, weight='weight', cycle=True)
                print("DEBUG: TSP Solver erfolgreich.")
            except Exception as e:
                print(f"DEBUG: TSP Solver fehlgeschlagen ({e}). Nutze Fallback.")
                # Fallback: Einfache Reihenfolge start -> g0 -> g1 -> ... -> start
                best_order = stations + ["start"]
        else:
            print("DEBUG: Zu wenige Knoten für TSP.")
            return []

        print(f"DEBUG: Best Order: {best_order}")

        # 8. Vollen Pfad rekonstruieren
        full_path = []
        for i in range(len(best_order) - 1):
            try:
                segment = nx.shortest_path(self.graph, best_order[i], best_order[i + 1], weight='weight')
                if i > 0:
                    full_path.extend(segment[1:]) 
                else:
                    full_path.extend(segment)
            except nx.NetworkXNoPath:
                print(f"DEBUG: CRITICAL - Kein Pfad im Graphen zwischen {best_order[i]} und {best_order[i+1]}!")
                pass 

        print(f"DEBUG: Finaler Pfad Länge: {len(full_path)}")
        return full_path