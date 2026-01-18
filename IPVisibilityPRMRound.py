
from IPVisibilityPRM import VisibilityStatsHandler, VisPRM
import networkx as nx
from scipy.spatial import cKDTree
from IPPerfMonitor import IPPerfMonitor
import math
import numpy as np

"""
Klasse zur implementierung eines Roundtrip-Pfades auf gunrdlage des VisPRMs
Erbt von VisPRM und nutz die Mulit-Quary-Roadmap erstellung der Ursprungsklasse
"""
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
        print("\n--- PLAN PATH OPTIMIZED ---")
        ### 0. Graph zurücksetzen ###
        self.graph.clear()
        
        ### 1. Start und Ziel prüfen ###
        checkedStartList, checkedGoalList = self._checkStartGoal(startList, goalsList)
        
        ### 2. Graph erstellen ###
        self._learnRoadmap(config["ntry"])

        ### 3. Gewichte aller Kanten berechnen für Dijkstra/TSP ###
        for u, v in self.graph.edges():
            # Berechnung wird nicht durchgeführt wenn bereits vorhanden
            if 'weight' not in self.graph[u][v]:
                p1 = self.graph.nodes[u]['pos']
                p2 = self.graph.nodes[v]['pos']
                self.graph[u][v]['weight'] = self._getDist(p1, p2)

        ### 4. Start/Ziele verbinden ###
        posList = nx.get_node_attributes(self.graph, 'pos')
        if not posList:
             print("DEBUG: Roadmap leer!")
             return []

        roadmap_keys = list(posList.keys())
        roadmap_values = list(posList.values())
        
        # Skalierung für KDTree um Winkelkomponente in verhältnis zu setzen
        SCALE_FACTOR_THETA = 0.05 
        roadmap_values_scaled = [
            [v[0], v[1], v[2] * SCALE_FACTOR_THETA] for v in roadmap_values
        ]
        
        # kdTree zur effizienten distanzbestimmung
        kdTree = cKDTree(roadmap_values_scaled)
        
        # Funktion zum Prüfen einer Verbindung und durchführen dieser
        def connect_node(pos_raw, name, node_type, color):
            pos_scaled = [pos_raw[0], pos_raw[1], pos_raw[2] * SCALE_FACTOR_THETA] # Grad in verhältniss setzen
            
            # Query: k erhöhen, falls anschluss nicht gefunden
            dist, indices = kdTree.query(pos_scaled, k=30) 

            # Falls indices = 1 -> in Liste umwandeln
            if not isinstance(indices, (list, np.ndarray)): indices = [indices]

            # Verbindung prüfen
            for nodeIndex in indices:
                if nodeIndex >= len(roadmap_keys): continue
                targetNode = roadmap_keys[nodeIndex]
                targetPos = roadmap_values[nodeIndex] 

                if not self._collisionChecker.lineInCollision(pos_raw, targetPos):

                    # Verbindung möglich, Knoten hinzufügen und verbinden
                    self.graph.add_node(name, pos=pos_raw, color=color, nodeType=node_type)
                    d = self._getDist(pos_raw, targetPos)
                    self.graph.add_edge(name, targetNode, weight=d)
                    return True
            return False

        # Start verbinden
        if not connect_node(checkedStartList[0], "start", "Start", "lightgreen"):
            print("DEBUG: Start connect failed.")
            return []

        # Ziele verbinden
        connected_goals = []
        for i, goalPos in enumerate(checkedGoalList):
            goalName = f"goal_{i+1}" # Ziele ab 1 aufsteigend benennen
            if connect_node(goalPos, goalName, "Goal", "blue"):
                connected_goals.append(goalName)
            else:
                print(f"DEBUG: {goalName} connect failed.")

        if not connected_goals:
            return []

        ### 5. TSP Graph erstellen ###
        stations = ["start"] + connected_goals

        # Graph für Zielreihenfolge und lösung des TSP
        tsp_graph = nx.Graph()
        
        # Berechnen aller Distanzen zwischen Zielen und Start für tsp_graph
        for startnode in stations:
            if startnode not in self.graph: continue

            try:
                # single_source berechnet Distanzen (auf dem Graph) von startnode zu allen erreichbaren Zielen
                lengths = nx.single_source_dijkstra_path_length(self.graph, startnode, weight='weight') # weight für Dijkstra

                # Kanten zu allen Zielen Zielen im tsp_graph hinzufügen
                for target_node in stations:
                    if startnode != target_node and target_node in lengths:
                         tsp_graph.add_edge(startnode, target_node, weight=lengths[target_node])
            except Exception as e:
                pass # Falls Inseln existieren

        # Prüfen ob für alle Ziele und Start Verbindungen existieren
        if tsp_graph.number_of_nodes() < len(stations):
            print("WARNUNG: Nicht alle Ziele sind untereinander erreichbar (Inseln im Graph).")

        ### 6. TSP Lösen ###
        if tsp_graph.number_of_nodes() > 1:
            try:
                # TSP-Funktion von NetworkX zur lösung, basierend auf Christofides-Algorithmus (Aproximation)
                best_order = nx.approximation.traveling_salesman_problem(tsp_graph, weight='weight', cycle=True)
            except:
                best_order = stations + ["start"] # TSP nicht lösbar, einfache Reihenfolge
        else:
            return []

        ### 7. Pfad rekonstruieren ###
        full_path = []
        for i in range(len(best_order) - 1):
            try:
                # Pfad nach ermittelter Reihenfolge zusammensetzen
                segment = nx.shortest_path(self.graph, best_order[i], best_order[i + 1], weight='weight')
                
                # Erstes Element nur beim ersten Segment hinzufügen um Duplikate zu vermeiden
                if i > 0:
                    full_path.extend(segment[1:]) 
                else:
                    full_path.extend(segment)
            except nx.NetworkXNoPath:
                pass 

        try:
            full_path = nx.shortest_path(self.graph,"start","goal")
        except:
            return []
        return full_path