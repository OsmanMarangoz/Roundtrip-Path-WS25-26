import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
from shapely import plotting as splot
from shapely.affinity import translate, rotate
from shapely.geometry import Polygon
from shapely import affinity

def visibilityPRMVisualizeRound(planner, solution, ax=None, nodeSize=300):
    """
    Angepasste Visualisierung als Grafik für Roundtrip-Pfade-Variante im Design des Standard-VisibilityPRM.
    
    Farbschema (Original):
    - Pfad: Grün (Breit)
    - Start: Hellgrün (#00dd00)
    - Ziel: Rot (#dd0000)
    """

    graph = planner.graph
    collChecker = planner._collisionChecker
    robot_shape = collChecker.robot_shape
    
    # Positionen und Farben aus Graph
    pos = nx.get_node_attributes(graph, 'pos')
    colors = nx.get_node_attributes(graph, 'color')
    node_types = nx.get_node_attributes(graph, 'nodeType')
    
    if not pos: return

    # Projektion auf 2D für NetworkX Zeichenfunktionen
    pos_2d = {node: (coords[0], coords[1]) for node, coords in pos.items()}
    
    if ax: 
        # Hindernisse zeichnen
        collChecker.drawObstacles(ax)
        # Limits setzen
        limits = collChecker.getEnvironmentLimits()
        ax.set_xlim(limits[0][0], limits[0][1])
        ax.set_ylim(limits[1][0], limits[1][1])

    ### 1. ROADMAP (Hintergrund) ###
    
    # A) Kanten des Graphen (dünn & grau)
    nx.draw_networkx_edges(graph, pos_2d, ax=ax, alpha=0.2, edge_color='grey')

    # B) Knoten sortieren nach Typ
    guards = []
    connections = []
    others = []
    
    for n in graph.nodes():
        # Start/Goal nicht beachtet (eigene Frabe)
        if str(n) == "start" or str(n).startswith("goal"):
            continue
            
        ntype = node_types.get(n, 'Unknown')
        if ntype == 'Guard':
            guards.append(n)
        elif ntype in ['Connection', 'Expansion']:
            connections.append(n)
        else:
            others.append(n)

    # C) Kanten erstellen
    if connections:
        # Farbe aus Attribut (cyan)
        c_list = [colors.get(n, 'cyan') for n in connections]
        nx.draw_networkx_nodes(graph, pos_2d, ax=ax, nodelist=connections, 
                               node_size=50, node_color=c_list, alpha=0.6, label="Connections")

    # D) Guards zeichnen
    
    if guards:
        my_guard_color = 'blue'
        nx.draw_networkx_nodes(graph, pos_2d, ax=ax, nodelist=guards, 
                               node_size=60, 
                               node_color=my_guard_color,
                               alpha=0.6, 
                               label="Guards")

    ### 2. Lösungspfad visualisieren ###

    if solution and len(solution) > 1:
        path_edges = list(zip(solution, solution[1:]))
        nx.draw_networkx_edges(graph, pos_2d, edgelist=path_edges, alpha=0.8, 
                               edge_color='g', width=5.0, ax=ax, label="Solution Path")

    ### 3. Start/Goals visualisieren ###
    
    # Start: #00dd00, Label "S"
    if "start" in graph.nodes():
        nx.draw_networkx_nodes(graph, pos_2d, nodelist=["start"],
                               node_size=nodeSize, node_color='#00dd00', ax=ax)
        # Label etwas verschoben oder direkt drauf
        ax.text(pos_2d["start"][0], pos_2d["start"][1], "S", 
                color="white", fontweight="bold", ha="center", va="center", zorder=20)

    # Ziele: #dd0000, Label "G" (oder G1, G2...)
    goal_nodes = [n for n in graph.nodes() if str(n).startswith("goal")]
    if goal_nodes:
        nx.draw_networkx_nodes(graph, pos_2d, nodelist=goal_nodes,
                               node_size=nodeSize, node_color='#dd0000', ax=ax)
        
        for g in goal_nodes:
            # Kurzes Label generieren (goal_12 -> G12)
            lbl = "G" + g.split('_')[-1] if '_' in str(g) else "G"
            ax.text(pos_2d[g][0], pos_2d[g][1], lbl, 
                    color="white", fontweight="bold", ha="center", va="center", zorder=20)

    ### 4. Roboter-Schemen einfügen ###

    # Orange für abkrenzung zu grünem Pfad
    if solution and len(solution) > 1:

        # Interpolationsschritte pro Segment -> größer = mehr Roboter-Schemen
        steps_per_segment = 5
        
        # Helper zum Zeichnen des Roboters auf Abruf
        def draw_shape(x, y, theta, alpha):
            rotated = rotate(robot_shape, theta)
            transformed = translate(rotated, x, y)
            try:
                splot.plot_polygon(transformed, ax=ax, add_points=False, 
                                   facecolor='orange', edgecolor='black', alpha=alpha)
            except:
                splot.plot_polygon(transformed, ax=ax, add_points=False, 
                                   color='orange', alpha=alpha)

        for i in range(len(solution) - 1):
            u, v = solution[i], solution[i+1]
            if u not in graph.nodes or v not in graph.nodes: continue
            
            p1 = graph.nodes[u]['pos']
            p2 = graph.nodes[v]['pos']
            
            for s in range(steps_per_segment):
                # Start überspringen (nicht doppelt zeichnen)
                if s == 0: continue
                
                t = s / float(steps_per_segment)
                cur_x = (1-t)*p1[0] + t*p2[0]
                cur_y = (1-t)*p1[1] + t*p2[1]
                cur_th = (1-t)*p1[2] + t*p2[2]
                
                # Alpha für Transparenz
                draw_shape(cur_x, cur_y, cur_th, alpha=0.15)

                

def get_interpolated_trajectory(graph, solution, steps_per_segment=10):
    """
    Berechnet eine Liste von [x, y, theta] Posen basierend auf dem Lösungspfad.
    Zur erstellung und Darstellung von Animationen und Bewegungen
    """
    if not solution or len(solution) < 2:
        return []
        
    traj = []
    for i in range(len(solution) - 1):
        u, v = solution[i], solution[i+1]
        if u not in graph.nodes or v not in graph.nodes: continue
        
        p1 = graph.nodes[u]['pos']
        p2 = graph.nodes[v]['pos']
        
        # Lineare Interpolation zwischen Knoten
        for s in range(steps_per_segment):
            t = s / float(steps_per_segment)
            cur_x = (1-t)*p1[0] + t*p2[0]
            cur_y = (1-t)*p1[1] + t*p2[1]
            cur_th = (1-t)*p1[2] + t*p2[2]
            traj.append([cur_x, cur_y, cur_th])
            
    # Letzten Knoten hinzufügen
    last_node = solution[-1]
    if last_node in graph.nodes:
        traj.append(graph.nodes[last_node]['pos'])
        
    return traj


def animateRoundtrip(planner, solution, steps_per_segment=10, interval=50, save_file=None):
    """
    Erstellt eine Animation des Pfades.
    
    Args:
        planner: Planer-Objekt (Graph & CollisionChecker)
        solution: Liste der Knoten (Pfad)
        steps_per_segment: Interpolationsschritte pro Kante
        interval: Zeit pro Frame in ms
        save_file (str, optional): Pfad zum Speichern (z.B. "animation.mp4" oder "anim.gif").
                                   Wenn None, wird das Animationsobjekt zurückgegeben.
    
    Returns:
        anim: Das Matplotlib Animationsobjekt (nur wenn save_file=None)
    """
    graph = planner.graph
    collChecker = planner._collisionChecker
    robot_shape = collChecker.robot_shape
    
    ### 1. Trajektorie berechnen (Interpolation mit Hilfsfunktion) ###
    trajectory = get_interpolated_trajectory(graph, solution, steps_per_segment)
    if not trajectory:
        print("Keine Trajektorie zum Animieren.")
        return None

    num_frames = len(trajectory)

    ### 2. Plot Setup ###
    # schließen vorherige Plots
    plt.close('all') 

    # Plot erstellen
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.set_title(f"Roundtrip Animation ({num_frames} Frames)")
    env_limits = collChecker.getEnvironmentLimits()
    ax.set_xlim(env_limits[0][0], env_limits[0][1])
    ax.set_ylim(env_limits[1][0], env_limits[1][1])
    ax.grid(True)
    ax.set_aspect('equal') # Gleiches Seitenverhältnis (keine Verzerrung)

    ### 3. Hintergrund zeichnen ###
    collChecker.drawObstacles(ax)
    
    # Graph und Knoten zeichnen
    pos = nx.get_node_attributes(graph, 'pos')
    if pos:
        pos_2d = {n: (c[0], c[1]) for n, c in pos.items()}
        nx.draw_networkx_edges(graph, pos_2d, ax=ax, edge_color='grey', alpha=0.2)
        
        start_nodes = [n for n in graph.nodes if str(n) == "start"]
        goal_nodes = [n for n in graph.nodes if str(n).startswith("goal")]
        
        if start_nodes:
            nx.draw_networkx_nodes(graph, pos_2d, nodelist=start_nodes, node_color='lime', node_size=100, ax=ax, label='Start')
        if goal_nodes:
            nx.draw_networkx_nodes(graph, pos_2d, nodelist=goal_nodes, node_color='blue', node_size=100, ax=ax, label='Ziele')
    
    ax.legend(loc='upper right')

    ### 4. Roboter-Patch Initialisierung ###
    # leeres Polygon, das gefüllt wird
    robot_patch = plt.Polygon([[0,0]], closed=True, facecolor='orange', edgecolor='black', alpha=0.9, zorder=10)
    ax.add_patch(robot_patch)

    ### 5. Update-Funktion ###
    # Aktualisiert den Roboter-Patch für jeden Frame
    def update(frame):
        pose = trajectory[frame]
        
        # Transformation des Roboters (Rotation + Translation)
        rot = affinity.rotate(robot_shape, pose[2], origin='centroid')
        trans = affinity.translate(rot, pose[0], pose[1])
        
        # Koordinaten extrahieren und Patch updaten
        if trans.geom_type == 'Polygon':
            coords = list(trans.exterior.coords)
            robot_patch.set_xy(coords)
        
        return robot_patch,

    ### 6. Animation erstellen ###
    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False)

    ### 7. Export oder Rückgabe ###
    if save_file:
        print(f"Speichere Animation als {save_file}...")
        try:
            if save_file.endswith('.mp4'):
                # Benötigt ffmpeg (meist installiert)
                anim.save(save_file, writer='ffmpeg', fps=30)
            elif save_file.endswith('.gif'):
                # Benötigt imagemagick oder pillow
                anim.save(save_file, writer='pillow', fps=15)
            elif save_file.endswith('.html'):
                # Speichert als Standalone HTML Datei
                with open(save_file, 'w') as f:
                    f.write(anim.to_jshtml())
            print("Speichern erfolgreich.")
        except Exception as e:
            print(f"Fehler beim Speichern: {e}")
        
        plt.close(fig) # Plot schließen nach Speichern
        return None
    else:
        plt.close(fig) 
        return anim