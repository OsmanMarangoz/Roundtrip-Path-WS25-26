import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.patches import Polygon as MplPolygon, Circle
import numpy as np
from shapely import plotting as splot
from shapely.affinity import translate, rotate
from shapely.geometry import Polygon
from shapely import affinity

def visibilityPRMVisualizeRound(planner, solution, ax=None, nodeSize=300):
    """
    Angepasste Visualisierung für 2D (Punkt) und 3D (Shape).
    """
    graph = planner.graph
    collChecker = planner._collisionChecker
    
    # Prüfen ob 3D-Shape vorhanden ist
    is_3d = hasattr(collChecker, 'robot_shape')
    robot_shape = getattr(collChecker, 'robot_shape', None)

    # Positionen holen
    pos = nx.get_node_attributes(graph, 'pos')
    colors = nx.get_node_attributes(graph, 'color')
    node_types = nx.get_node_attributes(graph, 'nodeType')

    if not pos: return

    # Projektion auf 2D für NetworkX
    pos_2d = {node: (coords[0], coords[1]) for node, coords in pos.items()}

    if ax:
        collChecker.drawObstacles(ax)
        limits = collChecker.getEnvironmentLimits()
        ax.set_xlim(limits[0][0], limits[0][1])
        ax.set_ylim(limits[1][0], limits[1][1])

    ### 1. ROADMAP (Hintergrund) ###
    nx.draw_networkx_edges(graph, pos_2d, ax=ax, alpha=0.2, edge_color='grey')

    guards = []
    connections = []
    
    for n in graph.nodes():
        if str(n) == "start" or str(n).startswith("goal"): continue
        ntype = node_types.get(n, 'Unknown')
        if ntype == 'Guard': guards.append(n)
        elif ntype in ['Connection', 'Expansion']: connections.append(n)

    if connections:
        c_list = [colors.get(n, 'cyan') for n in connections]
        nx.draw_networkx_nodes(graph, pos_2d, ax=ax, nodelist=connections,
                               node_size=50, node_color=c_list, alpha=0.6, label="Connections")

    if guards:
        nx.draw_networkx_nodes(graph, pos_2d, ax=ax, nodelist=guards,
                               node_size=60, node_color='blue', alpha=0.6, label="Guards")

    ### 2. Lösungspfad visualisieren ###
    if solution and len(solution) > 1:
        path_edges = list(zip(solution, solution[1:]))
        nx.draw_networkx_edges(graph, pos_2d, edgelist=path_edges, alpha=0.8,
                               edge_color='g', width=5.0, ax=ax, label="Solution Path")

    ### 3. Start/Goals visualisieren ###
    if "start" in graph.nodes():
        nx.draw_networkx_nodes(graph, pos_2d, nodelist=["start"],
                               node_size=nodeSize, node_color='#00dd00', ax=ax)
        ax.text(pos_2d["start"][0], pos_2d["start"][1], "S", color="white", fontweight="bold", ha="center", va="center")

    goal_nodes = [n for n in graph.nodes() if str(n).startswith("goal")]
    if goal_nodes:
        nx.draw_networkx_nodes(graph, pos_2d, nodelist=goal_nodes,
                               node_size=nodeSize, node_color='#dd0000', ax=ax)
        for g in goal_nodes:
            lbl = "G" + g.split('_')[-1] if '_' in str(g) else "G"
            ax.text(pos_2d[g][0], pos_2d[g][1], lbl, color="white", fontweight="bold", ha="center", va="center")

    ### 4. Roboter-Schemen (NUR 3D) ###
    if is_3d and solution and len(solution) > 1:
        steps_per_segment = 5
        
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
                if s == 0: continue
                t = s / float(steps_per_segment)
                cur_x = (1-t)*p1[0] + t*p2[0]
                cur_y = (1-t)*p1[1] + t*p2[1]
                cur_th = (1-t)*p1[2] + t*p2[2] # Index 2 existiert nur in 3D
                draw_shape(cur_x, cur_y, cur_th, alpha=0.15)


def get_interpolated_trajectory(graph, solution, steps_per_segment=10):
    """ Berechnet interpolierte Pfadpunkte (2D oder 3D kompatibel). """
    if not solution or len(solution) < 2: return []

    # Check Dimension des ersten Punktes
    first_node = solution[0]
    if first_node not in graph.nodes: return []
    dim = len(graph.nodes[first_node]['pos'])

    traj = []
    for i in range(len(solution) - 1):
        u, v = solution[i], solution[i+1]
        if u not in graph.nodes or v not in graph.nodes: continue

        p1 = graph.nodes[u]['pos']
        p2 = graph.nodes[v]['pos']

        for s in range(steps_per_segment):
            t = s / float(steps_per_segment)
            cur_x = (1-t)*p1[0] + t*p2[0]
            cur_y = (1-t)*p1[1] + t*p2[1]
            
            if dim >= 3:
                cur_th = (1-t)*p1[2] + t*p2[2]
                traj.append([cur_x, cur_y, cur_th])
            else:
                traj.append([cur_x, cur_y])

    # Letzten Punkt anhängen
    if solution[-1] in graph.nodes:
        traj.append(graph.nodes[solution[-1]]['pos'])

    return traj


def animateRoundtrip(planner, solution, steps_per_segment=10, interval=50, save_file=None):
    """ Erstellt Animation (2D und 3D kompatibel). """
    graph = planner.graph
    collChecker = planner._collisionChecker
    
    # Check 3D
    is_3d = hasattr(collChecker, 'robot_shape')
    robot_shape = getattr(collChecker, 'robot_shape', None)

    trajectory = get_interpolated_trajectory(graph, solution, steps_per_segment)
    if not trajectory:
        print("Keine Trajektorie.")
        return None

    num_frames = len(trajectory)
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 8))
    
    limits = collChecker.getEnvironmentLimits()
    ax.set_xlim(limits[0][0], limits[0][1])
    ax.set_ylim(limits[1][0], limits[1][1])
    ax.set_aspect('equal')
    ax.grid(True)

    collChecker.drawObstacles(ax)
    
    # Graph Hintergrund
    pos = nx.get_node_attributes(graph, 'pos')
    if pos:
        pos_2d = {n: (c[0], c[1]) for n, c in pos.items()}
        nx.draw_networkx_edges(graph, pos_2d, ax=ax, edge_color='grey', alpha=0.2)
        start_nodes = [n for n in graph.nodes if str(n) == "start"]
        goal_nodes = [n for n in graph.nodes if str(n).startswith("goal")]
        if start_nodes: nx.draw_networkx_nodes(graph, pos_2d, nodelist=start_nodes, node_color='lime', node_size=100, ax=ax)
        if goal_nodes: nx.draw_networkx_nodes(graph, pos_2d, nodelist=goal_nodes, node_color='blue', node_size=100, ax=ax)

    # Roboter Patch initialisieren
    if is_3d:
        # Polygon für 3D Shape
        robot_patch = plt.Polygon([[0,0]], closed=True, facecolor='orange', edgecolor='black', alpha=0.9, zorder=10)
        ax.add_patch(robot_patch)
    else:
        # Kreis für 2D Punkt
        robot_patch = plt.Circle((0,0), radius=0.5, facecolor='orange', edgecolor='black', zorder=10)
        ax.add_patch(robot_patch)

    def update(frame):
        pose = trajectory[frame]
        
        if is_3d:
            # 3D Update (Rotation + Translation)
            rot = affinity.rotate(robot_shape, pose[2], origin='centroid')
            trans = affinity.translate(rot, pose[0], pose[1])
            if trans.geom_type == 'Polygon':
                robot_patch.set_xy(list(trans.exterior.coords))
        else:
            # 2D Update (Nur Position)
            robot_patch.center = (pose[0], pose[1])

        return robot_patch,

    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False)

    if save_file:
        try:
            if save_file.endswith('.mp4'): anim.save(save_file, writer='ffmpeg', fps=30)
            elif save_file.endswith('.gif'): anim.save(save_file, writer='pillow', fps=15)
            elif save_file.endswith('.html'): 
                with open(save_file, 'w') as f: f.write(anim.to_jshtml())
            print(f"Gespeichert: {save_file}")
        except Exception as e: print(f"Fehler beim Speichern: {e}")
        plt.close(fig)
        return None
    else:
        plt.close(fig)
        return anim