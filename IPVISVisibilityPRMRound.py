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
    Visualisierer für Roundtrip-Pfade.
    Features:
    - 5-fache Interpolation der Zwischenschritte ("Ghost Robots")
    - Farbschema: Start(Grün), Ziel(Blau), Pfad(Orange), Pfeile(Rot)
    - Boundary-Check: Zeichnet nichts außerhalb der 0-25 Grenzen
    """
    graph = planner.graph
    collChecker = planner._collisionChecker
    robot_shape = collChecker.robot_shape # Das Shape-Objekt für Transformationen
    
    # Positionen holen
    pos = nx.get_node_attributes(graph, 'pos')
    if not pos: return

    # 1. Hintergrund-Netz (Graphen-Kanten)
    # Wir projizieren auf 2D für networkx
    pos_2d = {node: (coords[0], coords[1]) for node, coords in pos.items()}
    
    if ax: 
        # Hindernisse
        collChecker.drawObstacles(ax)
        # Limits erzwingen
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 25)

    # Dünnes graues Netz im Hintergrund
    valid_nodes = [n for n in graph.nodes() if n in pos_2d]
    nx.draw_networkx_nodes(graph, pos_2d, ax=ax, nodelist=valid_nodes, 
                           node_size=30, node_color='black', alpha=0.3)
    nx.draw_networkx_edges(graph, pos_2d, ax=ax, alpha=0.2, edge_color='grey')

    # --- HELPER FUNKTION ZUM ROBOTER ZEICHNEN ---
    def draw_robot_shape(x, y, theta, color, alpha, label=None, with_arrow=False):
        # 1. Boundary Check (Sicherstellen, dass wir im Raum bleiben)
        # Wir clippen die Koordinaten visuell auf 0-25
        x = np.clip(x, 0, 25)
        y = np.clip(y, 0, 25)
        
        # 2. Shape transformieren
        rotated = rotate(robot_shape, theta)
        transformed = translate(rotated, x, y)
        
        # 3. Zeichnen
        try:
            # Versuch mit facecolor (neuere Shapely/Matplotlib Versionen)
            splot.plot_polygon(transformed, ax=ax, add_points=False, 
                               facecolor=color, edgecolor='black', alpha=alpha)
        except:
            # Fallback falls facecolor nicht akzeptiert wird (ältere Versionen)
            splot.plot_polygon(transformed, ax=ax, add_points=False, 
                               color=color, alpha=alpha)

        # 4. Optional: Roter Richtungspfeil
        if with_arrow:
            rad = np.radians(theta)
            arrow_len = 1.5
            ax.arrow(x, y, 
                     np.cos(rad)*arrow_len, np.sin(rad)*arrow_len, 
                     head_width=0.4, head_length=0.6, fc='red', ec='red', zorder=10)

        # 5. Optional: Label
        if label:
            ax.text(x, y, label, color=color, weight="bold", fontsize=10, 
                    ha='center', va='center', zorder=20,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # 2. START & ZIELE ZEICHNEN
    
    # Start (Grün)
    if "start" in graph.nodes():
        p = graph.nodes["start"]['pos']
        draw_robot_shape(p[0], p[1], p[2], color='#00cc00', alpha=1.0, label="START")

    # Ziele (Blau)
    goal_nodes = [n for n in graph.nodes() if str(n).startswith("goal")]
    for gn in goal_nodes:
        p = graph.nodes[gn]['pos']
        # Label extrahieren (goal_1 -> G1)
        try: lbl = f"G{gn.split('_')[-1]}"
        except: lbl = "G"
        draw_robot_shape(p[0], p[1], p[2], color='#0000cc', alpha=1.0, label=lbl)


    # 3. PFAD & INTERPOLATION (Ghost Robots)
    if solution and len(solution) > 1:
        
        # A) Die magenta Linie (der Pfad im Graphen)
        path_edges = list(zip(solution, solution[1:]))
        nx.draw_networkx_edges(graph, pos_2d, edgelist=path_edges, alpha=0.8, 
                               edge_color='magenta', width=2.0, ax=ax)
        
        # B) Interpolierte Zwischenschritte (Orange Ghosts)
        steps_per_segment = 5 # Wie gewünscht: Mehr Zwischenschritte
        
        for i in range(len(solution) - 1):
            u = solution[i]
            v = solution[i+1]
            
            if u not in graph.nodes or v not in graph.nodes: continue
            
            p1 = graph.nodes[u]['pos'] # [x, y, theta]
            p2 = graph.nodes[v]['pos']
            
            # Interpolieren
            for s in range(steps_per_segment):
                # Faktor t von 0.0 bis 1.0
                t = s / float(steps_per_segment)
                
                # Lineare Interpolation für x, y, theta
                # (Bei theta ist linear okay, solange wir keine 350->10 Sprünge haben.
                #  Da der Planner lokal arbeitet, ist das meist sicher.)
                cur_x = (1-t)*p1[0] + t*p2[0]
                cur_y = (1-t)*p1[1] + t*p2[1]
                cur_th = (1-t)*p1[2] + t*p2[2]
                
                # Zeichne Ghost (Orange, Transparent, mit rotem Pfeil)
                # Wir zeichnen den Pfeil nur bei jedem 2. Ghost, damit es nicht zu voll wird
                draw_arrow = (s % 2 == 0) 
                draw_robot_shape(cur_x, cur_y, cur_th, color='orange', alpha=0.2, with_arrow=draw_arrow)


def get_interpolated_trajectory(graph, solution, steps_per_segment=10):
    """
    Berechnet eine Liste von [x, y, theta] Posen basierend auf dem Lösungspfad.
    """
    if not solution or len(solution) < 2:
        return []
        
    traj = []
    for i in range(len(solution) - 1):
        u, v = solution[i], solution[i+1]
        if u not in graph.nodes or v not in graph.nodes: continue
        
        p1 = graph.nodes[u]['pos']
        p2 = graph.nodes[v]['pos']
        
        # Lineare Interpolation zwischen zwei Knoten
        for s in range(steps_per_segment):
            t = s / float(steps_per_segment)
            cur_x = (1-t)*p1[0] + t*p2[0]
            cur_y = (1-t)*p1[1] + t*p2[1]
            cur_th = (1-t)*p1[2] + t*p2[2]
            traj.append([cur_x, cur_y, cur_th])
            
    # Den allerletzten Punkt noch hinzufügen
    last_node = solution[-1]
    if last_node in graph.nodes:
        traj.append(graph.nodes[last_node]['pos'])
        
    return traj


def animateRoundtrip(planner, solution, steps_per_segment=10, interval=50, save_file=None):
    """
    Erstellt eine Animation des Pfades.
    
    Args:
        planner: Das Planer-Objekt (enthält Graph & CollisionChecker)
        solution: Die Liste der Knoten (Pfad)
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
    
    # 1. Trajektorie berechnen (Interpolation)
    # HINWEIS: Stellen Sie sicher, dass Sie diese Hilfsfunktion definiert haben!
    trajectory = get_interpolated_trajectory(graph, solution, steps_per_segment)
    if not trajectory:
        print("Keine Trajektorie zum Animieren.")
        return None

    num_frames = len(trajectory)

    # 2. Plot Setup
    # Wir schließen vorherige Plots, um "Geisterbilder" im Notebook zu vermeiden
    plt.close('all') 
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.set_title(f"Roundtrip Animation ({num_frames} Frames)")
    env_limits = collChecker.getEnvironmentLimits()
    ax.set_xlim(env_limits[0][0], env_limits[0][1])
    ax.set_ylim(env_limits[1][0], env_limits[1][1])
    ax.grid(True)
    ax.set_aspect('equal') # Wichtig,damit der Roboter nicht verzerrt wird

    # 3. Hintergrund zeichnen
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

    # 4. Roboter-Patch Initialisierung
    # Wir nutzen ein leeres Polygon, das im Update gefüllt wird
    robot_patch = plt.Polygon([[0,0]], closed=True, facecolor='orange', edgecolor='black', alpha=0.9, zorder=10)
    ax.add_patch(robot_patch)

    # 5. Update-Funktion
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

    # 6. Animation erstellen
    # blit=True ist schneller, macht aber in Notebooks manchmal Probleme. False ist sicherer.
    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False)

    # 7. Export oder Rückgabe
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
        # Wichtig für Notebook Inline-Anzeige: Figure schließen, 
        # damit sie nicht statisch UND als Video angezeigt wird.
        plt.close(fig) 
        return anim
    # (Optional) Rückgabe des Animation-Objekts, falls man es in Notebooks speichern will
    return anim