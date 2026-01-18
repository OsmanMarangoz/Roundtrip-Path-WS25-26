# ============================================================================
# ANIMATION-KLASSE MIT ZIEL-VISUALISIERUNG
# ============================================================================
import matplotlib.animation
from IPython.display import HTML, display, Markdown
import numpy as np
import matplotlib.pyplot as plt
import os

class PathAnimator:
    """
    Hilfklasse zur Animation von geplanten Pfaden
    Funktioniert für 2-DoF Punkte und 3-DoF Shape-Roboter.
    Zeigt Start (Grün) und Ziele (Rot) an.
    """

    def __init__(self, benchmark, path, planner_name, planning_time):
        self.benchmark = benchmark
        self.path = path
        self.planner_name = planner_name
        self.planning_time = planning_time
        self.cc = benchmark.collisionChecker

    def _interpolate_segment(self, p_start, p_end, step_size=0.5):
        dist = np.linalg.norm(np.array(p_start[:2]) - np.array(p_end[:2]))
        if dist < 1e-6:
            return np.array([p_start])
        num_steps = max(2, int(np.ceil(dist / step_size)))
        x_interp = np.linspace(p_start[0], p_end[0], num_steps)
        y_interp = np.linspace(p_start[1], p_end[1], num_steps)

        # Falls 3D (x, y, theta), interpoliere Winkel korrekt
        if len(p_start) == 3 and len(p_end) == 3:
            unwrapped = np.unwrap([p_start[2], p_end[2]])
            ang_interp = np.linspace(unwrapped[0], unwrapped[1], num_steps)
            return np.stack([x_interp, y_interp, ang_interp], axis=1)

        return np.stack([x_interp, y_interp], axis=1)

    def _interpolate_path(self, poses, step_size=0.5):
        if len(poses) < 2:
            return np.array(poses)
        interpolated = []
        for i in range(len(poses) - 1):
            seg = self._interpolate_segment(poses[i], poses[i + 1], step_size)
            interpolated.append(seg[:-1])
        interpolated.append([poses[-1]])
        return np.concatenate(interpolated, axis=0)

    def animate(
        self,
        step_size=0.5,
        interval_ms=60,
        repeat=1,
        max_frames=None,
        embed_limit_mb=40,
        save_html_path=None,
        show_inline=True,
    ):
        if not self.path:
            print("Kein Pfad vorhanden!")
            return

        plt.rcParams["animation.embed_limit"] = embed_limit_mb

        # Pfad vorbereiten
        path_3d = [p if len(p) == 3 else [p[0], p[1], 0.0] for p in self.path]
        path_array = np.array(path_3d)
        interpolated_path = self._interpolate_path(path_3d, step_size)

        # Frame-Limitierung (Subsampling)
        if max_frames is not None and len(interpolated_path) > max_frames:
            idx = np.linspace(0, len(interpolated_path) - 1, max_frames).astype(int)
            interpolated_path = interpolated_path[idx]

        # Wiederholungen
        if repeat > 1:
            interpolated_path = np.concatenate([interpolated_path for _ in range(repeat)], axis=0)

        fig, ax = plt.subplots(figsize=(8, 8))
        limits = self.cc.getEnvironmentLimits()
        min_x, max_x = limits[0]
        min_y, max_y = limits[1]

        # Start- und Zielpunkte holen
        start_pos = self.benchmark.startList[0]
        goals = self.benchmark.goalList

        def animate_frame(i):
            ax.clear()
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)

            # 1. Hindernisse zeichnen
            self.cc.drawObstacles(ax)

            # 2. Startpunkt (Grün)
            ax.plot(start_pos[0], start_pos[1], 'go', markersize=10,
                    markeredgecolor='black', label='Start')

            # 3. Zielpunkte (Rot)
            # Wir nutzen enumerate, damit wir das Label nur 1x zur Legende hinzufügen
            for idx, g in enumerate(goals):
                label = 'Goal' if idx == 0 else None
                ax.plot(g[0], g[1], 'ro', markersize=8,
                        markeredgecolor='black', label=label)

            # 4. Pfad-Visualisierung
            # Waypoints (grüne Kreuze)
            ax.plot(path_array[:, 0], path_array[:, 1], 'gx', markersize=6, alpha=0.4, label='Waypoints')
            # Geplanter Gesamtpfad (grau gestrichelt)
            ax.plot(interpolated_path[:, 0], interpolated_path[:, 1], '--', color='gray', alpha=0.4, label='Planned Path')
            # Bereits gefahrener Pfad (blau durchgezogen)
            ax.plot(interpolated_path[:i+1, 0], interpolated_path[:i+1, 1], 'b-', linewidth=2, label='Traveled')

            # 5. Roboter an aktueller Position
            current_pos = interpolated_path[i]
            if hasattr(self.cc, 'drawRobot'):
                self.cc.drawRobot(ax, current_pos)
            else:
                ax.scatter(current_pos[0], current_pos[1], color='purple', s=150, zorder=10,
                           marker='o', edgecolors='black', linewidths=2)

            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{self.benchmark.name} - {self.planner_name}\nFrame {i+1}/{len(interpolated_path)}")

            # Legende (nur beim ersten Frame aufbauen für Performance)
            if i == 0:
                ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

        print(f"Generiere Animation: {len(path_array)} Waypoints → {len(interpolated_path)} Frames...")

        ani = matplotlib.animation.FuncAnimation(
            fig,
            animate_frame,
            frames=len(interpolated_path),
            interval=interval_ms,
        )

        if save_html_path:
            ani.save(save_html_path, writer=matplotlib.animation.HTMLWriter())
            print(f"Gespeichert als: {save_html_path}")

        if show_inline:
            plt.close(fig)
            display(HTML(ani.to_jshtml()))
            print("Animation fertig!")
        else:
            plt.close(fig)
            print("Animation gerendert (kein Inline-Display).")
