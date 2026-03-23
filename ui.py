"""
UI definitions: MazeEnv rendering, UI layout, and shared UI state data.
"""

import math
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import pygame

# Algorithm menu entries (key, label, tag)
ALGORITHMS = [
    ("qlearning", "Q-Learning", "Value-Based"),
    ("reinforce", "REINFORCE", "Policy Gradient"),
    ("a2c", "A2C", "Actor-Critic"),
    ("dqn", "DQN", "Value-Based (NN)"),
]


@dataclass
class UIState:
    algo_idx: int
    algo_label: str
    algo_tag: str
    episodes_target: int
    wandb_mode: str
    speed_ms: int = 120
    running: bool = False
    status: str = "idle"
    episode: int = 0
    step: int = 0
    reward: float = 0.0
    total_episodes: int = 0
    dragging_slider: bool = False
    dragging_speed: bool = False
    dropdown_open: bool = False


# Clamp helper for slider math.
def clamp(n, a, b):
    return max(a, min(n, b))


class MazeEnv:
    # Set up grid, pygame window, and drawing helpers.
    def __init__(self, render: bool = True, max_steps: int = 100) -> None:
        self.grid = np.array(
            [
                [0, 0, 0, 1, 0],
                [1, 1, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 1, 2],
            ]
        )
        self.start = (0, 0)
        self.goal = tuple(np.argwhere(self.grid == 2)[0])
        self.agent_pos = list(self.start)
        self.max_steps = max_steps
        self.steps = 0
        self.render_enabled = render
        self.cell_size = 68
        self.sidebar_width = 320
        self.margin = 16
        self.ui_rects: dict[str, pygame.Rect] = {}
        self.ui_state = None
        self.render_sleep = 0.1
        self.visited: set[tuple[int, int]] = set()
        if self.render_enabled:
            pygame.init()
            width = self.sidebar_width + self.margin * 3 + self.grid.shape[1] * self.cell_size
            height = self.margin * 2 + self.grid.shape[0] * self.cell_size + 220
            self.surface = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Maze RL Zoo")
            pygame.font.init()
            self.font = pygame.font.SysFont("arial", 16)
            self.small_font = pygame.font.SysFont("arial", 13)
            self.clock = pygame.time.Clock()
        self.colors = {
            "bg": (17, 22, 31),
            "panel": (26, 32, 44),
            "panel_border": (45, 55, 70),
            "grid": (54, 66, 84),
            "floor": (74, 88, 110),
            "wall": (22, 28, 38),
            "goal": (46, 204, 113),
            "start": (231, 76, 60),
            "agent": (46, 204, 250),
            "visited": (120, 140, 165),
            "text": (190, 200, 215),
            "accent": (46, 204, 113),
        }
        self.pygame = pygame

    # Return dimension of flattened state vector.
    @property
    def state_dim(self) -> int:
        return self._state_vector().shape[0]

    # Fetch pygame events.
    def poll_events(self) -> List[pygame.event.Event]:
        return pygame.event.get()

    # Quick collision check for named UI rectangles.
    def hit(self, key: str, pos) -> bool:
        return key in self.ui_rects and self.ui_rects[key].collidepoint(pos)

    # Reset environment to start.
    def reset(self):
        self.agent_pos = list(self.start)
        self.steps = 0
        self.visited = {self.start}
        return self._state_vector()

    # One environment step.
    def step(self, action: int):
        self.steps += 1
        move_map = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        dr, dc = move_map[action]
        new_r = self.agent_pos[0] + dr
        new_c = self.agent_pos[1] + dc
        reward = -0.01
        if 0 <= new_r < self.grid.shape[0] and 0 <= new_c < self.grid.shape[1] and self.grid[new_r, new_c] != 1:
            self.agent_pos = [new_r, new_c]
        else:
            reward -= 0.5
        self.visited.add(tuple(self.agent_pos))
        done = False
        if tuple(self.agent_pos) == self.goal:
            reward += 1.0
            done = True
        if self.steps >= self.max_steps:
            done = True
        return self._state_vector(), reward, done, {}

    # Flattened state vector.
    def _state_vector(self) -> np.ndarray:
        agent_plane = np.zeros_like(self.grid, dtype=np.float32)
        agent_plane[self.agent_pos[0], self.agent_pos[1]] = 1.0
        stacked = np.stack([self.grid == 0, self.grid == 1, self.grid == 2, agent_plane], axis=0).astype(np.float32)
        return stacked.flatten()

    # Render the maze and sidebar.
    def render(self) -> None:
        if not self.render_enabled:
            return
        self.surface.fill(self.colors["bg"])
        grid_x = self.margin + self.sidebar_width + self.margin
        grid_y = self.margin
        grid_rect = pygame.Rect(grid_x, grid_y, self.grid.shape[1] * self.cell_size, self.grid.shape[0] * self.cell_size)
        pygame.draw.rect(self.surface, self.colors["grid"], grid_rect, border_radius=6)
        for r in range(self.grid.shape[0]):
            for c in range(self.grid.shape[1]):
                rect = pygame.Rect(
                    grid_x + c * self.cell_size + 2,
                    grid_y + r * self.cell_size + 2,
                    self.cell_size - 4,
                    self.cell_size - 4,
                )
                cell_value = self.grid[r, c]
                if cell_value == 1:
                    color = self.colors["wall"]
                elif cell_value == 2:
                    color = self.colors["goal"]
                else:
                    color = self.colors["floor"]
                pygame.draw.rect(self.surface, color, rect, border_radius=6)
                if (r, c) in self.visited and cell_value == 0:
                    pygame.draw.rect(self.surface, self.colors["visited"], rect, border_radius=6)
        start_rect = pygame.Rect(
            grid_x + self.start[1] * self.cell_size + 2,
            grid_y + self.start[0] * self.cell_size + 2,
            self.cell_size - 4,
            self.cell_size - 4,
        )
        pygame.draw.rect(self.surface, self.colors["start"], start_rect, border_radius=6)
        pygame.draw.circle(
            self.surface,
            self.colors["agent"],
            (
                grid_x + self.agent_pos[1] * self.cell_size + self.cell_size // 2,
                grid_y + self.agent_pos[0] * self.cell_size + self.cell_size // 2,
            ),
            self.cell_size // 3,
        )
        legend_y = grid_y + self.grid.shape[0] * self.cell_size + 12
        items = [("Agent", self.colors["agent"]), ("Goal", self.colors["goal"]), ("Start", self.colors["start"]), ("Visited", self.colors["visited"]), ("Wall", self.colors["wall"])]
        x = grid_x
        for label, color in items:
            pygame.draw.rect(self.surface, color, pygame.Rect(x, legend_y, 18, 18), border_radius=4)
            text_surf = self.small_font.render(label, True, self.colors["text"])
            self.surface.blit(text_surf, (x + 22, legend_y + 1))
            x += 22 + text_surf.get_width() + 12
        if self.ui_state:
            self._draw_sidebar()
        pygame.display.flip()
        time.sleep(self.render_sleep)

    # Render sidebar with controls and stats.
    def _draw_sidebar(self) -> None:
        ui = self.ui_state
        self.ui_rects = {}
        x0 = self.margin
        y0 = self.margin
        panel_height = self.grid.shape[0] * self.cell_size + 210
        panel = pygame.Rect(x0, y0, self.sidebar_width, panel_height)
        pygame.draw.rect(self.surface, self.colors["panel"], panel, border_radius=12)
        pygame.draw.rect(self.surface, self.colors["panel_border"], panel, width=2, border_radius=12)

        def text(label, x, y, size=14, bold=False, color=None):
            font = pygame.font.SysFont("arial", size, bold=bold)
            surf = font.render(label, True, color or self.colors["text"])
            self.surface.blit(surf, (x, y))
            return surf.get_width(), surf.get_height()

        cursor_y = y0 + 18
        text("Maze RL Zoo", x0 + 14, cursor_y, size=18, bold=True)
        cursor_y += 28
        text("Algorithm", x0 + 14, cursor_y, size=12, bold=True, color=(140, 150, 165))
        cursor_y += 20
        algo_rect = pygame.Rect(x0 + 14, cursor_y, self.sidebar_width - 32, 38)
        pygame.draw.rect(self.surface, self.colors["grid"], algo_rect, border_radius=6)
        pygame.draw.rect(self.surface, self.colors["panel_border"], algo_rect, width=1, border_radius=6)
        text(ui.algo_label, algo_rect.x + 12, algo_rect.y + 10, size=14, bold=True)
        pygame.draw.polygon(
            self.surface,
            self.colors["text"],
            [(algo_rect.right - 18, algo_rect.y + 16), (algo_rect.right - 10, algo_rect.y + 16), (algo_rect.right - 14, algo_rect.y + 24)],
        )
        self.ui_rects["algo"] = algo_rect
        dropdown_rect = None
        dropdown_items = []
        if ui.dropdown_open:
            drop_height = 36 * len(ALGORITHMS)
            dropdown_rect = pygame.Rect(algo_rect.x, algo_rect.bottom + 4, algo_rect.width, drop_height)
            for i, (_, label, _) in enumerate(ALGORITHMS):
                item_rect = pygame.Rect(dropdown_rect.x, dropdown_rect.y + i * 36, dropdown_rect.width, 36)
                dropdown_items.append((i, item_rect, label))
                self.ui_rects[f"algo_item_{i}"] = item_rect

        cursor_y += 55
        text("Episodes", x0 + 14, cursor_y, size=12, bold=True, color=(140, 150, 165))
        text(str(ui.episodes_target), x0 + self.sidebar_width - 60, cursor_y, size=12, bold=True)
        cursor_y += 18
        track_rect = pygame.Rect(x0 + 14, cursor_y + 10, self.sidebar_width - 32, 6)
        pygame.draw.rect(self.surface, self.colors["panel_border"], track_rect, border_radius=4)
        knob_x = track_rect.x + int((ui.episodes_target - 10) / (500 - 10) * track_rect.width)
        knob_rect = pygame.Rect(knob_x - 8, track_rect.y - 6, 16, 18)
        pygame.draw.rect(self.surface, self.colors["accent"], knob_rect, border_radius=6)
        self.ui_rects["slider_track"] = track_rect
        self.ui_rects["slider_knob"] = knob_rect

        cursor_y += 48
        text("Speed", x0 + 14, cursor_y, size=12, bold=True, color=(140, 150, 165))
        text(f"{ui.speed_ms} ms", x0 + self.sidebar_width - 90, cursor_y, size=12, bold=True)
        cursor_y += 18
        speed_track = pygame.Rect(x0 + 14, cursor_y + 10, self.sidebar_width - 32, 6)
        pygame.draw.rect(self.surface, self.colors["panel_border"], speed_track, border_radius=4)
        rel_speed = (200 - ui.speed_ms) / (200 - 5)
        rel_speed = clamp(rel_speed, 0, 1)
        speed_knob_x = speed_track.x + int(rel_speed * speed_track.width)
        speed_knob = pygame.Rect(speed_knob_x - 8, speed_track.y - 6, 16, 18)
        pygame.draw.rect(self.surface, self.colors["accent"], speed_knob, border_radius=6)
        self.ui_rects["speed_track"] = speed_track
        self.ui_rects["speed_knob"] = speed_knob

        cursor_y += 48
        text("W&B Mode", x0 + 14, cursor_y, size=12, bold=True, color=(140, 150, 165))
        cursor_y += 20
        btn_off = pygame.Rect(x0 + 14, cursor_y, (self.sidebar_width - 40) // 2, 30)
        btn_on = pygame.Rect(btn_off.right + 8, cursor_y, (self.sidebar_width - 40) // 2, 30)
        for rect, mode, key in [(btn_off, "offline", "wandb_offline"), (btn_on, "online", "wandb_online")]:
            active = ui.wandb_mode == mode
            bg = self.colors["accent"] if active else self.colors["grid"]
            pygame.draw.rect(self.surface, bg, rect, border_radius=6)
            pygame.draw.rect(self.surface, self.colors["panel_border"], rect, width=1, border_radius=6)
            text(mode.upper(), rect.x + 10, rect.y + 7, size=12, bold=True, color=(15, 20, 25) if active else self.colors["text"])
            self.ui_rects[key] = rect

        cursor_y += 50
        start_rect = pygame.Rect(x0 + 14, cursor_y, self.sidebar_width - 110, 40)
        if ui.running:
            pygame.draw.rect(self.surface, (220, 90, 70), start_rect, border_radius=8)
            text("STOP", start_rect.x + 14, start_rect.y + 11, size=16, bold=True, color=(20, 20, 25))
        else:
            pygame.draw.rect(self.surface, self.colors["accent"], start_rect, border_radius=8)
            text("START", start_rect.x + 14, start_rect.y + 11, size=16, bold=True, color=(20, 30, 20))
        reset_rect = pygame.Rect(start_rect.right + 8, cursor_y, 70, 40)
        pygame.draw.rect(self.surface, self.colors["grid"], reset_rect, border_radius=8)
        pygame.draw.rect(self.surface, self.colors["panel_border"], reset_rect, width=1, border_radius=8)
        text("RESET", reset_rect.x + 10, reset_rect.y + 11, size=14, bold=True)
        self.ui_rects["start"] = start_rect
        self.ui_rects["reset"] = reset_rect

        cursor_y = reset_rect.bottom + 20
        text("Live Stats", x0 + 14, cursor_y, size=12, bold=True, color=(140, 150, 165))
        cursor_y += 20
        stats = [("Episode", f"{ui.episode} / {ui.total_episodes}"), ("Step", str(ui.step)), ("Reward", f"{ui.reward:.2f}"), ("Status", ui.status)]
        box_w = (self.sidebar_width - 40) // 2
        box_h = 40
        bx, by = x0 + 14, cursor_y
        for i, (title, value) in enumerate(stats):
            rect = pygame.Rect(bx, by, box_w, box_h)
            pygame.draw.rect(self.surface, self.colors["grid"], rect, border_radius=6)
            text(title, rect.x + 8, rect.y + 5, size=11, color=(150, 160, 175))
            text(value, rect.x + 8, rect.y + 20, size=14, bold=True)
            if i % 2 == 1:
                bx = x0 + 14
                by += box_h + 8
            else:
                bx = rect.right + 12
        if ui.dropdown_open and dropdown_rect:
            pygame.draw.rect(self.surface, self.colors["panel"], dropdown_rect, border_radius=8)
            pygame.draw.rect(self.surface, self.colors["panel_border"], dropdown_rect, width=1, border_radius=8)
            for i, item_rect, label in dropdown_items:
                bg = self.colors["grid"] if i != ui.algo_idx else self.colors["floor"]
                top = i == 0
                bottom = i == len(ALGORITHMS) - 1
                border_radius = 8
                pygame.draw.rect(
                    self.surface,
                    bg,
                    item_rect,
                    border_radius=border_radius,
                    border_top_left_radius=border_radius if top else 0,
                    border_top_right_radius=border_radius if top else 0,
                    border_bottom_left_radius=border_radius if bottom else 0,
                    border_bottom_right_radius=border_radius if bottom else 0,
                )
                text(label, item_rect.x + 10, item_rect.y + 8, size=14, bold=True)

    # Handle UI-specific events (dropdown toggles, sliders, W&B buttons).
    def handle_ui_event(self, event, ui_state: UIState) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            if self.hit("algo", (mx, my)) and not ui_state.running:
                ui_state.dropdown_open = not ui_state.dropdown_open
                return True
            for i in range(len(ALGORITHMS)):
                key = f"algo_item_{i}"
                if self.hit(key, (mx, my)) and not ui_state.running:
                    ui_state.algo_idx = i
                    ui_state.algo_label = ALGORITHMS[i][1]
                    ui_state.algo_tag = ALGORITHMS[i][2]
                    ui_state.dropdown_open = False
                    return True
            if self.hit("wandb_offline", (mx, my)) and not ui_state.running:
                ui_state.wandb_mode = "offline"
                return True
            if self.hit("wandb_online", (mx, my)) and not ui_state.running:
                ui_state.wandb_mode = "online"
                return True
            if self.hit("slider_track", (mx, my)) and not ui_state.running:
                ui_state.dragging_slider = True
                return True
            if self.hit("slider_knob", (mx, my)) and not ui_state.running:
                ui_state.dragging_slider = True
                return True
            if self.hit("speed_track", (mx, my)) and not ui_state.running:
                ui_state.dragging_speed = True
                return True
            if self.hit("speed_knob", (mx, my)) and not ui_state.running:
                ui_state.dragging_speed = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            ui_state.dragging_slider = False
            ui_state.dragging_speed = False
        elif event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            if ui_state.dragging_slider and not ui_state.running and "slider_track" in self.ui_rects:
                track = self.ui_rects["slider_track"]
                rel = clamp((mx - track.left) / track.width, 0, 1)
                value = int(10 + rel * (500 - 10))
                value = (value // 10) * 10
                ui_state.episodes_target = value
                ui_state.total_episodes = value
                return True
            if ui_state.dragging_speed and not ui_state.running and "speed_track" in self.ui_rects:
                track = self.ui_rects["speed_track"]
                rel = clamp((mx - track.left) / track.width, 0, 1)
                value = int(200 - rel * (200 - 5))
                ui_state.speed_ms = value
                return True
        if event.type == pygame.MOUSEBUTTONDOWN and ui_state.dropdown_open:
            mx, my = event.pos
            if not (self.hit("algo", (mx, my)) or any(self.hit(f"algo_item_{i}", (mx, my)) for i in range(len(ALGORITHMS)))):
                ui_state.dropdown_open = False
                return True
        return False

    # Close pygame.
    def close(self) -> None:
        if self.render_enabled:
            pygame.quit()
