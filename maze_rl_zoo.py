"""
Main runner: wires the pygame UI to the four RL algorithms and Weights & Biases logging.
"""

import argparse
import os
import random
import time
from collections import deque

import wandb

from algorithms import (
    A2CAgent,
    DecisionTransformerAgent,
    QLearningAgent,
    ReinforceAgent,
    Transition,
    to_tensor,
    warmup_buffer,
)
from ui import ALGORITHMS, MazeEnv, UIState
from wandb_utils import start_wandb_run


# Ensure reproducibility across numpy, random, and torch.
def set_seed(seed: int) -> None:
    random.seed(seed)


# Parse command-line arguments (kept minimal since most controls live in the UI).
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL Algorithm Zoo on a tiny maze.")
    parser.add_argument("--algo", type=str, choices=[a[0] for a in ALGORITHMS], default="qlearning")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# Main loop: handles events, training steps, rendering, and W&B logging.
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    algo_idx = next(i for i, a in enumerate(ALGORITHMS) if a[0] == args.algo)
    ui_state = UIState(
        algo_idx=algo_idx,
        algo_label=ALGORITHMS[algo_idx][1],
        algo_tag=ALGORITHMS[algo_idx][2],
        episodes_target=args.episodes,
        wandb_mode=os.environ.get("WANDB_MODE", "online"),
        total_episodes=args.episodes,
        speed_ms=120,
    )

    env = MazeEnv(render=True)
    env.render_sleep = ui_state.speed_ms / 1000.0
    env.ui_state = ui_state

    agent = None
    recent_states = deque(maxlen=10)
    log_probs: list = []
    rewards: list = []
    episodes_done = 0
    episode_active = False
    state = env.reset()
    recent_states.append(state)
    wandb_run = None
    clock = env.clock
    running_loop = True

    while running_loop:
        for event in env.poll_events():
            if event.type == env.pygame.QUIT:
                running_loop = False
            elif env.handle_ui_event(event, ui_state):
                # UI handler toggled something; no-op here.
                pass
            else:
                # Manual handling for training-related clicks.
                if event.type == env.pygame.MOUSEBUTTONDOWN:
                    if env.hit("start", event.pos):
                        if not ui_state.running:
                            episodes_done = 0
                            ui_state.running = True
                            ui_state.status = "running"
                            ui_state.episode = 0
                            ui_state.step = 0
                            ui_state.reward = 0.0
                            if wandb_run:
                                wandb_run.finish()
                            wandb_run = start_wandb_run(
                                algo=ALGORITHMS[ui_state.algo_idx][0],
                                episodes=ui_state.episodes_target,
                                mode=ui_state.wandb_mode,
                            )
                            state_dim = env.state_dim
                            key = ALGORITHMS[ui_state.algo_idx][0]
                            if key == "qlearning":
                                agent = QLearningAgent(state_dim)
                            elif key == "reinforce":
                                agent = ReinforceAgent(state_dim)
                                log_probs, rewards = [], []
                            elif key == "a2c":
                                agent = A2CAgent(state_dim)
                            else:
                                agent = DecisionTransformerAgent(state_dim)
                                recent_states = deque(maxlen=agent.seq_len)
                                warmup_buffer(env, agent, episodes=20)
                            episode_active = False
                        else:
                            ui_state.running = False
                            ui_state.status = "stopped"
                            if wandb_run:
                                wandb_run.finish()
                                wandb_run = None
                    if env.hit("reset", event.pos):
                        if wandb_run:
                            wandb_run.finish()
                            wandb_run = None
                        ui_state.running = False
                        env.reset()
                        ui_state.status = "idle"
                        ui_state.reward = 0.0
                        ui_state.step = 0
                        ui_state.episode = 0
                        episodes_done = 0
                        episode_active = False

        if ui_state.running:
            env.render_sleep = ui_state.speed_ms / 1000.0
            if not episode_active:
                if episodes_done >= ui_state.episodes_target:
                    ui_state.running = False
                    ui_state.status = "done"
                    if wandb_run:
                        wandb_run.finish()
                        wandb_run = None
                    env.render()
                    continue
                state = env.reset()
                recent_states = deque(maxlen=10)
                recent_states.append(state)
                ui_state.episode = episodes_done + 1
                ui_state.step = 0
                ui_state.reward = 0.0
                episode_active = True

            key = ALGORITHMS[ui_state.algo_idx][0]
            if key == "qlearning":
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state
            elif key == "reinforce":
                action, log_p = agent.select_action(to_tensor(state))
                next_state, reward, done, _ = env.step(action)
                log_probs.append(log_p)
                rewards.append(reward)
                state = next_state
            elif key == "a2c":
                action, log_prob, value = agent.select_action(to_tensor(state))
                next_state, reward, done, _ = env.step(action)
                next_value = agent.value(to_tensor(next_state).unsqueeze(0)).detach()
                agent.update(log_prob, value, reward, next_value, done)
                state = next_state
            else:
                action = agent.select_action(recent_states)
                next_state, reward, done, _ = env.step(action)
                agent.store(Transition(state, action, reward, next_state, done))
                recent_states.append(next_state)
                state = next_state

            ui_state.step += 1
            ui_state.reward += reward

            if done or env.steps >= env.max_steps:
                episodes_done += 1
                episode_active = False
                if wandb_run:
                    if key == "reinforce":
                        loss = agent.update_policy(log_probs, rewards)
                        log_probs, rewards = [], []
                        wandb.log({"episode": episodes_done, "reward": ui_state.reward, "loss": loss, "algo": key})
                    elif key == "a2c":
                        wandb.log({"episode": episodes_done, "reward": ui_state.reward, "algo": key})
                    elif key == "qlearning":
                        wandb.log({"episode": episodes_done, "reward": ui_state.reward, "algo": key})
                    else:
                        train_loss = agent.train_step(batch_size=32)
                        wandb.log({"episode": episodes_done, "reward": ui_state.reward, "loss": train_loss, "algo": key})
                ui_state.step = 0
                ui_state.reward = 0.0

        env.ui_state = ui_state
        env.render()
        clock.tick(60)

    env.close()
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
