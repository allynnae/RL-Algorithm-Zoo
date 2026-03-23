"""
W&B helpers: thin wrappers so the main loop stays clean.
"""

import time
import wandb


# Start a W&B run with consistent naming and config.
def start_wandb_run(algo: str, episodes: int, mode: str):
    return wandb.init(
        project="rl-algorithm-zoo",
        entity="am893120",
        name=f"{algo}-{int(time.time())}",
        config={"algo": algo, "episodes": episodes},
        mode=mode,
        reinit=True,
    )
