"""Spirit-v1.5 policy adapter for the RoboChallenge Inference API.

The official RoboChallengeInference client hands `state` dicts (PNG image bytes +
current joint/pose values) to `GPUClient.infer` and expects a list of absolute
per-step actions back. Spirit-v1.5 already ships a vendored copy of this API
client plus `RoboChallengeExecutor`, which implements exactly that mapping, so we
delegate to it here.

Set SPIRIT_ROOT to the Spirit-v1.5 checkout (default /home/spirit-run/spirit-v1.5).
"""

import os
import sys
import types

SPIRIT_ROOT = os.environ.get("SPIRIT_ROOT", "/home/spirit-run/spirit-v1.5")
if SPIRIT_ROOT not in sys.path:
    sys.path.insert(0, SPIRIT_ROOT)

from robochallenge.runner.executor import RoboChallengeExecutor
from robochallenge.runner.task_info import TASK_INFO


class SpiritPolicy:
    def __init__(
        self,
        checkpoint,
        task_name,
        backbone=None,
        run_id="local",
        used_chunk_size=60,
        raw_embodiment_stats_json_path=None,
        use_embodiment_specific_norm=False,
        embodiment_stats_path=None,
    ):
        if task_name not in TASK_INFO:
            raise KeyError(f"unknown task {task_name!r}; known: {sorted(TASK_INFO)}")
        cfg = types.SimpleNamespace(
            single_task=task_name,
            robochallenge_job_id=run_id,
            ckpt_path=checkpoint,
            used_chunk_size=used_chunk_size,
            backbone=backbone,
            raw_embodiment_stats_json_path=raw_embodiment_stats_json_path,
            use_embodiment_specific_norm=use_embodiment_specific_norm,
            embodiment_stats_path=embodiment_stats_path,
        )
        self.task_name = task_name
        self.executor = RoboChallengeExecutor(cfg)

    def run_policy(self, state):
        if isinstance(state, dict) and "job_id" not in state:
            state = dict(state)
            state["job_id"] = "local"
        return self.executor.infer(state)
