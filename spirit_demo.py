"""Run Spirit-v1.5 against the RoboChallenge online API.

Same flow as demo.py, but with the Spirit policy and the per-task image/action
conventions from Spirit's TASK_INFO table.

Example:
  python spirit_demo.py --user_token <token> --run_id <run_id> \
      --checkpoint /home/Spirit-v1.5 --backbone /home/qwen3vl-4b-small \
      --single_task press_three_buttons
"""

import argparse
import logging

# Use Spirit's vendored copy of the official RoboChallenge client to avoid the
# top-level `utils` namespace clash with Spirit's own `utils` package.
from spirit_policy import SpiritPolicy, TASK_INFO
from robochallenge.robot.interface_client import InterfaceClient
from robochallenge.robot.job_worker import job_loop

logging.basicConfig(
    filename="spirit_rc.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
)


class GPUClient:
    def __init__(self, policy):
        self.policy = policy

    def infer(self, state):
        return self.policy.run_policy(state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_token", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--single_task", type=str, required=True)
    parser.add_argument("--used_chunk_size", type=int, default=60)
    parser.add_argument("--duration", type=float, default=1 / 15)
    args = parser.parse_args()

    info = TASK_INFO[args.single_task]
    image_size = [320, 240]
    image_type = (
        ["high", "left_hand", "right_hand"]
        if info["robot_type"] != "UR5"
        else ["left_hand", "right_hand"]
    )
    action_type = info["action_type"]

    client = InterfaceClient(args.user_token)
    policy = SpiritPolicy(
        args.checkpoint,
        args.single_task,
        backbone=args.backbone,
        run_id=args.run_id,
        used_chunk_size=args.used_chunk_size,
    )
    gpu_client = GPUClient(policy)

    logging.info(
        "task=%s robot=%s action_type=%s images=%s",
        args.single_task,
        info["robot_type"],
        action_type,
        image_type,
    )
    job_loop(client, gpu_client, args.run_id, image_size, image_type, action_type, args.duration)


if __name__ == "__main__":
    main()
