"""End-to-end Spirit-v1.5 test against the local RoboChallenge mock server.

Start the mock server first:
  cd mock_server && python mock_robot_server.py

Then:
  python test_spirit.py --checkpoint /home/Spirit-v1.5 \
      --backbone /home/qwen3vl-4b-small --single_task press_three_buttons

The mock server replays a recorded episode (its robot side ignores incoming
actions), so this validates the API contract end to end: state fetch -> Spirit
inference -> absolute action posting -> shape/queue handling -> latency.
"""

import argparse
import json
import os
import time

# Import Spirit's vendored copy of the official RoboChallenge client: this repo's
# top-level `utils` package collides with Spirit's `utils` package, so mixing
# `robot.*` here with Spirit's model import breaks module resolution.
from spirit_policy import SpiritPolicy, TASK_INFO
from robochallenge.robot.interface_client import InterfaceClient


class GPUClient:
    def __init__(self, policy):
        self.policy = policy

    def infer(self, state):
        return self.policy.run_policy(state)


def wait_for_server(client, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = client.session.get(f"{client.robot_url}/clock-sync", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--single_task", type=str, default="press_three_buttons")
    parser.add_argument("--mock_url", type=str, default="http://127.0.0.1:9098")
    parser.add_argument("--used_chunk_size", type=int, default=60)
    parser.add_argument("--duration", type=float, default=1 / 15)
    parser.add_argument("--max_inferences", type=int, default=3)
    parser.add_argument("--max_seconds", type=float, default=120)
    parser.add_argument("--image_size", type=str, default="320,240", help="width,height")
    parser.add_argument("--output", type=str, default="spirit_mock_result.json")
    args = parser.parse_args()

    info = TASK_INFO[args.single_task]
    image_size = [int(x) for x in args.image_size.split(",")]
    action_type = info["action_type"]
    image_type = (
        ["high", "left_hand", "right_hand"]
        if info["robot_type"] != "UR5"
        else ["left_hand", "right_hand"]
    )

    client = InterfaceClient("test_user", mock=True)
    client.robot_url = args.mock_url.rstrip("/")
    client.job_id = "test_job"
    client.robot_id = "test_robot"
    client.clock_offset = 0.0
    if not wait_for_server(client):
        raise RuntimeError(f"mock server not reachable at {client.robot_url}")

    policy = SpiritPolicy(
        args.checkpoint,
        args.single_task,
        backbone=args.backbone,
        run_id="mock",
        used_chunk_size=args.used_chunk_size,
    )
    gpu_client = GPUClient(policy)

    results = []
    client.start_motion()
    deadline = time.time() + args.max_seconds
    try:
        while time.time() < deadline and len(results) < args.max_inferences:
            state = client.get_state(image_size, image_type, action_type)
            if not state:
                time.sleep(0.5)
                continue
            if state.get("state") != "normal" or state.get("pending_actions", 0) != 0:
                time.sleep(0.2)
                continue
            start = time.perf_counter()
            actions = gpu_client.infer(state)
            latency_ms = (time.perf_counter() - start) * 1000.0
            if not actions:
                raise RuntimeError("policy returned no actions")
            shape = [len(actions), len(actions[0])]
            client.post_actions(actions, args.duration, action_type)
            entry = {
                "index": len(results),
                "action_type": action_type,
                "num_steps": shape[0],
                "action_dim": shape[1],
                "num_actions_posted": len(actions),
                "latency_ms": latency_ms,
                "first_action": [round(float(x), 5) for x in actions[0]],
                "last_action": [round(float(x), 5) for x in actions[-1]],
            }
            results.append(entry)
            print(json.dumps(entry))
    finally:
        client.end_motion()

    summary = {
        "task": args.single_task,
        "robot_type": info["robot_type"],
        "action_type": action_type,
        "image_type": image_type,
        "image_size": image_size,
        "duration": args.duration,
        "inferences": results,
        "mean_latency_ms": sum(r["latency_ms"] for r in results) / len(results) if results else None,
        "passed": len(results) > 0,
    }
    print("SUMMARY:", json.dumps(summary, indent=2))
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"json -> {args.output}")


if __name__ == "__main__":
    main()
