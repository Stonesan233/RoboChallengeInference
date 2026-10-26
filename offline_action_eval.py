"""Offline action-accuracy scoring for Spirit-v1.5 on RoboChallenge recordings.

The RoboChallenge mock server only replays recorded episodes and ignores incoming
actions, so it cannot produce a task-success score. This script instead measures
how close Spirit's predicted absolute targets are to the robot states actually
recorded in the episode (human teleoperation), at several sampled frames:

  - position MAE (cm)
  - orientation MAE (deg; quaternion/euler converted to angular distance)
  - joint MAE (rad, UR5 / joint control)
  - gripper MAE (native units; not comparable for UR5 due to executor scaling)

A "hold current pose" baseline is computed with the same protocol so the model
numbers have context. This is an offline imitation proxy, not the official
RoboChallenge success rate (which requires submitting a real-robot job).

Usage:
  cd RoboChallengeInference
  SPIRIT_ROOT=/home/spirit-run/spirit-v1.5 python offline_action_eval.py \
      --checkpoint /home/Spirit-v1.5 --backbone /home/qwen3vl-4b-small \
      --num_samples 6 --output offline_action_eval.json
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
MOCK_DIR = os.path.join(HERE, "mock_server")
sys.path.insert(0, MOCK_DIR)
from mock_rc_robot import MockRCRobot
sys.path.remove(MOCK_DIR)

SPIRIT_ROOT = os.environ.get("SPIRIT_ROOT", "/home/spirit-run/spirit-v1.5")
if SPIRIT_ROOT not in sys.path:
    sys.path.insert(0, SPIRIT_ROOT)
if HERE in sys.path:
    sys.path.remove(HERE)

import torch  # noqa: E402
from model.modeling_spirit_vla import SpiritVLAPolicy  # noqa: E402
from robochallenge.runner.executor import _prepare_batch, _post_process_action  # noqa: E402
from robochallenge.runner.task_info import TASK_INFO  # noqa: E402

CASES = [
    ("franka", "20250919/press_three_buttons/data/episode_FFFFFF", "press_three_buttons"),
    ("aloha", "20250919/pour_fries_into_plate/data/episode_001533", "pour_fries_into_plate"),
    ("arx5", "20250919/arrange_paper_cups/data/episode_000772", "arrange_paper_cups"),
    ("ur5", "20250919/shred_scrap_paper/data/episode_000775", "shred_scrap_paper"),
]

IMAGE_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def state_action(robot, action_type):
    if "left" in action_type:
        return robot.left_get_pose() if "pos" in action_type else robot.left_get_joint()
    if "right" in action_type:
        return robot.right_get_pose() if "pos" in action_type else robot.right_get_joint()
    if "pos" in action_type:
        return robot.left_get_pose() + robot.right_get_pose()
    return robot.left_get_joint() + robot.right_get_joint()


def api_images(robot, t, task):
    ret = robot.get_imgs()
    frames = {"left_hand": ret[0][1], "right_hand": ret[1][1]}
    if len(ret) > 2:
        frames["high"] = ret[2][1]
    images = {}
    for cam_key in IMAGE_KEYS:
        name = TASK_INFO[task][cam_key]
        frame = frames.get(name)
        if frame is None:
            continue
        frame = cv2.resize(frame, (320, 240))
        images[name] = cv2.imencode(".png", frame)[-1].tobytes()
    return images


def rot_error_deg(q1, q2):
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)
    dot = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def split_pose(robot_type, arr):
    arr = np.asarray(arr, dtype=np.float64)
    if robot_type == "Franka":
        return {
            "pos": arr[:, 0:3],
            "quat": arr[:, 3:7],
            "grip": arr[:, 7],
        }
    if robot_type == "ARX5":
        return {
            "pos": arr[:, 0:3],
            "quat": Rotation.from_euler("xyz", arr[:, 3:6]).as_quat(),
            "grip": arr[:, 6],
        }
    if robot_type == "aloha":
        return {
            "pos": np.stack([arr[:, 0:3], arr[:, 8:11]], axis=1),
            "quat": np.stack([arr[:, 3:7], arr[:, 11:15]], axis=1),
            "grip": np.stack([arr[:, 7], arr[:, 15]], axis=1),
        }
    if robot_type == "UR5":
        return {"joints": arr[:, 0:6]}
    raise ValueError(robot_type)


def compare(robot_type, pred, actual):
    p = split_pose(robot_type, pred)
    a = split_pose(robot_type, actual)
    out = {}
    if robot_type == "UR5":
        diff = np.abs(p["joints"] - a["joints"])
        out["joint_mae_rad"] = float(diff.mean())
        out["joint_mae_rad_first"] = float(diff[0].mean())
        return out
    pos = np.linalg.norm(p["pos"] - a["pos"], axis=-1)
    rot = rot_error_deg(p["quat"], a["quat"])
    out["pos_mae_cm"] = float(pos.mean() * 100.0)
    out["pos_mae_cm_first"] = float(pos[0].mean() * 100.0)
    out["rot_mae_deg"] = float(rot.mean())
    out["rot_mae_deg_first"] = float(rot[0].mean())
    if "grip" in p:
        out["grip_mae"] = float(np.abs(p["grip"] - a["grip"]).mean())
        out["grip_mae_first"] = float(np.abs(p["grip"] - a["grip"])[0].mean())
    return out


def sample_indices(total, num_samples, horizon, stride):
    hi = total - horizon * stride - 1
    if hi <= 1:
        return [max(0, total // 2)]
    lo = 1
    return sorted({int(x) for x in np.linspace(lo, hi, num=num_samples)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/home/Spirit-v1.5")
    parser.add_argument("--backbone", default="/home/qwen3vl-4b-small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_samples", type=int, default=6)
    parser.add_argument("--tag", default=None, help="single-case robot tag (with --record_dir)")
    parser.add_argument("--task", default=None, help="single-case TASK_INFO key")
    parser.add_argument("--record_dir", default=None, help="single-case episode dir (with --tag/--task)")
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--frame_stride", type=int, default=2,
                        help="recorded frames per control step (30fps video vs 15Hz control)")
    parser.add_argument("--output", default="offline_action_eval.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"loading policy from {args.checkpoint} (backbone: {args.backbone}) ...")
    policy = SpiritVLAPolicy.from_pretrained(args.checkpoint, backbone=args.backbone)
    policy.eval()
    policy.to(device)

    img_dir = Path(tempfile.mkdtemp(prefix="spirit_offline_imgs_"))
    report = {"checkpoint": args.checkpoint, "num_samples": args.num_samples,
              "horizon": args.horizon, "cases": {}}

    for tag, rel_dir, task in (
        [(args.tag or "franka", args.record_dir, args.task or "move_objects_into_box")]
        if args.record_dir
        else CASES
    ):
        record_dir = os.path.normpath(os.path.join(HERE, rel_dir))
        robot = MockRCRobot.create_robot(tag, None, record_dir)
        info = TASK_INFO[task]
        action_type = info["action_type"]
        robot_type = info["robot_type"]
        total = robot.get_frame_number()
        stride = max(1, args.frame_stride)
        horizon = min(args.horizon, max(1, (total - 2) // stride))
        samples = sample_indices(total, args.num_samples, horizon, stride)
        print(f"\n=== {tag} / {task} frames={total} samples={samples} horizon={horizon} stride={stride} ===")

        states = []
        frames = {}
        for t in range(total):
            if t > 0:
                robot.fill()
            states.append(state_action(robot, action_type))
            if t in samples:
                frames[t] = api_images(robot, t, task)

        model_metrics = []
        base_metrics = []
        for t in samples:
            h = min(horizon, (total - t - 1) // stride)
            if h <= 0:
                continue
            item_tmp = {"images": frames[t], "action": states[t], "job_id": f"offline_{tag}_{t}"}
            batch = _prepare_batch(item_tmp, task, img_dir, t, None, None)
            with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
                actions = policy.select_action(batch)
            pred = _post_process_action(
                actions[0].float().cpu().numpy(),
                batch["observation.state.before_norm"].numpy(),
                robot_type,
                h,
                None,
                None,
            )
            actual = np.asarray([states[t + 1 + i * stride] for i in range(h)], dtype=np.float64)
            model_metrics.append(compare(robot_type, np.asarray(pred[:h]), actual))
            baseline = np.tile(np.asarray(states[t], dtype=np.float64), (h, 1))
            base_metrics.append(compare(robot_type, baseline, actual))

        def agg(rows, key):
            vals = [r[key] for r in rows if key in r]
            return float(np.mean(vals)) if vals else None

        keys = sorted({k for r in model_metrics for k in r})
        model_agg = {k: agg(model_metrics, k) for k in keys}
        base_agg = {k: agg(base_metrics, k) for k in keys}
        print("model   :", {k: round(v, 4) for k, v in model_agg.items() if v is not None})
        print("baseline:", {k: round(v, 4) for k, v in base_agg.items() if v is not None})
        report["cases"][tag] = {
            "task": task,
            "robot_type": robot_type,
            "action_type": action_type,
            "frames": total,
            "samples": samples,
            "horizon": horizon,
            "frame_stride": stride,
            "model": model_agg,
            "baseline_hold_pose": base_agg,
            "per_sample": model_metrics,
        }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\njson -> {args.output}")


if __name__ == "__main__":
    main()
