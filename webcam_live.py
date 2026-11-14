"""
Project 72: Interactive Live Video Stream Yoga Pose Correction CLI
Processes camera frames, classifies yoga asanas in real time, evaluates joint angles
against ideal alignment geometries, and outputs natural language correction cues.
"""

import os
import sys
import time
import numpy as np
import torch

from main import (
    YogaNet,
    YOGA_POSES,
    NUM_CLASSES,
    DEVICE,
    synthesize_yoga_landmarks,
    normalize_yoga_landmarks,
    evaluate_alignment_and_feedback
)

def run_simulated_stream_cli(model):
    print("\n" + "=" * 78)
    print("  SIMULATED REAL-TIME VIDEO STREAM YOGA POSE CORRECTION")
    print("=" * 78)
    print("  Tracking 33-point body skeleton, joint angle alignment, and cues...")
    print("  Press Ctrl+C to stop session.\n")

    stream_cycle = [
        ("Warrior_II",   0.80, "Warrior II (Front knee too high at 110°)"),
        ("Warrior_II",   0.00, "Warrior II (Perfect 90° Knee Alignment)"),
        ("Tree_Pose",    0.00, "Tree Pose (Balanced on Right Leg)"),
        ("Downward_Dog", 0.00, "Downward-Facing Dog (Inverted V Shape)"),
        ("Goddess_Pose", 0.65, "Goddess Pose (Shallow Plié Squat)"),
        ("Goddess_Pose", 0.00, "Goddess Pose (Deep 90° Plié Form)"),
        ("Cobra_Pose",   0.00, "Cobra Pose (Chest Lifted with Spinal Arch)")
    ]

    try:
        frame_idx = 0
        while True:
            target_pose, dev, desc = stream_cycle[frame_idx % len(stream_cycle)]
            pts = synthesize_yoga_landmarks(target_pose, apply_noise=True, form_deviation=dev)

            # Neural Classification
            feat = normalize_yoga_landmarks(pts)
            feat_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            t0 = time.perf_counter()
            with torch.no_grad():
                logits = model(feat_t)
                probs = torch.softmax(logits, dim=1)[0]
                conf, pred_idx = torch.max(probs, 0)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            pred_pose = YOGA_POSES[pred_idx.item()]

            # Alignment Evaluation
            eval_res = evaluate_alignment_and_feedback(pred_pose, pts)
            score = eval_res["alignment_score"]
            status = "OPTIMAL" if eval_res["is_optimal"] else "ADJUST"
            cue = eval_res["feedback_cues"][0]

            print(f"[Frame {frame_idx:03d} | {latency_ms:4.2f}ms] "
                  f"Pose: {pred_pose:<12} ({conf.item()*100:5.1f}%) | "
                  f"Score: {score:5.1f}% [{status:<7}] | Cue: {cue}")

            frame_idx += 1
            time.sleep(0.35)

    except KeyboardInterrupt:
        print("\nLive yoga alignment session terminated by user.")

def main():
    weights_path = os.path.join(os.path.dirname(__file__), "outputs", "yogaclassifier_best.pth")
    model = YogaNet(in_features=99, num_classes=NUM_CLASSES).to(DEVICE)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()

    run_simulated_stream_cli(model)

if __name__ == "__main__":
    main()

