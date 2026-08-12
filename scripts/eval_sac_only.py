"""Run SAC-only evaluation using the latest new checkpoint."""
import os
import sys
import json

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '.')

from scripts.evaluate import ModelEvaluator

MODEL_DIR = "output/models/20260808_153957"
EVAL_MAX_STEPS = int(os.environ.get("IBS_SAC_EVAL_MAX_STEPS", "2500"))
NUM_EPISODES = int(os.environ.get("IBS_SAC_EVAL_EPISODES", "10"))

print(f"Loading SAC model from: {MODEL_DIR}")
print(f"Eval max steps: {EVAL_MAX_STEPS}, episodes per surface: {NUM_EPISODES}")

ev = ModelEvaluator(models_dir=MODEL_DIR)

if 'sac' not in ev.models:
    print("[ERROR] SAC model not found in checkpoint directory!")
    sys.exit(1)

print(f"SAC model loaded successfully.")

results = ev.evaluate_sac(num_episodes=NUM_EPISODES)

print("\n\nFINAL SUMMARY (new reward function):")
for surface, m in results.items():
    print(f"  {surface:8s}: dist={m['stopping_distance']:.1f}m  time={m['stopping_time']:.1f}s  "
          f"success={m['stop_success_rate']*100:.0f}%  timeouts={m['timeout_count']}/{m['num_episodes']}")

print(f"\nMetrics saved to output/metrics/sac_metrics.json")
