"""
Run the full pipeline sequentially: analyze -> preprocess -> train -> evaluate -> simulate -> report
Logs output to output/run_all.log
"""
import subprocess
import datetime
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "output" / "run_all.log"
LOG.parent.mkdir(parents=True, exist_ok=True)

commands = [
    r".\.venv\Scripts\python.exe main.py --step analyze",
    r".\.venv\Scripts\python.exe scripts\preprocess_data.py",
    r".\.venv\Scripts\python.exe scripts\train.py",
    r".\.venv\Scripts\python.exe scripts\evaluate.py",
    r".\.venv\Scripts\python.exe scripts\simulate.py",
    r".\.venv\Scripts\python.exe scripts\report.py"
]

with open(LOG, 'a', encoding='utf-8') as f:
    f.write(f"Run started: {datetime.datetime.now().isoformat()}\n")
    run_env = os.environ.copy()
    run_env.setdefault("IBS_FORCE_CPU", "1")
    run_env.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    for cmd in commands:
        f.write(f"\n>>> Running: {cmd}\n")
        f.flush()
        try:
            proc = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True, env=run_env)
            f.write(proc.stdout or '')
            if proc.stderr:
                f.write('\n[stderr]\n')
                f.write(proc.stderr)
            f.write(f"\nExit code: {proc.returncode}\n")
        except Exception as e:
            f.write(f"[ERROR] Exception when running {cmd}: {e}\n")
            break
    f.write(f"Run finished: {datetime.datetime.now().isoformat()}\n")

print(f"Started full-run logger. See {LOG} for progress.")
