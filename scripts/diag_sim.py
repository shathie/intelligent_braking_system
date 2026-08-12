"""Quick diagnostic: test simulator physics with different braking forces."""
import os, sys, numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '.')
from scripts.simulate import VehicleSimulator

print("=== FULL BRAKING (force=1.0) on dry (mu=0.8) ===")
sim = VehicleSimulator()
sim.set_surface('dry', 0.8)
obs = sim.step(0)
print(f"Initial v_x: {obs['v_x']:.2f} m/s, a_x: {obs['a_x']:.4f}, dt={sim.dt}")

for i in range(350):
    obs = sim.step(1.0)
    if i % 30 == 0 or obs['v_x'] < 1.0:
        print(f"  step {i+1:4d}: v_x={obs['v_x']:.3f}  a_x={obs['a_x']:.4f}")
    if obs['v_x'] < 0.1:
        print(f"  => Vehicle stopped at step {i+1}!")
        break
else:
    print(f"  => NOT stopped after 350 steps. Final v_x={obs['v_x']:.3f}")

print("\n=== WEAK BRAKING (force=0.03) on dry (mu=0.8) ===")
sim2 = VehicleSimulator()
sim2.set_surface('dry', 0.8)
obs2 = sim2.step(0)
for i in range(50):
    obs2 = sim2.step(0.03)
    if i % 5 == 0:
        print(f"  step {i+1:4d}: v_x={obs2['v_x']:.3f}  a_x={obs2['a_x']:.4f}")
    if obs2['v_x'] < 0.1:
        print(f"  => Stopped at step {i+1}!")
        break
