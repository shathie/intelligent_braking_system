import torch
import torch.nn as nn

class PhysicsInformedNetwork(nn.Module):
    """Physics-Informed Neural Network for friction estimation"""
    def __init__(self, input_dim=512, hidden_dims=[256, 128, 64], output_dim=1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.Tanh())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # μ ∈ [0, 1]
        
        self.mlp = nn.Sequential(*layers)
        
        # Vehicle parameters (can be learned or fixed)
        self.m = nn.Parameter(torch.tensor(1500.0))  # Vehicle mass [kg]
        self.I_w = nn.Parameter(torch.tensor(1.5))   # Wheel inertia [kg·m²]
        self.R_w = nn.Parameter(torch.tensor(0.3))   # Wheel radius [m]
        self.C = nn.Parameter(torch.tensor(10.0))   # Tire stiffness coefficient
    
    def forward(self, x):
        # x: (B, input_dim) - fused features
        mu = self.mlp(x)  # (B, 1)
        return mu
    
    def physics_loss(self, x, vehicle_state):
        """
        Compute physics-informed loss terms.
        vehicle_state: dict containing:
            - a_x: longitudinal acceleration (B,)
            - omega: wheel speeds (B, 4)
            - v_x: vehicle speed (B,)
            - F_brake: braking force (B,)
            - F_z: normal force (B,)
        """
        batch_size = x.size(0)
        mu = torch.clamp(self.forward(x), min=1e-3, max=1.0)  # (B, 1)
        
        # Unpack vehicle state
        a_x = vehicle_state["a_x"].reshape(-1)  # (B,)
        omega = vehicle_state["omega"].reshape(batch_size, 4)  # (B, 4)
        v_x = vehicle_state["v_x"].reshape(-1)  # (B,)
        F_brake = vehicle_state["F_brake"].reshape(-1)  # (B,)
        F_z = vehicle_state["F_z"]
        if F_z.dim() == 1:
            F_z = F_z.unsqueeze(1).repeat(1, 4)
        else:
            F_z = F_z.reshape(batch_size, 4)
        
        # Compute slip ratio for each wheel.
        s = torch.zeros(batch_size, 4, device=x.device)
        for i in range(4):
            s[:, i] = (self.R_w * omega[:, i] - v_x) / torch.clamp(
                torch.maximum(self.R_w * omega[:, i], v_x),
                min=1e-6
            )
        
        # If CAN brake signal is normalized [0, 1], map it to a rough physical range [0, 12000]N.
        if torch.mean(torch.abs(F_brake)) < 50:
            F_brake = F_brake * 12000.0

        # Tire force (Brush Model) with explicit broadcasting.
        mu_w = mu.repeat(1, 4)  # (B, 4)
        F_x = mu_w * F_z * (1 - torch.exp(-self.C * s.abs() / mu_w))
        F_x_total = F_x.sum(dim=1)  # (B,)
        
        # Longitudinal dynamics: m * a_x = F_brake - F_x_total - F_rolling - F_aero
        # Simplified: m * a_x ≈ F_brake - F_x_total (ignoring rolling/aero for simplicity)
        dynamics_residual = self.m * a_x - (F_brake - F_x_total)
        
        # Wheel dynamics: I_w * omega_dot = T_brake - F_x * R_w
        # We don't have omega_dot, so we use finite difference approximation
        # This would require sequential data, so we skip for now
        
        # Normalize residual to keep physics term numerically stable.
        residual_norm = dynamics_residual / torch.clamp(torch.abs(self.m) * 9.81, min=1.0)

        # Physics loss: robust MSE of normalized residual.
        physics_loss = torch.mean(residual_norm ** 2)
        physics_loss = torch.nan_to_num(physics_loss, nan=1e3, posinf=1e3, neginf=1e3)
        
        return physics_loss