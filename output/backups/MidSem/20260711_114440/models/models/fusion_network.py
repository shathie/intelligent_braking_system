import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttentionFusion(nn.Module):
    """Fuse ViT features and temporal features using cross-modal attention"""
    def __init__(self, vit_dim=768, temporal_dim=256, fusion_dim=512, num_heads=8):
        super().__init__()
        self.vit_dim = vit_dim
        self.temporal_dim = temporal_dim
        self.fusion_dim = fusion_dim
        
        # Project temporal features to match ViT dimension
        self.temporal_proj = nn.Linear(temporal_dim, vit_dim)
        
        # Cross-modal attention: temporal features attend to ViT features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=vit_dim,
            num_heads=num_heads,
            dropout=0.1
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(vit_dim)
        self.norm2 = nn.LayerNorm(vit_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(vit_dim, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        # Final projection
        self.final_proj = nn.Linear(vit_dim + fusion_dim, fusion_dim)
    
    def forward(self, vit_features, temporal_features):
        # vit_features: (B, vit_dim)
        # temporal_features: (B, temporal_dim)
        
        batch_size = vit_features.size(0)
        
        # Add sequence dimension for attention
        vit_features = vit_features.unsqueeze(0)  # (1, B, vit_dim)
        temporal_features = temporal_features.unsqueeze(0)  # (1, B, temporal_dim)
        
        # Project temporal features to ViT dimension
        temporal_proj = self.temporal_proj(temporal_features)  # (1, B, vit_dim)
        
        # Cross-modal attention: temporal features as query, ViT as key/value
        attn_out, _ = self.cross_attn(
            query=temporal_proj,
            key=vit_features,
            value=vit_features
        )  # (1, B, vit_dim)
        
        # Residual connection and normalization
        temporal_proj = temporal_proj + attn_out
        temporal_proj = self.norm1(temporal_proj)
        
        # Feed-forward network
        ffn_out = self.ffn(temporal_proj)
        temporal_proj = temporal_proj + ffn_out
        temporal_proj = self.norm2(temporal_proj)
        
        # Remove sequence dimension
        temporal_proj = temporal_proj.squeeze(0)  # (B, vit_dim)
        
        # Concatenate and project to final fusion dimension
        combined = torch.cat([vit_features.squeeze(0), temporal_proj], dim=1)
        fusion_features = self.final_proj(combined)  # (B, fusion_dim)
        
        return fusion_features


class GatedFusion(nn.Module):
    """Alternative: Gated fusion network"""
    def __init__(self, vit_dim=768, temporal_dim=256, fusion_dim=512):
        super().__init__()
        self.vit_proj = nn.Linear(vit_dim, fusion_dim)
        self.temporal_proj = nn.Linear(temporal_dim, fusion_dim)
        
        # Gate mechanism
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid()
        )
        
        # Final projection
        self.final_proj = nn.Linear(fusion_dim, fusion_dim)
    
    def forward(self, vit_features, temporal_features):
        vit_proj = self.vit_proj(vit_features)
        temporal_proj = self.temporal_proj(temporal_features)
        
        # Concatenate for gate computation
        concat = torch.cat([vit_proj, temporal_proj], dim=1)
        gate = self.gate(concat)
        
        # Weighted sum
        fusion = gate * vit_proj + (1 - gate) * temporal_proj
        fusion = self.final_proj(fusion)
        
        return fusion