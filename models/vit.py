import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from timm.layers import trunc_normal_

class RoadSurfaceViT(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, **kwargs):
        super().__init__()
        self.vit = VisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=num_classes,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.1,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            **kwargs
        )
        
        # Additional layer for feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.LayerNorm(512)
        )
    
    def forward(self, x):
        # x: (B, 3, 224, 224)
        features = self.vit.forward_features(x)  # (B, 197, 768)
        cls_token = features[:, 0]  # (B, 768)
        
        # For classification
        logits = self.vit.head(cls_token)  # (B, num_classes)
        
        # For feature extraction
        extracted_features = self.feature_extractor(cls_token)  # (B, 512)
        
        return {
            "logits": logits,
            "features": extracted_features
        }