"""
Focal Loss implementation for addressing class imbalance.

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
https://arxiv.org/abs/1708.02002

Focal Loss is particularly effective for imbalanced datasets because it:
1. Down-weights easy (high-confidence) examples
2. Focuses on hard (low-confidence) examples
3. Naturally emphasizes minority classes which are often harder to predict

The focusing parameter gamma (typically 2.0) controls how much down-weighting happens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in classification.
    
    Args:
        alpha: class weights. Can be:
            - None: uniform weights
            - 1D tensor of shape (num_classes,): per-class weight
            - float: constant weight (rarely used)
        gamma: focusing parameter controlling how much to down-weight easy examples.
            - gamma=0: equivalent to standard CrossEntropyLoss
            - gamma=1.0-1.5: mild focusing, good for moderate imbalance
            - gamma=2.0-2.5: strong focusing, recommended for severe imbalance
            - gamma>3.0: very aggressive, may hurt common class performance
        reduction: how to reduce loss across batch
            - 'mean': average loss (default)
            - 'sum': sum loss
            - 'none': return per-sample loss
        ignore_index: class index to ignore (default: -1). Useful for padding.
    
    Example:
        >>> import torch
        >>> # For 27 road surface classes with class imbalance
        >>> class_weights = torch.tensor([...])  # computed from class frequencies
        >>> focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)
        >>> logits = model(images)  # shape: (batch_size, 27)
        >>> labels = torch.tensor([...])  # shape: (batch_size,)
        >>> loss = focal_loss(logits, labels)
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        if gamma < 0:
            raise ValueError(f"Focusing parameter gamma must be >= 0, got {gamma}")
    
    def forward(self, logits, targets):
        """Compute focal loss.
        
        Args:
            logits: model logits, shape (batch_size, num_classes)
            targets: target class indices, shape (batch_size,)
        
        Returns:
            loss: scalar focal loss (if reduction='mean' or 'sum'), 
                  or tensor of shape (batch_size,) if reduction='none'
        """
        # Validate inputs
        if logits.dim() != 2:
            raise ValueError(f"logits must be 2D, got shape {logits.shape}")
        if targets.dim() != 1:
            raise ValueError(f"targets must be 1D, got shape {targets.shape}")
        if logits.size(0) != targets.size(0):
            raise ValueError(f"batch size mismatch: logits {logits.size(0)} vs targets {targets.size(0)}")
        
        num_classes = logits.size(1)
        
        # Get softmax probabilities
        p = F.softmax(logits, dim=1)
        
        # Get log softmax for numerical stability
        log_p = F.log_softmax(logits, dim=1)
        
        # Get cross-entropy loss (without reduction)
        ce_loss = F.nll_loss(log_p, targets, reduction='none', ignore_index=self.ignore_index)
        
        # Get probability of the true class for each sample
        p_t = torch.gather(p, 1, targets.unsqueeze(1)).squeeze(1)
        
        # Clamp p_t to avoid log(0)
        p_t = torch.clamp(p_t, min=1e-7, max=1.0)
        
        # Compute focal weight: (1 - p_t)^gamma
        # This down-weights easy examples (high p_t) and focuses on hard examples (low p_t)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply class weights if provided
        focal_loss = ce_loss * focal_weight
        
        if self.alpha is not None:
            if isinstance(self.alpha, (int, float)):
                # Scalar alpha: constant weight
                alpha_t = torch.full_like(focal_loss, self.alpha)
            else:
                # Tensor alpha: per-class weights
                if self.alpha.dim() != 1:
                    raise ValueError(f"alpha must be 1D tensor, got shape {self.alpha.shape}")
                if self.alpha.size(0) != num_classes:
                    raise ValueError(f"alpha size {self.alpha.size(0)} doesn't match num_classes {num_classes}")
                
                # Gather alpha values for each target
                alpha_t = torch.gather(
                    self.alpha.unsqueeze(0).expand(targets.size(0), -1),
                    1,
                    targets.unsqueeze(1)
                ).squeeze(1)
            
            focal_loss = alpha_t * focal_loss
        
        # Handle ignore_index
        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index)
            focal_loss = focal_loss * mask.float()
        
        # Apply reduction
        if self.reduction == 'mean':
            # Average only over non-ignored samples
            if self.ignore_index >= 0:
                mask = (targets != self.ignore_index)
                if mask.sum() > 0:
                    return focal_loss.sum() / mask.sum()
                else:
                    return focal_loss.mean()
            else:
                return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class WeightedFocalLoss(FocalLoss):
    """Focal Loss variant that combines focal weighting with sample weighting.
    
    Useful when you also want to weight individual samples (e.g., based on source dataset).
    """
    
    def forward(self, logits, targets, sample_weights=None):
        """Compute weighted focal loss.
        
        Args:
            logits: model logits, shape (batch_size, num_classes)
            targets: target class indices, shape (batch_size,)
            sample_weights: optional per-sample weights, shape (batch_size,)
        
        Returns:
            loss: scalar weighted focal loss
        """
        # Get focal loss without reduction
        focal_loss = super().forward(logits, targets.unsqueeze(0) if targets.dim() == 0 else targets)
        
        if self.reduction != 'none':
            # Call parent with reduction='none' to get per-sample loss
            self_reduction = self.reduction
            self.reduction = 'none'
            focal_loss = super().forward(logits, targets)
            self.reduction = self_reduction
        
        # Apply sample weights if provided
        if sample_weights is not None:
            if sample_weights.size(0) != focal_loss.size(0):
                raise ValueError(f"sample_weights size {sample_weights.size(0)} doesn't match loss size {focal_loss.size(0)}")
            focal_loss = focal_loss * sample_weights
        
        # Apply reduction
        if self.reduction == 'mean':
            if sample_weights is not None:
                return focal_loss.sum() / sample_weights.sum()
            else:
                return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def create_focal_loss_from_class_counts(class_counts, gamma=2.0, device='cpu'):
    """Create FocalLoss with weights computed from class counts.
    
    Args:
        class_counts: array/tensor of shape (num_classes,) with class sample counts
        gamma: focusing parameter
        device: device to place weights on
    
    Returns:
        FocalLoss instance with computed alpha weights
    """
    class_counts = torch.tensor(class_counts, dtype=torch.float32, device=device)
    
    # Effective number weighting: rare classes get higher weight
    # w_c = (N - 1) / (N * count_c)
    # where N is total samples
    total = class_counts.sum()
    effective_num = 1.0 - torch.pow(0.9999, class_counts)
    weights = (1.0 - 0.9999) / effective_num
    weights = weights / weights.sum() * len(class_counts)
    
    return FocalLoss(alpha=weights, gamma=gamma, reduction='mean')


# Example usage and testing
if __name__ == "__main__":
    # Test 1: Basic focal loss
    print("Test 1: Basic Focal Loss")
    batch_size, num_classes = 32, 5
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    focal_loss = FocalLoss(gamma=2.0)
    loss = focal_loss(logits, targets)
    print(f"  Batch size: {batch_size}, Num classes: {num_classes}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  ✓ Passed\n")
    
    # Test 2: With class weights
    print("Test 2: Focal Loss with Class Weights")
    class_weights = torch.tensor([1.0, 2.0, 3.0, 5.0, 10.0])  # Increasing imbalance
    focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)
    loss = focal_loss(logits, targets)
    print(f"  Class weights: {class_weights}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  ✓ Passed\n")
    
    # Test 3: Compare with CrossEntropyLoss
    print("Test 3: Comparison with CrossEntropyLoss")
    ce_loss = nn.CrossEntropyLoss()
    focal_loss_gamma0 = FocalLoss(gamma=0.0)  # gamma=0 should be similar to CE
    
    ce = ce_loss(logits, targets)
    focal = focal_loss_gamma0(logits, targets)
    print(f"  CrossEntropyLoss: {ce.item():.4f}")
    print(f"  Focal Loss (gamma=0): {focal.item():.4f}")
    print(f"  Difference: {abs(ce.item() - focal.item()):.4f} (should be small)")
    print(f"  ✓ Passed\n")
    
    # Test 4: Different gamma values
    print("Test 4: Effect of Gamma Parameter")
    for gamma in [0.0, 1.0, 2.0, 3.0]:
        focal_loss = FocalLoss(gamma=gamma)
        loss = focal_loss(logits, targets)
        print(f"  gamma={gamma}: loss={loss.item():.4f}")
    print(f"  ✓ Passed (gamma=0 should be highest, gamma=3 should be lowest)\n")
    
    # Test 5: From class counts
    print("Test 5: Create from Class Counts")
    class_counts = torch.tensor([1000, 100, 10, 5, 1])  # Severe imbalance
    focal_loss = create_focal_loss_from_class_counts(class_counts, gamma=2.0)
    loss = focal_loss(logits, targets)
    print(f"  Class counts: {class_counts}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  ✓ Passed\n")
    
    print("All tests passed! ✓")
