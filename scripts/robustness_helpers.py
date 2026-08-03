import os
import torch

class EarlyStopping:
    """Early stopping to prevent overfitting when validation loss stops improving."""
    def __init__(self, patience=5, min_delta=1e-4, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric):
        score = -val_metric if self.mode == "min" else val_metric
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class CheckpointSaver:
    """Manages checkpoint saving, preserving only the best model weights on disk."""
    def __init__(self, checkpoint_dir, file_name="best_model.pth", mode="min"):
        self.checkpoint_dir = checkpoint_dir
        self.file_name = file_name
        self.mode = mode
        self.best_score = float('inf') if mode == "min" else float('-inf')
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, model, optimizer, epoch, score):
        is_best = (score < self.best_score) if self.mode == "min" else (score > self.best_score)
        
        if is_best:
            self.best_score = score
            save_path = os.path.join(self.checkpoint_dir, self.file_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'score': score
            }, save_path)
            print(f"[*] New best validation score: {score:.4f}. Saved model weights to {save_path}")
            return True
        return False

def configure_robust_optimizer_and_scheduler(model, lr=1e-4, weight_decay=1e-4):
    """Configures AdamW with weight decay and ReduceLROnPlateau scheduler."""
    # AdamW incorporates weight decay directly into parameter updates for better L2 regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Decays learning rate when a metric has stopped improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=3, 
        factor=0.1, 
        verbose=True
    )
    
    return optimizer, scheduler
