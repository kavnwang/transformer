import torch


def save_checkpoint(model, optimizer, scheduler, step: int, checkpoint_dir: str):
    checkpoint_path = f"{checkpoint_dir}/checkpoint_step_{step + 1}.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
        },
        checkpoint_path,
    )


def load_checkpoint(model, optimizer, scheduler, checkpoint_path: str, device=None) -> int:
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    step = checkpoint["step"]
    return step
