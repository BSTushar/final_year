import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import create_dataloader
from .decode import ctc_decode
from .model import create_model
from .utils import cer, indices_to_text, safe_mkdir, update_history, wer


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    idx2char: dict,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_chars = 0
    total_wer = 0.0
    num_batches = 0
    n_samples = 0

    # Create progress bar for training
    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs} [Train]",
        unit="batch",
        ncols=120,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    for batch in pbar:
        if batch is None:
            continue
        feats = batch["features"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        texts = batch["texts"]

        optimizer.zero_grad()
        log_probs = model(feats)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        if torch.isfinite(loss) and loss.item() > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_loss = float(loss.detach().cpu())
            total_loss += batch_loss
        else:
            batch_loss = 100.0
            total_loss += batch_loss

        decoded = ctc_decode(log_probs, beam_width=0)  # Use greedy for training (faster)
        idx = 0
        batch_correct = 0
        batch_chars = 0
        batch_wer = 0.0
        batch_samples = 0
        
        for i, L in enumerate(target_lengths):
            ref = targets[idx : idx + L]
            hyp = decoded[i]
            idx += L
            m = min(len(ref), len(hyp))
            if m > 0:
                ref_sub = ref[:m].cpu().tolist()
                hyp_sub = hyp[:m]
                batch_correct += sum(int(r == h) for r, h in zip(ref_sub, hyp_sub))
                batch_chars += m
            
            # Calculate WER for this sample
            ref_text = texts[i]
            hyp_text = indices_to_text(hyp, idx2char)
            batch_wer += wer(ref_text, hyp_text)
            batch_samples += 1
        
        total_correct += batch_correct
        total_chars += batch_chars
        total_wer += batch_wer
        n_samples += batch_samples
        num_batches += 1

        # Update progress bar with current metrics including WER
        current_loss = total_loss / num_batches
        current_acc = total_correct / max(1, total_chars)
        current_wer = total_wer / max(1, n_samples)
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.4f}',
            'WER': f'{current_wer:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

    avg_loss = total_loss / max(1, num_batches)
    acc = total_correct / max(1, total_chars)
    avg_wer = total_wer / max(1, n_samples)
    return avg_loss, acc, avg_wer


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    device: torch.device,
    idx2char: dict,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float, float, dict]:
    model.eval()
    total_loss = 0.0
    total_wer = 0.0
    total_cer = 0.0
    n = 0
    num_batches = 0
    total_blank_ratio = 0.0  # Track blank token predictions

    subs = ins = dels = 0

    # Create progress bar for validation
    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs} [Val]  ",
        unit="batch",
        ncols=120,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    with torch.no_grad():
        for batch in pbar:
            if batch is None:
                continue
            feats = batch["features"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            texts = batch["texts"]

            log_probs = model(feats)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            batch_loss = float(loss.detach().cpu())
            total_loss += batch_loss
            num_batches += 1

            # Calculate blank token ratio (monitoring for blank collapse issue)
            probs = torch.exp(log_probs)
            blank_probs = probs[:, :, 0]  # Blank token is index 0
            blank_ratio = (blank_probs > 0.5).float().mean().item()  # % of timesteps with blank > 0.5
            total_blank_ratio += blank_ratio

            decoded = ctc_decode(log_probs, beam_width=5)  # Use beam search for validation
            for i, hyp_idx in enumerate(decoded):
                ref = texts[i]
                hyp = indices_to_text(hyp_idx, idx2char)
                w = wer(ref, hyp)
                c = cer(ref, hyp)
                total_wer += w
                total_cer += c
                n += 1

            # Update progress bar with current metrics
            current_loss = total_loss / num_batches
            current_wer = total_wer / max(1, n)
            current_cer = total_cer / max(1, n)
            current_blank = total_blank_ratio / num_batches
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'WER': f'{current_wer:.4f}',
                'CER': f'{current_cer:.4f}',
                'blank%': f'{current_blank:.2%}'
            })

    avg_loss = total_loss / max(1, num_batches)
    avg_wer = total_wer / max(1, n)
    avg_cer = total_cer / max(1, n)
    avg_blank_ratio = total_blank_ratio / max(1, num_batches)
    error_stats = {
        "substitutions": subs, 
        "insertions": ins, 
        "deletions": dels,
        "blank_ratio": avg_blank_ratio
    }
    return avg_loss, avg_wer, avg_cer, error_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)  # Fixed: lowered from 1e-3
    parser.add_argument("--history_path", type=str, default="training_history.json")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (default: auto-detect last_epoch.pt)")
    args = parser.parse_args()

    device = torch.device("cpu")

    train_loader, char2idx, idx2char = create_dataloader(
        args.train_csv, batch_size=args.batch_size, shuffle=True, augment=True
    )
    val_loader, _, _ = create_dataloader(
        args.val_csv, batch_size=args.batch_size, shuffle=False, augment=False
    )

    vocab_size = len(char2idx)
    model = create_model(vocab_size=vocab_size).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')  # Fixed: True for stability
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
    last_lr = optimizer.param_groups[0]['lr']
    
    safe_mkdir("checkpoints")
    
    # Try to resume from checkpoint - prefer best model, fallback to last epoch
    start_epoch = 1
    best_wer = float("inf")
    no_improve_count = 0
    
    # Load best model weights (lowest WER), but use last epoch for epoch number
    best_checkpoint = os.path.join("checkpoints", "best_by_wer.pt")
    last_checkpoint = os.path.join("checkpoints", "last_epoch.pt")
    
    if args.resume:
        checkpoint_path = args.resume
        ckpt = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            print("[OK] Model weights loaded from specified checkpoint")
        
        # Load optimizer state if available
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            print("[OK] Optimizer state loaded")
        
        # Resume from next epoch
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
            print(f"[OK] Resuming from epoch {start_epoch}")
        
        # Load best WER if available
        if "best_wer" in ckpt:
            best_wer = ckpt["best_wer"]
            print(f"[OK] Best WER so far: {best_wer:.4f}")
        
        # Load no_improve_count if available
        if "no_improve_count" in ckpt:
            no_improve_count = ckpt["no_improve_count"]
    elif os.path.exists(best_checkpoint):
        # Load EVERYTHING from BEST checkpoint - start from epoch 50
        print(f"\n{'='*80}")
        print("Loading EVERYTHING from BEST checkpoint (best_by_wer.pt)")
        print("Resuming from epoch 50 with best model state")
        print(f"{'='*80}")
        
        best_ckpt = torch.load(best_checkpoint, map_location=device)
        
        # Load model state from BEST checkpoint
        if "model_state" in best_ckpt:
            model.load_state_dict(best_ckpt["model_state"])
            print("[OK] Best model weights loaded")
        
        # Load optimizer state from BEST checkpoint
        if "optimizer_state" in best_ckpt:
            optimizer.load_state_dict(best_ckpt["optimizer_state"])
            print("[OK] Optimizer state loaded (from best checkpoint)")
        
        # Start from epoch 50 (user requested)
        start_epoch = 50
        print(f"[OK] Starting from epoch {start_epoch} (as requested)")
        
        # Load best WER from best checkpoint
        if "best_wer" in best_ckpt:
            best_wer = best_ckpt["best_wer"]
            print(f"[OK] Best WER: {best_wer:.4f}")
        
        # Reset no_improve_count to 0 (fresh start from epoch 50)
        no_improve_count = 0
        print(f"[OK] No improve count reset to 0 (fresh start from epoch 50)")
        
        # Verify char2idx matches
        if "char2idx" in best_ckpt and best_ckpt["char2idx"] != char2idx:
            print("[WARNING] Vocabulary mismatch! Starting fresh.")
            start_epoch = 1
            best_wer = float("inf")
            no_improve_count = 0
            # Re-initialize if vocabulary mismatch
            for name, param in model.named_parameters():
                if param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param, gain=1.0)
                elif 'bias' in name:
                    torch.nn.init.constant_(param, 0.0)
    elif os.path.exists(last_checkpoint):
        # Fallback: use last epoch checkpoint
        print(f"\n{'='*80}")
        print("Loading checkpoint: last_epoch.pt")
        print(f"{'='*80}")
        ckpt = torch.load(last_checkpoint, map_location=device)
        
        # Load model state
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            print("[OK] Model weights loaded")
        
        # Load optimizer state if available
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            print("[OK] Optimizer state loaded")
        
        # Resume from next epoch
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
            print(f"[OK] Resuming from epoch {start_epoch}")
        
        # Load best WER if available
        if "best_wer" in ckpt:
            best_wer = ckpt["best_wer"]
            print(f"[OK] Best WER so far: {best_wer:.4f}")
        
        # Load no_improve_count if available
        if "no_improve_count" in ckpt:
            no_improve_count = ckpt["no_improve_count"]
        
        # Verify char2idx matches
        if "char2idx" in ckpt and ckpt["char2idx"] != char2idx:
            print("[WARNING] Vocabulary mismatch! Starting fresh.")
            start_epoch = 1
            best_wer = float("inf")
            no_improve_count = 0
            # Re-initialize if vocabulary mismatch
            for name, param in model.named_parameters():
                if param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param, gain=1.0)
                elif 'bias' in name:
                    torch.nn.init.constant_(param, 0.0)
        
    else:
        # Initialize model weights properly (fixed: was gain=0.1, too small)
        print(f"\n{'='*80}")
        print("No checkpoint found. Initializing new model.")
        print(f"{'='*80}")
        for name, param in model.named_parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param, gain=1.0)  # Fixed: was 0.1
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0.0)

    # Early stopping patience: 7 epochs (user requested)
    patience = 7

    print(f"\n{'='*80}")
    print(f"Training: Epochs {start_epoch} to {args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    print(f"Early stopping: Will stop if no WER improvement for {patience} consecutive epochs")
    print(f"{'='*80}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        # Learning rate warmup only for first 5 epochs of initial training
        if epoch <= 5 and start_epoch == 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * (epoch / 5.0)
        
        train_loss, train_acc, train_wer = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs, idx2char
        )
        val_loss, val_wer, val_cer, error_stats = validate(
            model, val_loader, criterion, device, idx2char, epoch, args.epochs
        )
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != last_lr:
            print(f"  ⚠ Learning rate reduced: {last_lr:.6f} → {current_lr:.6f}")
            last_lr = current_lr

        # Print epoch summary
        blank_ratio = error_stats.get("blank_ratio", 0.0)
        print(f"\n{'─'*80}")
        print(f"Epoch {epoch}/{args.epochs} Summary:")
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}, WER={train_wer:.4f}")
        print(f"  Val:   loss={val_loss:.4f}, WER={val_wer:.4f}, CER={val_cer:.4f}, Blank%={blank_ratio:.2%}")
        print(f"  LR:    {optimizer.param_groups[0]['lr']:.6f}")
        if blank_ratio > 0.9:
            print(f"  ⚠️  WARNING: High blank ratio ({blank_ratio:.2%}) - model may be collapsing to blanks!")
        if val_wer < best_wer:
            print(f"  [NEW BEST] WER: {val_wer:.4f} (previous: {best_wer:.4f})")
        print(f"{'─'*80}\n")

        last_ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "char2idx": char2idx,
            "idx2char": idx2char,
            "epoch": epoch,
            "best_wer": best_wer,
            "no_improve_count": no_improve_count,
        }
        torch.save(last_ckpt, os.path.join("checkpoints", "last_epoch.pt"))

        if val_wer < best_wer:
            best_wer = val_wer
            no_improve_count = 0
            torch.save(last_ckpt, os.path.join("checkpoints", "best_by_wer.pt"))
        else:
            no_improve_count += 1

        if no_improve_count >= patience and epoch >= start_epoch + 5:
            print(f"\n{'='*80}")
            print(f"Early stopping triggered: No WER improvement for {patience} consecutive epochs")
            print(f"Best WER achieved: {best_wer:.4f}")
            print(f"Stopped at epoch: {epoch}")
            print(f"{'='*80}")
            break

        update_history(
            args.history_path,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            train_wer=train_wer,
            val_wer=val_wer,
            val_cer=val_cer,
            error_stats=error_stats,
        )


if __name__ == "__main__":
    main()


