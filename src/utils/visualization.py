import matplotlib.pyplot as plt
import numpy as np
import logging

# Set up logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def plot_dann_losses(train_total, train_task, train_domain, val_losses, batch_indices=None, per_epoch=False, output_dir="results/"):
    """
    Plot DANN losses. Supports both batch-level and epoch-level plots.
    
    Args:
        train_total (list of float): Total training losses.
        train_task (list of float): Task-specific training losses.
        train_domain (list of float): Domain classification training losses.
        val_losses (list of float): Validation losses.
        batch_indices (list of int, optional): Batch indices to align with validation losses.
        per_epoch (bool): If True, plot per-epoch losses instead of batch-smoothed.
    """

    if per_epoch:
        # Plot per-epoch directly
        x_axis = list(range(1, len(train_total) + 1))
        title_suffix = " per Epoch"
        xlabel = "Epoch"
    else:
        # Smooth training losses (every 35 batches to align with val)
        def smooth_losses(losses, window=35):
            return [np.mean(losses[i:i+window]) for i in range(0, len(losses), window) if losses[i:i+window]]

        train_total = smooth_losses(train_total)
        train_task = smooth_losses(train_task)
        train_domain = smooth_losses(train_domain)
        val_losses = val_losses[:len(train_task)]  # trim for alignment
        x_axis = batch_indices[:len(train_task)] if batch_indices else list(range(len(train_task)))
        title_suffix = " (Smoothed by Batch)"
        xlabel = "Batch Number"

    # Ensure all lengths match
    min_len = min(len(train_total), len(train_task), len(train_domain), len(val_losses))
    train_total = train_total[:min_len]
    train_task = train_task[:min_len]
    train_domain = train_domain[:min_len]
    val_losses = val_losses[:min_len]
    x_axis = x_axis[:min_len]

    # === PLOT ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Task vs Val loss
    ax1.plot(x_axis, train_task, label="Train Task Loss", color='blue', marker='o', markersize=3)
    ax1.plot(x_axis, val_losses, label="Val Task Loss", color='orange', marker='s', markersize=3)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Task BCE Loss")
    ax1.set_title("Task Loss" + title_suffix)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: All loss components
    ax2.plot(x_axis, train_total, label="Total Loss", color='red', marker='o', markersize=3)
    ax2.plot(x_axis, train_task, label="Task Loss", color='blue', marker='s', markersize=3)
    ax2.plot(x_axis, train_domain, label="Domain Loss", color='green', marker='^', markersize=3)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss Components" + title_suffix)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/dann_losses_epoch" if per_epoch else "dann_losses_batch", dpi=300)
    plt.show()

    # === DIAGNOSTICS ===
    logger.info(f"\n📊 LOSS DIAGNOSTICS:")
    logger.info(f"Final Train Task Loss: {train_task[-1]:.4f}")
    logger.info(f"Final Val Task Loss: {val_losses[-1]:.4f}")
    logger.info(f"Final Domain Loss: {train_domain[-1]:.4f}")
    logger.info(f"Final Total Loss: {train_total[-1]:.4f}")
    logger.info(f"Task/Domain Ratio: {train_task[-1]/train_domain[-1]:.2f}")
    
    if train_task[-1] > 1.0:
        logger.info("⚠️  WARNING: Task loss > 1.0 suggests potential issues")
    if train_domain[-1] > 1.0:
        logger.info("⚠️  WARNING: Domain loss > 1.0 suggests potential issues")
    if abs(train_task[-1] - val_losses[-1]) > 0.2:
        logger.info("⚠️  WARNING: Large train/val gap suggests overfitting")

def plot_metrics(f1s, aucs, name="ALL", output_dir="results/"):
    """
    Enhanced F1 and AUROC plotting with class labels
    """
    CHEXPERT_CLASSES = [
        "Cardiomegaly",
        "Edema",
        "Consolidation",
        "Atelectasis",
        "Pleural Effusion"
    ]
    
    epochs = range(1, len(f1s) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # ---- F1 Plot ----
    plt.subplot(1, 2, 1)
    plt.plot(epochs, f1s, label=f"F1 Score ({name})", color='green', marker='o', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title(f"F1 Score Evolution ({name})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # ---- AUROC Plot ----
    plt.subplot(1, 2, 2)
    if isinstance(aucs[0], np.ndarray):
        mean_aucs = [np.nanmean(auc) for auc in aucs]
        plt.plot(epochs, mean_aucs, label=f"Mean AUROC ({name})", color='purple', marker='s', linewidth=2)
        
        for i, cls_name in enumerate(CHEXPERT_CLASSES):
            class_aucs = [auc[i] if not np.isnan(auc[i]) else 0 for auc in aucs]
            plt.plot(epochs, class_aucs, alpha=0.4, linewidth=1, label=f"{cls_name} ({name})")
    else:
        plt.plot(epochs, aucs, label=f"AUROC ({name})", color='purple', marker='s', linewidth=2)
    
    plt.xlabel("Epoch")
    plt.ylabel("AUROC")
    plt.title(f"AUROC Evolution ({name})")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc='lower right', ncol=2)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_evolution_{name}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"\n📈 METRICS SUMMARY ({name}):")
    logger.info(f"Best F1: {max(f1s):.4f} at epoch {np.argmax(f1s)+1}")
    if isinstance(aucs[0], np.ndarray):
        logger.info(f"Best Mean AUROC: {max(mean_aucs):.4f} at epoch {np.argmax(mean_aucs)+1}")
    else:
        logger.info(f"Best AUROC: {max(aucs):.4f} at epoch {np.argmax(aucs)+1}")

def plot_gradient_metrics(grad_conflicts, output_dir="results/"):
    """
    Create separate plots for each metric for detailed analysis
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    if not grad_conflicts:
        logger.info("[WARN] No gradient conflict data recorded.")
        return
    
    # Filter out None values
    valid_data = [(e, b, c, tm, dm) for e, b, c, tm, dm in grad_conflicts 
                  if c is not None and tm is not None and dm is not None]
    
    if not valid_data:
        logger.info("[WARN] No valid gradient conflict data found.")
        return
    
    epochs, batches, cosines, task_mags, domain_mags = zip(*valid_data)
    steps = [e * 10000 + b for e, b in zip(epochs, batches)]
    
    # Individual cosine similarity plot
    plt.figure(figsize=(12, 6))
    plt.plot(steps, cosines, marker="o", linestyle="-", alpha=0.7, linewidth=2)
    plt.axhline(0, color="red", linestyle="--", label="Conflict boundary")
    plt.axhline(-0.5, color="orange", linestyle=":", label="High conflict")
    plt.xlabel("Training Step")
    plt.ylabel("Cosine Similarity")
    plt.title("Task vs Domain Gradient Direction Conflict")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/cosine_similarity.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Individual magnitude plot
    plt.figure(figsize=(12, 6))
    plt.plot(steps, task_mags, marker="s", linestyle="-", alpha=0.7, label='Task Gradients', linewidth=2)
    plt.plot(steps, domain_mags, marker="^", linestyle="-", alpha=0.7, label='Domain Gradients', linewidth=2)
    plt.xlabel("Training Step")
    plt.ylabel("Gradient Magnitude (log scale)")
    plt.yscale('log')
    plt.title("Gradient Magnitudes Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/gradient_magnitudes.png", dpi=150, bbox_inches='tight')
    plt.show()