import matplotlib.pyplot as plt
import numpy as np

def plot_dann_losses(train_total, train_task, train_domain, val_losses, per_epoch=True):
    """Plot DANN training losses"""
    
    if per_epoch:
        x_axis = list(range(1, len(train_total) + 1))
        title_suffix = " per Epoch"
        xlabel = "Epoch"
    else:
        # Batch-level smoothing
        def smooth_losses(losses, window=35):
            return [np.mean(losses[i:i+window]) for i in range(0, len(losses), window) if losses[i:i+window]]
        
        train_total = smooth_losses(train_total)
        train_task = smooth_losses(train_task)
        train_domain = smooth_losses(train_domain)
        val_losses = val_losses[:len(train_task)]
        x_axis = list(range(len(train_task)))
        title_suffix = " (Smoothed)"
        xlabel = "Batch Interval"
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Task vs Val loss
    ax1.plot(x_axis, train_task, label="Train Task Loss", color='blue', marker='o', markersize=3)
    ax1.plot(x_axis, val_losses, label="Val Task Loss", color='orange', marker='s', markersize=3)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Task BCE Loss")
    ax1.set_title("Task Loss" + title_suffix)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # All loss components
    ax2.plot(x_axis, train_total, label="Total Loss", color='red', marker='o', markersize=3)
    ax2.plot(x_axis, train_task, label="Task Loss", color='blue', marker='s', markersize=3)
    ax2.plot(x_axis, train_domain, label="Domain Loss", color='green', marker='^', markersize=3)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss Components" + title_suffix)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/dann_losses.png", dpi=300)
    plt.show()

def plot_metrics(f1s, aucs):
    """Plot F1 and AUROC metrics"""
    epochs = range(1, len(f1s) + 1)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, f1s, label="F1 Score", color='green', marker='o', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1 Score Evolution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    if isinstance(aucs[0], np.ndarray):
        mean_aucs = [np.nanmean(auc) for auc in aucs]
        plt.plot(epochs, mean_aucs, label="Mean AUROC", color='purple', marker='s', linewidth=2)
    else:
        plt.plot(epochs, aucs, label="AUROC", color='purple', marker='s', linewidth=2)
    
    plt.xlabel("Epoch")
    plt.ylabel("AUROC")
    plt.title("AUROC Evolution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("results/metrics_evolution.png", dpi=300)
    plt.show()
