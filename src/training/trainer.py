import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import logging
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import torch.backends.cudnn as cudnn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

def save_checkpoint(state, dir_path, filename):
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, filename)
    torch.save(state, path)

def cosine_similarity(grads_a, grads_b): 
    """
    Calculate cosine similarity AND gradient magnitudes in one function
    
    Returns:
        tuple: (cosine_similarity, magnitude_a, magnitude_b)
    """
    dot = sum([(ga*gb).sum() for ga, gb in zip(grads_a, grads_b)]) 
    norm_a = torch.sqrt(sum([(ga**2).sum() for ga in grads_a])) 
    norm_b = torch.sqrt(sum([(gb**2).sum() for gb in grads_b])) 
    
    cosine_sim = (dot / (norm_a*norm_b + 1e-8)).item()
    magnitude_a = norm_a.item()
    magnitude_b = norm_b.item()
    
    return cosine_sim, magnitude_a, magnitude_b

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        """
        Focal Loss implementation for binary classification
        
        Args:
            alpha (float): Weighting factor for positive class vs negative class
                         - Common values: 0.25 (more weight to negatives) or 1.0 (balanced)
                         - Can also be a tensor for per-class alpha values
            gamma (float): Focusing parameter to down-weight easy examples (default: 2)
            reduction (str): Specifies the reduction to apply to the output
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, inputs, targets):
        # Calculate BCE loss without pos_weight (focal loss handles imbalance)
        bce_loss = self.bce(inputs, targets)
        
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(inputs)
        
        # Calculate p_t: probability of the true class
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Calculate focal weight: (1-p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting for class balance
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def lamdba_computing(model, dataLoaderVal, nnClassCount, factor, strategy, epoch, max_epoch):
    def compute_lambda_entropy(model, dataLoaderVal, nnClassCount, factor):
        model.eval()
        eps = 1e-8
        all_entropy = []
    
        with torch.no_grad():
            for i, (images, _, _, _) in enumerate(dataLoaderVal):
                if i >= 5:  # average over first 5 batches
                    break
                images = images.cuda(non_blocking=True)
                features = model(images)
                task_preds = model.module.get_task_predictions(features)
                probs = torch.sigmoid(task_preds)
    
                # Compute per-class binary entropy
                entropy = - (probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps))
                # Mean over classes for each sample, then mean over batch
                entropy_batch_mean = entropy.mean(dim=1).mean().item()
                all_entropy.append(entropy_batch_mean)
    
        entropy_mean = np.mean(all_entropy)
        max_entropy = np.log(2)  # mean per class max entropy
        lambda_dyn = (entropy_mean / (max_entropy + eps)) * factor
    
        model.train()
        return lambda_dyn
    
    def compute_lambda_schedule(epoch, max_epoch, factor):
        p = epoch / max_epoch
        return factor*(2. / (1+np.exp(-10.*p)) - 1.)

    if strategy == 'task_entropy':
        lambda_dyn = compute_lambda_entropy(model, dataLoaderVal, nnClassCount, factor)
    elif strategy == 'schedule':
        lambda_dyn = compute_lambda_schedule(epoch, max_epoch, factor)
    else:
        print("Strategy is invalid!")

    return lambda_dyn

class CheXpertTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

        # Set up logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Separate optimizers for DANN
        feature_params = list(model.module.features.parameters())
        task_params = list(model.module.task_classifier.parameters())
        domain_params = list(model.module.domain_classifier.parameters())  # No more domain_conv
        shared_attention_params = list(model.module.shared_attention.parameters())  # New shared attention
        
        # Ensure weight_decay is float
        att_lr = float(config['training']['attention_learning_rate'])
        lr = float(config['training']['learning_rate'])
        weight_decay = float(config['training']['weight_decay'])
        
        self.SA_optimizer = optim.Adam(shared_attention_params, lr=att_lr, weight_decay=weight_decay)
        self.F_optimizer = optim.Adam(feature_params, lr=lr, weight_decay=weight_decay)
        self.C_optimizer = optim.Adam(task_params, lr=lr, weight_decay=weight_decay)
        self.D_optimizer = optim.Adam(domain_params, lr=lr, weight_decay=weight_decay)
        
        train_csv = config['paths']['train_csv']
        df = pd.read_csv(train_csv)

        label_cols = config['class_names'][:-1]  # exclude Support Devices
        df[label_cols] = df[label_cols].replace(-1, 1).fillna(0)

        labels_np = df[label_cols].values.astype(np.float32)
        binary_labels_np = (labels_np >= 0.5).astype(np.float32)
        task_labels_np = binary_labels_np[:, :config['model']['num_classes']]

        pos_counts = task_labels_np.sum(axis=0)
        N = task_labels_np.shape[0]

        # Original focal loss approach — inverse frequency, unscaled
        alpha_values = N / pos_counts
        min_a, max_a = alpha_values.min(), alpha_values.max()
        scaled_alpha = 0.25 + 0.5 * (alpha_values - min_a) / (max_a - min_a)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alpha_tensor = torch.tensor(scaled_alpha, dtype=torch.float32, device=device)  

        self.loss_task = FocalLoss(alpha=alpha_tensor, gamma=2, reduction='mean')
        self.loss_domain = nn.BCEWithLogitsLoss()

    def train(self, dataLoaderTrain, dataLoaderVal, trMaxEpoch, launchTimestamp, checkpoint=None):
        start_epoch = 0
        lossMIN = float('inf')
        aucMAX = float('-inf')

        epoch_total_losses = []
        epoch_task_losses = []
        epoch_domain_losses = []
        val_losses = []
        f1s = []
        aucs = []
        aucs_dev = []
        aucs_nodev = []
        all_grad_conflicts = []

        # === Epoch control logic ===
        total_epochs = trMaxEpoch  # used for correct lambda schedule
        run_epochs = trMaxEpoch  # default to full training

        if checkpoint is not None:
            self.logger.info(f"Resuming from checkpoint: {checkpoint}")
            modelCheckpoint = torch.load(checkpoint, weights_only=False)
            self.model.load_state_dict(modelCheckpoint['state_dict'])
            self.F_optimizer.load_state_dict(modelCheckpoint.get('F_optimizer', {}))
            self.C_optimizer.load_state_dict(modelCheckpoint.get('C_optimizer', {}))
            self.D_optimizer.load_state_dict(modelCheckpoint.get('D_optimizer', {}))
            self.SA_optimizer.load_state_dict(modelCheckpoint.get('SA_optimizer', {}))
            start_epoch = modelCheckpoint.get('epoch', 0)
            lossMIN = modelCheckpoint.get('val_loss', float('inf'))
            aucMAX = modelCheckpoint.get('mean_auc', float('-inf'))

            # Calculate remaining epochs and limit to 15 per session
            remaining_epochs = max(0, trMaxEpoch - start_epoch)
            run_epochs = min(15, remaining_epochs)

            if remaining_epochs <= 0:
                self.logger.info(f"Training already completed! Current epoch {start_epoch} >= target {trMaxEpoch}")
                return epoch_total_losses, epoch_task_losses, epoch_domain_losses, val_losses, f1s, aucs
            elif remaining_epochs <= 15:
                self.logger.info(f"Continuing training for {remaining_epochs} final epochs (to reach {trMaxEpoch})")
            else:
                self.logger.info(
                    f"Continuing training for 15 more epochs ({remaining_epochs} remaining to reach {trMaxEpoch})")

        else:
            # Fresh training - limit to 15 epochs per session
            run_epochs = min(15, trMaxEpoch)
            if trMaxEpoch > 15:
                self.logger.info(f"No checkpoint. Requested {trMaxEpoch} epochs, running first 15 epochs.")
            else:
                self.logger.info(f"No checkpoint. Running {trMaxEpoch} epochs.")

        # Learning rate schedulers
        T_max = run_epochs  # number of epochs for one cosine cycle
        eta_min = 1e-6      # minimum learning rate at the end of annealing
        
        F_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.F_optimizer, T_max=T_max, eta_min=eta_min)
        C_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.C_optimizer, T_max=T_max, eta_min=eta_min)
        D_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.D_optimizer, T_max=T_max, eta_min=eta_min)
        SA_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.SA_optimizer, T_max=T_max, eta_min=eta_min)

        for epochID in range(run_epochs):
            current_epoch = start_epoch + epochID
            p = current_epoch / total_epochs
            
            lambda_ = lamdba_computing(self.model, dataLoaderVal, self.config['model']['num_classes'], self.config['training']['factor'], self.config['training']['strategy'], current_epoch, total_epochs)

            self.logger.info(
                f"Epoch {current_epoch + 1}/{total_epochs} | λ: {lambda_:.3f} | LR: {self.F_optimizer.param_groups[0]['lr']:.6f}")

            # Unfreeze backbone at epoch 3
            if current_epoch == 3:
                self.logger.info("Unfreezing backbone (DenseNet121 features) with reduced LR")
                for param in self.model.module.features.parameters():
                    param.requires_grad = True

                def freeze_bn(module):
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()
                        module.track_running_stats = False

                self.model.module.features.apply(freeze_bn)

            # Train epoch
            total_loss, task_loss, domain_loss, grad_conflicts_epoch = self._train_epoch(
                dataLoaderTrain, dataLoaderVal, lambda_, total_epochs,
                self.config['model']['num_classes'], current_epoch
            )

            # Validation epoch
            val_loss, (y_true, y_pred), (y_true_dev, y_pred_dev), (y_true_nodev, y_pred_nodev) = self._validate_epoch(dataLoaderVal)
            competition_indices = [2, 5, 6, 8, 10]
            auroc = self._compute_auroc(y_true[:, competition_indices],
                                        y_pred[:, competition_indices],
                                        len(competition_indices))
            
            auroc_dev = self._compute_auroc(
                                        y_true_dev[:, competition_indices],
                                        y_pred_dev[:, competition_indices],
                                        len(competition_indices)
                                    ) if y_true_dev is not None else None
            
            auroc_nodev = self._compute_auroc(
                                        y_true_nodev[:, competition_indices],
                                        y_pred_nodev[:, competition_indices],
                                        len(competition_indices)
                                    ) if y_true_nodev is not None else None
            f1 = self._compute_f1(y_true, y_pred)
            mean_auc = np.nanmean(auroc)

            # Store metrics
            epoch_total_losses.append(total_loss)
            epoch_task_losses.append(task_loss)
            epoch_domain_losses.append(domain_loss)
            all_grad_conflicts.extend(grad_conflicts_epoch)
            aucs.append(auroc)
            aucs_dev.append(auroc_dev)
            aucs_nodev.append(auroc_nodev)
            f1s.append(f1)
            val_losses.append(val_loss)

            # Save checkpoints
            checkpoint_state = {
                'epoch': current_epoch + 1,
                'state_dict': self.model.state_dict(),
                'F_optimizer': self.F_optimizer.state_dict(),
                'C_optimizer': self.C_optimizer.state_dict(),
                'D_optimizer': self.D_optimizer.state_dict(),
                'SA_optimizer': self.SA_optimizer.state_dict(),
                'val_loss': val_loss,
                'mean_auc': mean_auc
            }

            if val_loss < lossMIN:
                lossMIN = val_loss
                save_checkpoint(checkpoint_state, "models/checkpoints/best_loss",
                                f"best_loss_epoch{current_epoch}_{launchTimestamp}.pth.tar")
                self.logger.info(f"SAVE - Epoch {current_epoch + 1} | 🟢 New best val loss: {val_loss:.4f}")

            if mean_auc > aucMAX:
                aucMAX = mean_auc
                save_checkpoint(checkpoint_state, "models/checkpoints/best_auc",
                                f"best_auc_epoch{current_epoch}_{launchTimestamp}.pth.tar")
                self.logger.info(f"SAVE - Epoch {current_epoch + 1} | 🔵 New best AUC: {mean_auc:.4f}")

            save_checkpoint(checkpoint_state, "models/checkpoints/last",
                            f"last_epoch{current_epoch}_{launchTimestamp}.pth.tar")
            self.logger.info(
                f"SAVE - Epoch {current_epoch + 1} | 💾 Saved last checkpoint | Val Loss = {val_loss:.4f} | AUC Score = {mean_auc:.4f} | F1 Score = {f1:.4f}")

            if epochID == 0:
                best_auc = mean_auc
                epochs_no_improve = 0
            else:
                if mean_auc > best_auc:
                    best_auc = mean_auc
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
            
            if epochs_no_improve >= 5:
                self.logger.info(f"[EARLY STOP] No improvement in mean AUC for 5 consecutive epochs. Stopping at epoch {current_epoch+1}.")
                break

            # Update schedulers
            F_scheduler.step()
            C_scheduler.step()
            D_scheduler.step()
            SA_scheduler.step()

        return epoch_total_losses, epoch_task_losses, epoch_domain_losses, val_losses, f1s, aucs, aucs_dev, aucs_nodev, all_grad_conflicts

    def _train_epoch(self, dataLoader, dataLoaderVal, lambda_, trMaxEpoch, nnClassCount, epochID):
        self.model.train()

        batchs = []
        losstrain_total, losstrain_task, losstrain_domain = [], [], []
        losseval = []
        grad_conflicts = []

        progress = tqdm(enumerate(dataLoader), total=len(dataLoader), desc="Training")
        for batchID, (images, task_labels, domain_labels, _) in progress:
            images = images.cuda()
            task_labels = task_labels.float().view(-1, nnClassCount).cuda(non_blocking=True)
            domain_labels = domain_labels.float().view(-1, 1).cuda(non_blocking=True)

            # Step 1: Train Domain Discriminator
            features = self.model(images).detach()
            domain_preds = self.model.module.get_domain_predictions(features)
            domain_loss = self.loss_domain(domain_preds, domain_labels)

            self.D_optimizer.zero_grad()
            domain_loss.backward()
            self.D_optimizer.step()

            # Step 2: Train Feature Extractor + Task Classifier
            features = self.model(images)

            # Domain loss for adversarial training
            domain_preds = self.model.module.get_domain_predictions(features)
            domain_loss_adv = self.loss_domain(domain_preds, domain_labels)

            task_preds = self.model.module.get_task_predictions(features)
            task_loss = self.loss_task(task_preds, task_labels)

            # Combined loss
            total_loss = task_loss - lambda_ * domain_loss_adv

            # === Every 35 batches: compute gradient conflict (safely, before .backward()) ===
            grad_cos = None
            if batchID % 35 == 0:
                try:
                    # select only feature parameters that require grad
                    features_params = [p for p in self.model.module.features.parameters() if p.requires_grad]
                    if len(features_params) > 0:
                        # compute grads via autograd.grad (does not write into .grad buffers)
                        task_grads = torch.autograd.grad(
                            task_loss,
                            features_params,
                            retain_graph=True,
                            allow_unused=True
                        )
                        domain_grads = torch.autograd.grad(
                            domain_loss_adv,
                            features_params,
                            retain_graph=True,
                            allow_unused=True
                        )
    
                        # filter out None entries
                        task_grads = [g.detach() for g in task_grads if g is not None]
                        domain_grads = [g.detach() for g in domain_grads if g is not None]
                        domain_grads_adv = [ -lambda_ * g for g in domain_grads ]
    
                        if len(task_grads) > 0 and len(domain_grads) > 0:
                            # compute cosine similarity safely
                            grad_cos, task_magnitude, domain_magnitude = cosine_similarity(task_grads, domain_grads_adv)
                        else:
                            grad_cos, task_magnitude, domain_magnitude = None, None, None
                    else:
                        grad_cos, task_magnitude, domain_magnitude = None, None, None
                except Exception as e:
                    # don't crash training for analysis errors
                    self.logger.info(f"[GradConflict] skipped at Epoch {epochID} Batch {batchID}: {e}")
                    grad_cos, task_magnitude, domain_magnitude = None, None, None

            self.F_optimizer.zero_grad()
            self.C_optimizer.zero_grad()
            self.SA_optimizer.zero_grad()
            total_loss.backward()

            # Check for NaN gradients
            has_nan = False
            for param in self.model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan = True
                    break

            if not has_nan:
                self.F_optimizer.step()
                self.C_optimizer.step()
                self.SA_optimizer.step()
            else:
                self.logger.warning("NaN gradients detected, skipping step")

            # Track losses
            losstrain_total.append(total_loss.item())
            losstrain_task.append(task_loss.item())
            losstrain_domain.append(domain_loss.item())

            # Update progress
            if batchID % 35 == 0:
                batchs.append(batchID)
                val_loss, _, _, _ = self._validate_epoch(dataLoaderVal)
                losseval.append(val_loss)
                postfix = {
                    'Total Loss': f"{total_loss.item():.4f}",
                    'Task Loss': f"{task_loss.item():.4f}",
                    'Domain Loss': f"{domain_loss.item():.4f}",
                    'Domain Adv': f"{domain_loss_adv.item():.4f}",
                    'Val Loss': f"{val_loss:.4f}"
                }
                if grad_cos is not None and task_magnitude is not None and domain_magnitude is not None:
                    grad_conflicts.append((epochID, batchID, grad_cos, task_magnitude, domain_magnitude))
    
                progress.set_postfix(postfix)
    

        return (np.mean(losstrain_total), np.mean(losstrain_task), np.mean(losstrain_domain), grad_conflicts)

    def _validate_epoch(self, dataLoader):
        self.model.eval()
        lossVal, lossValNorm = 0, 0
        y_true, y_pred = [], []
        y_true_devices, y_pred_devices = [], []
        y_true_no_devices, y_pred_no_devices = [], []

        with torch.no_grad():
            for varInput, task_target, domain_labels, _ in tqdm(dataLoader, desc="Validating", leave=False):
                varInput = varInput.cuda()
                varTarget = task_target.cuda(non_blocking=True)
                domain_labels = domain_labels.cuda(non_blocking=True)
    
                # Only use task classifier for validation
                features = self.model(varInput)
                varTaskOutput = self.model.module.get_task_predictions(features)
    
                losstensor = self.loss_task(varTaskOutput, varTarget)
                lossVal += losstensor.item()
                lossValNorm += 1
    
                y_true.append(varTarget.cpu())
                y_pred.append(torch.sigmoid(varTaskOutput).cpu())  # Apply sigmoid for BCEWithLogitsLoss

                mask_devices = (domain_labels.squeeze() == 1)
                mask_no_devices = (domain_labels.squeeze() == 0)
                
                if mask_devices.any():
                    y_true_devices.append(varTarget[mask_devices].cpu())
                    y_pred_devices.append(torch.sigmoid(varTaskOutput[mask_devices]).cpu())
                
                if mask_no_devices.any():
                    y_true_no_devices.append(varTarget[mask_no_devices].cpu())
                    y_pred_no_devices.append(torch.sigmoid(varTaskOutput[mask_no_devices]).cpu())
    

        def finalize_metrics(y_true_list, y_pred_list):
            if len(y_true_list) == 0:
                return None, None
            y_true = torch.cat(y_true_list, 0)
            y_pred = torch.cat(y_pred_list, 0)
            return y_true, y_pred
        
        y_true, y_pred = finalize_metrics(y_true, y_pred)
        y_true_devices, y_pred_devices = finalize_metrics(y_true_devices, y_pred_devices)
        y_true_no_devices, y_pred_no_devices = finalize_metrics(y_true_no_devices, y_pred_no_devices)

        return (lossVal / lossValNorm, 
                (y_true, y_pred),
                (y_true_devices, y_pred_devices),
                (y_true_no_devices, y_pred_no_devices))
    
    def test(self, dataLoaderTest, nnClassCount, checkpoint, class_names, use_gpu=True):
        cudnn.benchmark = True

        # === Load model checkpoint ===
        if checkpoint is not None:
            ckpt = torch.load(checkpoint, weights_only=False)
            self.model.load_state_dict(ckpt['state_dict'])

        if use_gpu:
            self.model = self.model.cuda()
            outGT = torch.FloatTensor().cuda()
            outPRED = torch.FloatTensor().cuda()
            outDomainGT = torch.LongTensor().cuda()
            outDomainPred = torch.LongTensor().cuda()
        else:
            self.model = self.model.cpu()
            outGT = torch.FloatTensor()
            outPRED = torch.FloatTensor()
            outDomainGT = torch.LongTensor()
            outDomainPred = torch.LongTensor()

        self.model.eval()

        # === Forward pass ===
        with torch.no_grad():
            for input, target, domain_labels, _ in dataLoaderTest:
                target = target.cuda() if use_gpu else target
                domain_labels = domain_labels.cuda() if use_gpu else domain_labels

                outGT = torch.cat((outGT, target), 0)
                outDomainGT = torch.cat((outDomainGT, domain_labels), 0)

                varInput = input.cuda() if use_gpu else input
                features = self.model(varInput)

                # Task head
                task_out = self.model.module.get_task_predictions(features)
                task_out = torch.sigmoid(task_out)
                outPRED = torch.cat((outPRED, task_out), 0)

                # Domain head
                domain_out = self.model.module.get_domain_predictions(features)
                domain_pred = (torch.sigmoid(domain_out) > 0.5).long().view(-1)
                outDomainPred = torch.cat((outDomainPred, domain_pred), 0)

        # === AUROC metrics ===
        aurocIndividual = self._compute_auroc(outGT, outPRED, nnClassCount)
        aurocMean = np.nanmean(aurocIndividual)

        domain_acc = accuracy_score(outDomainGT.cpu().numpy(), outDomainPred.cpu().numpy())

        self.logger.info(f"\nAUROC mean: {aurocMean:.4f}")
        self.logger.info(f"Domain head accuracy (device presence): {domain_acc:.4f}")

        for i in range(len(aurocIndividual)):
            self.logger.info(f"{class_names[i]} : {aurocIndividual[i]:.4f}")

        # === CheXpert 5-class analysis ===
        chexpert5 = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
        label_to_index = {name: idx for idx, name in enumerate(class_names)}
        indices5 = [label_to_index[lbl] for lbl in chexpert5]

        (best_f1, best_thresh,
        best_prec, best_rec,
        best_f1_pc, best_prec_pc, best_rec_pc) = self._find_best_f1(outGT, outPRED, focus_indices=indices5)

        self.logger.info(f"\nBest threshold F1 (5-class): {best_f1:.4f} at threshold={best_thresh:.3f}")
        for i, idx in enumerate(indices5):
            self.logger.info(f"{class_names[idx]}: F1={best_f1_pc[i]:.4f}, "
                            f"Precision={best_prec_pc[i]:.4f}, Recall={best_rec_pc[i]:.4f}")

        chexpert5_aucs = [aurocIndividual[i] for i in indices5]
        mean5 = np.nanmean(chexpert5_aucs)

        self.logger.info("\nCheXpert 5-class AUROC:")
        for i in indices5:
            self.logger.info(f"{class_names[i]}: {aurocIndividual[i]:.4f}")
        self.logger.info(f"Mean AUROC (5-class): {mean5:.4f}")

        return outGT, outPRED, outDomainGT, outDomainPred, aurocIndividual
    
    def _find_best_f1(self, dataGT, dataPRED, num_thresholds=1000, focus_indices=None):
        thresholds = np.linspace(0, 1, num_thresholds)
        best_f1 = -1
        best_thresh = None
        best_precision = None
        best_recall = None
        best_f1_per_class = None
        best_precision_per_class = None
        best_recall_per_class = None
    
        y_true = dataGT.cpu().numpy()
        y_scores = dataPRED.cpu().numpy()
    
        # If focus_indices provided, filter GT and predictions
        if focus_indices is not None:
            y_true = y_true[:, focus_indices]
            y_scores = y_scores[:, focus_indices]
    
        for t in thresholds:
            preds = (y_scores > t).astype(int)
            f1 = f1_score(y_true, preds, average='macro', zero_division=0)
    
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
                # Macro
                best_precision = precision_score(y_true, preds, average='macro', zero_division=0)
                best_recall = recall_score(y_true, preds, average='macro', zero_division=0)
                # Per-class
                best_f1_per_class = f1_score(y_true, preds, average=None, zero_division=0)
                best_precision_per_class = precision_score(y_true, preds, average=None, zero_division=0)
                best_recall_per_class = recall_score(y_true, preds, average=None, zero_division=0)
    
        return (best_f1, best_thresh,
                best_precision, best_recall,
                best_f1_per_class, best_precision_per_class, best_recall_per_class)

    def _compute_auroc(self, dataGT, dataPRED, classCount):
        scores = []
        npGT = dataGT.cpu().numpy()
        npPRED = dataPRED.cpu().numpy()

        for i in range(classCount):
            try:
                scores.append(roc_auc_score(npGT[:, i], npPRED[:, i]))
            except:
                scores.append(float('nan'))

        return np.array(scores)

    def _compute_f1(self, dataGT, dataPRED, threshold=0.5):
        preds = (dataPRED > threshold).float()
        return f1_score(dataGT.numpy(), preds.numpy(), average='macro')

    def setup_file_logging(self, log_file_path):
        """Add file handler to logger for saving logs to file"""
        file_handler = logging.FileHandler(log_file_path)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.info(f"File logging enabled: {log_file_path}")