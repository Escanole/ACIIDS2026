import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
from torch.optim.lr_scheduler import LambdaLR

def save_checkpoint(state, dir_path, filename):
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, filename)
    torch.save(state, path)

class CheXpertTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Separate optimizers for DANN
        feature_params = list(model.module.features.parameters())
        task_params = list(model.module.task_classifier.parameters())
        domain_params = list(model.module.domain_conv.parameters()) + list(model.module.domain_classifier.parameters())
        
        # Ensure weight_decay is float
        lr = float(config['training']['learning_rate'])
        weight_decay = float(config['training']['weight_decay'])
        
        self.F_optimizer = optim.Adam(feature_params, lr=lr, weight_decay=weight_decay)
        self.C_optimizer = optim.Adam(task_params, lr=lr, weight_decay=weight_decay)
        self.D_optimizer = optim.Adam(domain_params, lr=lr, weight_decay=weight_decay)
        
        self.loss_task = nn.BCEWithLogitsLoss()
        self.loss_domain = nn.BCEWithLogitsLoss()

    def train(self, dataLoaderTrain, dataLoaderVal, trMaxEpoch, launchTimestamp, checkpoint=None):
        start_epoch = 0
        lossMIN = float('inf')
        aucMAX = float('-inf')
        
        if checkpoint is not None:
            print(f"[INFO] Resuming from checkpoint: {checkpoint}")
            modelCheckpoint = torch.load(checkpoint, weights_only=False)
            self.model.load_state_dict(modelCheckpoint['state_dict'])
            self.F_optimizer.load_state_dict(modelCheckpoint.get('F_optimizer', {}))
            self.C_optimizer.load_state_dict(modelCheckpoint.get('C_optimizer', {}))
            self.D_optimizer.load_state_dict(modelCheckpoint.get('D_optimizer', {}))
            start_epoch = modelCheckpoint.get('epoch', 0)
            lossMIN = modelCheckpoint.get('val_loss', float('inf'))
            aucMAX = modelCheckpoint.get('mean_auc', float('-inf'))
        
        total_epochs = start_epoch + trMaxEpoch
        
        # Learning rate schedulers
        lr_lambda = lambda epoch: 1 / (1 + 5 * (epoch / total_epochs))**0.5
        F_scheduler = LambdaLR(self.F_optimizer, lr_lambda)
        C_scheduler = LambdaLR(self.C_optimizer, lr_lambda)
        D_scheduler = LambdaLR(self.D_optimizer, lr_lambda)
        
        epoch_total_losses = []
        epoch_task_losses = []
        epoch_domain_losses = []
        val_losses = []
        f1s = []
        aucs = []
        
        for epochID in range(trMaxEpoch):
            global_epoch = start_epoch + epochID
            p = global_epoch / total_epochs
            lambda_ = 0.1 * (2. / (1 + np.exp(-10. * p)) - 1.)
            
            print(f"[INFO] Epoch {global_epoch+1}/{total_epochs} | λ: {lambda_:.3f} | LR: {self.F_optimizer.param_groups[0]['lr']:.6f}")
            
            # Unfreeze backbone at epoch 3
            if global_epoch == 3:
                print("[INFO] Unfreezing backbone (DenseNet121 features) with reduced LR")
                for param in self.model.module.features.parameters():
                    param.requires_grad = True
                for param_group in self.F_optimizer.param_groups:
                    param_group['lr'] = 1e-5
                    
                def freeze_bn(module):
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()
                        module.track_running_stats = False
                self.model.module.features.apply(freeze_bn)
            
            # Train epoch
            total_loss, task_loss, domain_loss = self._train_epoch(
                dataLoaderTrain, dataLoaderVal, lambda_, total_epochs, 
                self.config['model']['num_classes'], global_epoch
            )
            
            # Validation epoch
            val_loss, y_true, y_pred = self._validate_epoch(dataLoaderVal)
            auroc = self._compute_auroc(y_true, y_pred, self.config['model']['num_classes'])
            f1 = self._compute_f1(y_true, y_pred)
            mean_auc = np.nanmean(auroc)
            
            # Store metrics
            epoch_total_losses.append(total_loss)
            epoch_task_losses.append(task_loss)
            epoch_domain_losses.append(domain_loss)
            val_losses.append(val_loss)
            f1s.append(f1)
            aucs.append(auroc)
            
            # Save checkpoints
            checkpoint_state = {
                'epoch': global_epoch + 1,
                'state_dict': self.model.state_dict(),
                'F_optimizer': self.F_optimizer.state_dict(),
                'C_optimizer': self.C_optimizer.state_dict(),
                'D_optimizer': self.D_optimizer.state_dict(),
                'val_loss': val_loss,
                'mean_auc': mean_auc
            }
            
            if val_loss < lossMIN:
                lossMIN = val_loss
                save_checkpoint(checkpoint_state, "models/checkpoints/best_loss", 
                              f"best_loss_epoch{global_epoch}_{launchTimestamp}.pth.tar")
                print(f"[SAVE] Epoch {global_epoch+1} | 🟢 New best val loss: {val_loss:.4f}")
            
            if mean_auc > aucMAX:
                aucMAX = mean_auc
                save_checkpoint(checkpoint_state, "models/checkpoints/best_auc", 
                              f"best_auc_epoch{global_epoch}_{launchTimestamp}.pth.tar")
                print(f"[SAVE] Epoch {global_epoch+1} | 🔵 New best AUC: {mean_auc:.4f}")
            
            save_checkpoint(checkpoint_state, "models/checkpoints/last", 
                          f"last_epoch{global_epoch}_{launchTimestamp}.pth.tar")
            
            # Update schedulers
            F_scheduler.step()
            C_scheduler.step()
            D_scheduler.step()
        
        return epoch_total_losses, epoch_task_losses, epoch_domain_losses, val_losses, f1s, aucs

    def _train_epoch(self, dataLoader, dataLoaderVal, lambda_, trMaxEpoch, nnClassCount, epochID):
        self.model.train()

        batchs = []
        losstrain_total, losstrain_task, losstrain_domain = [], [], []
        losseval = []
        
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
            
            # Task loss (only on source domain - no support device)
            source_mask = (domain_labels.squeeze() == 0)
            if source_mask.sum() > 0:
                task_preds = self.model.module.get_task_predictions(features[source_mask])
                task_loss = self.loss_task(task_preds, task_labels[source_mask])
            else:
                task_loss = torch.tensor(0.0, requires_grad=True).cuda()
            
            # Domain loss for adversarial training
            domain_preds = self.model.module.get_domain_predictions(features)
            domain_loss_adv = self.loss_domain(domain_preds, domain_labels)
            
            # Combined loss
            total_loss = task_loss - lambda_ * domain_loss_adv
            
            self.F_optimizer.zero_grad()
            self.C_optimizer.zero_grad()
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
            else:
                print("[WARNING] NaN gradients detected, skipping step")

            # Track losses
            losstrain_total.append(total_loss.item())
            losstrain_task.append(task_loss.item())
            losstrain_domain.append(domain_loss.item())

            # Update progress
            if batchID % 35 == 0:
                batchs.append(batchID)
                val_loss, _, _ = self._validate_epoch(dataLoaderVal)
                losseval.append(val_loss)
                progress.set_postfix({
                    'Total Loss': f"{total_loss.item():.4f}",
                    'Task Loss': f"{task_loss.item():.4f}",
                    'Domain Loss': f"{domain_loss.item():.4f}",
                    'Domain Adv': f"{domain_loss_adv.item():.4f}",
                    'Val Loss': f"{val_loss:.4f}"
                })
        
        return (np.mean(losstrain_total), np.mean(losstrain_task), np.mean(losstrain_domain),)

    def _validate_epoch(self, dataLoader):
        self.model.eval()
        lossVal, lossValNorm = 0, 0
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for images, task_labels, _, _ in tqdm(dataLoader, desc="Validating", leave=False):
                images = images.cuda()
                task_labels = task_labels.cuda(non_blocking=True)
                
                features = self.model(images)
                task_output = self.model.module.get_task_predictions(features)

                losstensor = self.loss_task(task_output, task_labels)
                lossVal += losstensor.item()
                lossValNorm += 1
                
                y_true.append(task_labels.cpu())
                y_pred.append(torch.sigmoid(task_output).cpu())
        
        y_true = torch.cat(y_true, 0)
        y_pred = torch.cat(y_pred, 0)
        
        return lossVal / lossValNorm, y_true, y_pred

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

    def test(self, dataLoaderTest, nnClassCount, checkpoint, class_names):
        if checkpoint is not None:
            modelCheckpoint = torch.load(checkpoint, weights_only=False)
            self.model.load_state_dict(modelCheckpoint['state_dict'])
        
        self.model.eval()
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
        
        with torch.no_grad():
            for images, task_labels, _, paths in dataLoaderTest:
                task_labels = task_labels.cuda()
                outGT = torch.cat((outGT, task_labels), 0)
                
                images = images.cuda()
                features = self.model(images)
                task_out = self.model.module.get_task_predictions(features)
                task_out = torch.sigmoid(task_out)
                outPRED = torch.cat((outPRED, task_out), 0)
        
        aurocIndividual = self._compute_auroc(outGT, outPRED, nnClassCount)
        aurocMean = np.nanmean(aurocIndividual)
        
        print('\n📊 AUROC mean:', aurocMean)
        for i in range(len(aurocIndividual)):
            print(f'{class_names[i]} : {aurocIndividual[i]:.4f}')
        
        return outGT, outPRED
