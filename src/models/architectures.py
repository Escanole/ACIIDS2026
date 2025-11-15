import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.init as init

class SharedAttentionAdapter(nn.Module):
    """Shared attention mechanism that processes features once for both task and domain branches"""
    def __init__(self, in_channels, embed_dim=256, num_heads=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)  # Reduce dim: 1024 → 256
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.apply(self._init_weights)
        
    def forward(self, x):  # x: [B, C=1024, H, W]
        x = self.proj(x)          # [B, 256, H, W]
        B, C, H, W = x.shape
        x = x.view(B, C, H*W).permute(0, 2, 1)  # [B, HW, C]
        attn_out, _ = self.attn(x, x, x)       # Self-attention over spatial locations
        attn_out = attn_out.permute(0, 2, 1)   # [B, C, HW]
        pooled = self.pool(attn_out)           # [B, C, 1]
        return pooled.squeeze(-1)              # [B, C=256]

    def _init_weights(self, m): 
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d): 
            init.kaiming_uniform_(m.weight, nonlinearity='relu') 
            if m.bias is not None: 
                init.zeros_(m.bias)

class DANN_DenseNet121(nn.Module):
    def __init__(self, task_out_size, domain_out_size=1, freeze_backbone=False):
        super(DANN_DenseNet121, self).__init__()
        self.features = models.densenet121(pretrained=True).features
        
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
       
        # Shared attention mechanism - processes features once for both branches
        self.shared_attention = SharedAttentionAdapter(in_channels=1024, embed_dim=256, num_heads=4)
        
        # Task classifier branch
        self.task_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, task_out_size)
        )
        
        # Domain classifier branch  
        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, domain_out_size)
        )
    
    def forward(self, x):
        """Extract features from backbone"""
        features = self.features(x)  # [B, 1024, H, W]
        return features
    
    def get_shared_representation(self, features):
        """Get shared attention-processed representation"""
        return self.shared_attention(features)  # [B, 256]
    
    def get_task_predictions(self, features):
        """Get task predictions using shared attention"""
        shared_repr = self.get_shared_representation(features)  # [B, 256]
        return self.task_classifier(shared_repr)                # [B, task_out]
    
    def get_domain_predictions(self, features):
        """Get domain predictions using shared attention"""
        shared_repr = self.get_shared_representation(features)  # [B, 256]
        return self.domain_classifier(shared_repr)              # [B, domain_out_size]