import torch
import torch.nn as nn
import torchvision.models as models

class DomainAttentionAdapter(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):  # x: [B, C=1024, H, W]
        x = self.proj(x)          # [B, 256, H, W]
        B, C, H, W = x.shape
        x = x.view(B, C, H*W).permute(0, 2, 1)  # [B, HW, C]
        attn_out, _ = self.attn(x, x, x)       # Self-attention over spatial locations
        attn_out = attn_out.permute(0, 2, 1)   # [B, C, HW]
        pooled = self.pool(attn_out)           # [B, C, 1]
        return pooled.squeeze(-1)              # [B, C]

class TaskAttentionAdapter(nn.Module):
    def __init__(self, in_channels, embed_dim=256, num_heads=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):  # x: [B, 1024, H, W]
        x = self.proj(x)                # [B, 256, H, W]
        B, C, H, W = x.shape
        x = x.view(B, C, H*W).permute(0, 2, 1)  # [B, HW, C]
        attn_out, _ = self.attn(x, x, x)
        attn_out = attn_out.permute(0, 2, 1)   # [B, C, HW]
        pooled = self.pool(attn_out)           # [B, C, 1]
        return pooled.squeeze(-1)              # [B, C]

class DANN_DenseNet121(nn.Module):
    def __init__(self, task_out_size, domain_out_size=1, freeze_backbone=False):
        super(DANN_DenseNet121, self).__init__()
        self.features = models.densenet121(pretrained=True).features

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        # Task classifier with attention
        self.task_attention = TaskAttentionAdapter(in_channels=1024, embed_dim=256, num_heads=4)
        self.task_classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, task_out_size)
        )

        # Domain classifier with attention
        self.domain_conv = DomainAttentionAdapter(in_channels=1024, embed_dim=256, num_heads=4)
        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, domain_out_size)
        )

    def forward(self, x):
        features = self.features(x)  # [B, 1024, H, W]
        return features

    def get_task_predictions(self, features):
        attn = self.task_attention(features)      # [B, 256]
        return self.task_classifier(attn)         # [B, task_out]
    
    def get_domain_predictions(self, features):
        conv = self.domain_conv(features)         # [B, 256]
        return self.domain_classifier(conv)       # [B, domain_out_size]

# Keep legacy models for compatibility
class DenseNet121(nn.Module):
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, out_size):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.resnet18(x)
        return x
