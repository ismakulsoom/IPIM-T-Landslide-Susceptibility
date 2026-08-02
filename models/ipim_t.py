import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CrossAttentionFusion


class CNNEncoder(nn.Module):
    def __init__(self, in_channels, channels=(32, 64, 128), dropout=0.25):
        super().__init__()
        layers = []
        c = in_channels
        for i, o in enumerate(channels):
            layers += [
                nn.Conv2d(c, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ]
            if i < len(channels) - 1:
                layers.append(nn.Dropout2d(dropout))
            c = o
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class InSARTransformerEncoder(nn.Module):
    def __init__(self, embed_dim=128, heads=8, layers=4):
        super().__init__()
        self.proj = nn.Conv2d(1, embed_dim, 1)

        block = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=512,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            block,
            num_layers=layers
        )

    def forward(self, x):
        """
        Input: x of shape [batch, 1, H, W]
        Output: [batch, 128, 4, 4] (fixed spatial size for compatibility)
        """
        b, c, h, w = x.shape
        
        # Project to transformer dimension: [batch, 128, H, W]
        x = self.proj(x)
        
        # Flatten spatial dimensions for transformer: [batch, 128, H*W] -> [batch, H*W, 128]
        x = x.flatten(2).transpose(1, 2)
        
        # Apply transformer: [batch, H*W, 128]
        x = self.transformer(x)
        
        # Reshape back to spatial: [batch, 128, H, W]
        x = x.transpose(1, 2)
        
        # Use adaptive pooling to ensure consistent output size [batch, 128, 4, 4]
        # This handles any input spatial dimension
        x = x.reshape(b, 128, h, w)
        x = F.adaptive_avg_pool2d(x, (4, 4))
        
        return x


class IPIMT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.topo_encoder = CNNEncoder(
            len(config.TOPO_INDICES)
        )

        self.env_encoder = CNNEncoder(
            len(config.ENV_INDICES)
        )

        self.insar_encoder = InSARTransformerEncoder(
            config.TRANSFORMER_DIM,
            config.TRANSFORMER_HEADS,
            config.TRANSFORMER_LAYERS
        )

        self.fusion = CrossAttentionFusion(
            config.TRANSFORMER_DIM,
            config.TRANSFORMER_HEADS
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        if config.USE_PHYSICS:
            self.cohesion = nn.Parameter(
                torch.tensor(config.COHESION_INIT, dtype=torch.float32)
            )

            self.friction = nn.Parameter(
                torch.tensor(config.FRICTION_INIT, dtype=torch.float32)
            )

    def forward(self, env, insar):
        # Split env into topo and env_static
        topo = env[:, 0:8]           # [batch, 8, H, W]
        env_static = env[:, 8:14]    # [batch, 6, H, W]
        
        # Pad env_static from 6 to 15 channels to match env_encoder's Conv2d(15, ...)
        # The env_encoder expects 15 channels
        if env_static.shape[1] == 6:
            pad_channels = 15 - env_static.shape[1]  # 9 channels
            env_static = F.pad(
                env_static,
                (0, 0, 0, 0, 0, pad_channels)  # Pad channels at the end
            )  # [batch, 15, H, W]
        
        # Pass through encoders
        zt = self.topo_encoder(topo)      # [batch, 128, H/8, W/8] after pooling
        zl = self.env_encoder(env_static)  # [batch, 128, H/8, W/8] after pooling
        zi = self.insar_encoder(insar)     # [batch, 128, 4, 4] (fixed)
        
        # Pool zt and zl to match zi's spatial size (4x4)
        # This ensures all features have the same spatial dimensions before fusion
        target_size = zi.shape[2]  # Should be 4
        if zt.shape[2] != target_size or zt.shape[3] != target_size:
            zt = F.adaptive_avg_pool2d(zt, (target_size, target_size))
            zl = F.adaptive_avg_pool2d(zl, (target_size, target_size))
        
        # Fusion
        fused = self.fusion(zi, zt, zl)
        
        # Classifier
        return self.classifier(fused).squeeze(1)