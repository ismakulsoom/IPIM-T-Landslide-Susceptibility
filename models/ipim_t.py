
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CrossAttentionFusion


class CNNEncoder(nn.Module):
    def __init__(self, in_channels, channels=(32,64,128), dropout=0.25):
        super().__init__()
        layers=[]
        c=in_channels
        for i,o in enumerate(channels):
            layers += [
                nn.Conv2d(c,o,3,padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ]
            if i < len(channels)-1:
                layers.append(nn.Dropout2d(dropout))
            c=o
        self.encoder=nn.Sequential(*layers)

    def forward(self,x):
        return self.encoder(x)


class InSARTransformerEncoder(nn.Module):
    def __init__(self, embed_dim=128, heads=8, layers=4):
        super().__init__()
        self.proj=nn.Conv2d(1,embed_dim,1)

        block=nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=512,
            batch_first=True,
            activation="gelu"
        )

        self.transformer=nn.TransformerEncoder(
            block,
            num_layers=layers
        )

    def forward(self,x):
        b=x.size(0)

        x=self.proj(x)

        x=x.flatten(2).transpose(1,2)

        x=self.transformer(x)

        x=x.transpose(1,2)

        x=x.reshape(
            b,
            128,
            32,
            32
        )

        return F.adaptive_avg_pool2d(
            x,(4,4)
        )


class IPIMT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.topo_encoder=CNNEncoder(
            len(config.TOPO_INDICES)
        )

        self.env_encoder=CNNEncoder(
            len(config.ENV_INDICES)
        )

        self.insar_encoder=InSARTransformerEncoder(
            config.TRANSFORMER_DIM,
            config.TRANSFORMER_HEADS,
            config.TRANSFORMER_LAYERS
        )


        self.fusion=CrossAttentionFusion(
            config.TRANSFORMER_DIM,
            config.TRANSFORMER_HEADS
        )


        self.classifier=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,1),
            nn.Sigmoid()
        )


        if config.USE_PHYSICS:
            self.cohesion=nn.Parameter(
                torch.tensor(config.COHESION_INIT)
            )

            self.friction=nn.Parameter(
                torch.tensor(config.FRICTION_INIT)
            )


    def forward(self,env,insar):

        topo=env[:,0:8]
        env_static=env[:,8:14]


        zt=self.topo_encoder(topo)

        zl=self.env_encoder(env_static)

        zi=self.insar_encoder(insar)


        fused=self.fusion(
            zi,
            zt,
            zl
        )

        return self.classifier(fused).squeeze(1)
