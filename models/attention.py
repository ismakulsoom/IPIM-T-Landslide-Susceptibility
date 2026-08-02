
import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):

    def __init__(self, dim=128, heads=8):
        super().__init__()

        self.attn_topo=nn.MultiheadAttention(
            dim,
            heads,
            batch_first=True
        )

        self.attn_env=nn.MultiheadAttention(
            dim,
            heads,
            batch_first=True
        )

        self.conv=nn.Conv2d(
            dim*3,
            dim,
            1
        )


    def forward(self,z_insar,z_topo,z_env):

        b,c,h,w=z_insar.shape

        q=z_insar.flatten(2).transpose(1,2)

        kt=z_topo.flatten(2).transpose(1,2)

        ke=z_env.flatten(2).transpose(1,2)


        at,_=self.attn_topo(
            q,kt,kt
        )

        ae,_=self.attn_env(
            q,ke,ke
        )


        out=torch.cat(
            [q,at,ae],
            dim=-1
        )


        out=out.transpose(1,2)

        out=out.reshape(
            b,
            c*3,
            h,
            w
        )


        return self.conv(out)
