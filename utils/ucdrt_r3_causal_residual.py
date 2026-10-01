import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalResidualEncoder(nn.Module):

    def __init__(self, feat_channels):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                feat_channels + 3,
                256,
                kernel_size=3,
                padding=1
            ),
            nn.GELU(),

            nn.Conv2d(
                256,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.GELU()
        )

        self.error_head = nn.Conv2d(128,1,1)

        self.type_head = nn.Conv2d(
            128,
            4,
            1
        )

        self.mag_head = nn.Conv2d(
            128,
            1,
            1
        )


    def forward(
        self,
        feat,
        prob,
        uncertainty,
        residual
    ):

        x=torch.cat(
            [
                feat,
                prob,
                uncertainty,
                residual
            ],
            dim=1
        )

        h=self.encoder(x)

        return {
            "error":
            torch.sigmoid(
                self.error_head(h)
            ),

            "type":
            self.type_head(h),

            "magnitude":
            torch.sigmoid(
                self.mag_head(h)
            )
        }



def uncertainty(prob):

    return prob*(1-prob)



def causal_residual_loss(
    pred,
    error_gt,
    type_gt,
    mag_gt
):

    loss1=F.binary_cross_entropy(
        pred["error"],
        error_gt
    )


    loss2=F.cross_entropy(
        pred["type"],
        type_gt.long()
    )


    loss3=F.smooth_l1_loss(
        pred["magnitude"],
        mag_gt
    )


    return (
        0.35*loss1+
        0.35*loss2+
        0.30*loss3
    )
