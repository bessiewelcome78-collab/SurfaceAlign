import torch
import torch.nn.functional as F



def gaussian_target(
        offset,
        radius=8,
        sigma=1.5
):

    actions = torch.arange(
        -radius,
        radius+1,
        device=offset.device
    )

    target = torch.exp(
        -(actions-offset.unsqueeze(-1))**2
        /(2*sigma*sigma)
    )

    target = target / (
        target.sum(
            dim=-1,
            keepdim=True
        ) + 1e-8
    )

    return target



def offset_distribution_loss(
        logits,
        gt_offset,
        radius=8
):

    prob = torch.softmax(
        logits,
        dim=-1
    )


    target = gaussian_target(
        gt_offset,
        radius
    )


    kl = F.kl_div(
        torch.log(
            prob+1e-8
        ),
        target,
        reduction="batchmean"
    )


    entropy = (
        -prob*
        torch.log(
            prob+1e-8
        )
    ).sum(-1).mean()


    return (
        kl
        +
        0.01*entropy
    )
