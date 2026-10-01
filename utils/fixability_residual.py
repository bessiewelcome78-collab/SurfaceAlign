import torch


def build_fixability_weight(
        geometry_prob,
        gt_mask,
        action_weight,
        sigma=3.0):

    """
    Learn where refinement is useful,
    not only where prediction is wrong.
    """

    error = (
        (geometry_prob > 0.5)
        !=
        (gt_mask > 0.5)
    ).float()


    boundary = torch.exp(
        -torch.abs(
            geometry_prob-0.5
        ) / sigma
    )


    weight = (
        error *
        boundary *
        action_weight
    )


    return weight



def build_local_target(
        geometry_prob,
        gt_mask,
        weight):

    """
    Local reconstruction target.

    Not residual regression.
    Direct mask reconstruction.
    """

    target = gt_mask.clone()

    return (
        target,
        weight
    )
