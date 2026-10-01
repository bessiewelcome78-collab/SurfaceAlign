import torch
import torch.nn.functional as F


def synthetic_error(mask):

    r=torch.rand(1).item()

    if r < 0.25:

        return F.max_pool2d(
            mask,
            5,
            1,
            2
        )


    if r < 0.5:

        return 1-F.max_pool2d(
            1-mask,
            5,
            1,
            2
        )


    if r < 0.75:

        return torch.roll(
            mask,
            shifts=2,
            dims=-1
        )


    out=mask.clone()

    h,w=mask.shape[-2:]

    out[
        ...,
        h//3:h//3+8,
        w//3:w//3+8
    ]=0

    return out
