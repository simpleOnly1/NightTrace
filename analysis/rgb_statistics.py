import torch


def channel_saturation_ratio(
        img,
        threshold=0.98):

    """
    img:
    B,C,H,W
    range [0,1]

    return:
    R,G,B saturation ratio
    """

    sat = (
        img > threshold
    ).float()


    ratio = sat.mean(
        dim=(0,2,3)
    )


    return {
        "R_sat":ratio[0].item(),
        "G_sat":ratio[1].item(),
        "B_sat":ratio[2].item(),
        "RGB_sat":
            sat.all(dim=1)
            .float()
            .mean()
            .item()
    }

def channel_diversity(img):

    mean=img.mean(dim=1)

    std=img.std(dim=1)

    cdi = 1 - std / (mean + 0.05)

    return cdi.mean().item()