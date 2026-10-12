
from .illumination import apply_illumination
from .noise import apply_shot_read_noise
from .flare import apply_flare

def nighttime_degradation(img, illumination="medium",
                          noise="medium",
                          flare_distance=None,
                          flare_color="white"):
    mask = None
    x=apply_illumination(img,illumination)

    if noise:
        x=apply_shot_read_noise(x,noise)
    
    if flare_distance:
        x, mask = apply_flare(
            x,
            flare_distance,
            flare_color
        )
    return x, mask

def generate_condition(img,condition):

    if condition=="illum_mild":
        out=nighttime_degradation(
            img,
            illumination="mild",
            noise=None,
            flare_distance=None
        )


    elif condition=="illum_medium":
        out=nighttime_degradation(
            img,
            illumination="medium",
            noise=None,
            flare_distance=None
        )


    elif condition=="illum_extreme":
        out=nighttime_degradation(
            img,
            illumination="extreme",
            noise=None,
            flare_distance=None
        )


    elif condition=="noise_medium":
        out=nighttime_degradation(
            img,
            illumination="medium",
            noise="medium",
            flare_distance=None
        )


    elif condition=="near_white_flare":
        out=nighttime_degradation(
            img,
            illumination="medium",
            noise="medium",
            flare_distance="near",
            flare_color="white"
        )


    elif condition=="far_white_flare":
        out=nighttime_degradation(
            img,
            illumination="medium",
            noise="medium",
            flare_distance="far",
            flare_color="white"
        )


    else:
        raise ValueError(condition)


    # 当前没有精确 flare mask
    # 先返回全0占位
    mask=torch.zeros_like(img[:,0:1])

    return out,mask