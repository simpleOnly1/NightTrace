"""Compatibility wrapper; noise belongs after optical radiance in joint use."""
from .pipeline import nighttime_degradation


def apply_shot_read_noise(img, level='medium'):
    return nighttime_degradation(img,illumination=None,noise=level)[0]
