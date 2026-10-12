"""Compatibility wrapper; use the joint pipeline for experiments."""
from .pipeline import nighttime_degradation


def apply_illumination(img, level='medium'):
    return nighttime_degradation(img,illumination=level,noise=None)[0]
