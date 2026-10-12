
Place this directory into NightTrace/analysis/.

Usage:
from analysis.degradation_v2.pipeline import nighttime_degradation

The order is:
illumination -> shot/read noise -> optical flare.

Flare distance controls PSF size:
near: large area and strong intensity
medium: normal
far: small source contribution.
