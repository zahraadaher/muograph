from typing import Tuple

# Number of bins for histograms
n_bins: int = 50

# Histogram transparency
alpha: float = 0.6
alpha_sns: float = 0.4

# Distance unit
d_unit: str = "mm"

# Matplotlib default font
fontsize: int = 16
labelsize: int = 16
titlesize: int = 20

fontweigh: str = "normal"
font = {"weight": "normal", "size": fontsize, "family": "sans-serif"}

# Matplotlib colormaps
cmap: str = "jet"

# Histogram XY ratio
xy_golden_ratio: float = 1.5
hist_scale = 6
hist_figsize: Tuple[float, float] = (xy_golden_ratio * hist_scale, 1 * hist_scale)
hist2_figsize: Tuple[float, float] = (2 * xy_golden_ratio * hist_scale, 1 * hist_scale)

# tracking plot
tracking_figsize: Tuple[float, float] = (10, 5)

# Scale of matplotlib figures with subplots
scale: int = 3

# 2D histogram
n_bins_2D = 100
