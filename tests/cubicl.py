"""pmkmp 'CubicL' (Matteo Niccoli, perceptually improved colormaps):
Lab-based rainbow, cubic-law luminance, red at high end. Base 16-color
table from pmkmp.m release 4.04; interpolated like the MATLAB original."""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
_BASE = np.array([
 [0.4706,0.0000,0.5216],[0.5137,0.0527,0.7096],[0.4942,0.2507,0.8781],
 [0.4296,0.3858,0.9922],[0.3691,0.5172,0.9495],[0.2963,0.6191,0.8515],
 [0.2199,0.7134,0.7225],[0.2643,0.7836,0.5756],[0.3094,0.8388,0.4248],
 [0.3623,0.8917,0.2858],[0.5200,0.9210,0.3137],[0.6800,0.9255,0.3386],
 [0.8000,0.9255,0.3529],[0.8706,0.8549,0.3608],[0.9514,0.7466,0.3686],
 [0.9765,0.5887,0.3569]])
cubicL = LinearSegmentedColormap.from_list('CubicL', _BASE, N=256)
