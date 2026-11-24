"""The contrast of text on a colored background, as WCAG 2.2 defines it."""

import numpy as np
from matplotlib.colors import to_rgb

# The twelve colors of the colorblind palette and the color of text on each,
# the one of higher WCAG contrast: white on blue, black, the purples and
# brown, black on the others.
COLORBLIND_TEXT_COLORS = {
    "#0072B2": "#ffffff",
    "#E69F00": "#000000",
    "#009E73": "#000000",
    "#D55E00": "#000000",
    "#CC79A7": "#000000",
    "#56B4E9": "#000000",
    "#F0E442": "#000000",
    "#000000": "#ffffff",
    "#6F4E7C": "#ffffff",
    "#2E8B57": "#000000",
    "#A05195": "#ffffff",
    "#8C564B": "#ffffff",
}

# The relative luminance of colors that labels are drawn on, and the WCAG 2.2
# contrast ratio of black and of white text on each, worked out by hand in
# 50-digit decimal arithmetic and checked against WebAIM's contrast checker,
# which truncates to two decimals.
LABEL_CONTRASTS = {
    # Issue #595's composition colors: the showcase's blue, red and orange,
    # and four colors of the colorblind palette.
    "#4e79a7": (0.180845, 4.6169, 4.5485),
    "#e15759": (0.235452, 5.7090, 3.6784),
    "#f28e2b": (0.383977, 8.6795, 2.4195),
    "#009e73": (0.256915, 6.1383, 3.4211),
    "#d55e00": (0.221515, 5.4303, 3.8672),
    "#cc79a7": (0.293021, 6.8604, 3.0610),
    "#2e8b57": (0.197342, 4.9468, 4.2451),
    # Issue #595's viridis colors of the clone and repertoire matrices.
    "#25838d": (0.185490, 4.7098, 4.4588),
    "#20908c": (0.221470, 5.4294, 3.8678),
    "#20a485": (0.285515, 6.7103, 3.1295),
    "#23878d": (0.196084, 4.9217, 4.2668),
    "#25ac82": (0.315101, 7.3020, 2.8759),
    # Black and white reach the same ratio, the square root of 21, at a
    # luminance of 0.1791288; these colors lie on either side of it.
    "#747474": (0.174647, 4.4929, 4.6740),
    "#757575": (0.177888, 4.5578, 4.6075),
    "#767676": (0.181164, 4.6233, 4.5422),
    "#26828e": (0.183303, 4.6661, 4.5006),
    # viridis' colors at indices 110 and 111, rounded to 8 bits.
    "#27808e": (0.178227, 4.5645, 4.6007),
    "#26818e": (0.180656, 4.6131, 4.5522),
    # The 8-bit colors nearest the crossover: 6.5e-8 below it and 6.0e-9 above.
    "#766cb5": (0.179128719, 4.5825744, 4.5825770),
    "#cf0dcc": (0.179128791, 4.5825758, 4.5825756),
}


def higher_contrast_text(color: str) -> str:
    """Black or white, whichever ``LABEL_CONTRASTS`` gives the higher ratio on ``color``."""
    _luminance, black, white = LABEL_CONTRASTS[color]
    return "#000000" if black > white else "#ffffff"


def relative_luminance(color: object) -> float:
    """The relative luminance of a color, from its linearized sRGB channels."""
    channels = np.asarray(to_rgb(color))
    linear = np.where(
        channels <= 0.04045, channels / 12.92, ((channels + 0.055) / 1.055) ** 2.4
    )
    return float(linear @ np.array([0.2126, 0.7152, 0.0722]))


def wcag_contrast(first: object, second: object) -> float:
    """The contrast ratio of two colors, (L1 + 0.05) / (L2 + 0.05), the lighter's luminance first."""
    lighter, darker = sorted(
        (relative_luminance(first), relative_luminance(second)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)
