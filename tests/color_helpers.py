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
