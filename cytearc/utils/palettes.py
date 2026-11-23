"""The generated colors of categories, which plots draw and merges store.

Plots color categories without a stored color from these palettes, in the
order of their categories, and a merge gives the same colors to the
categories that it adds to a column's stored display, so both layers share
one copy of them.
"""

from typing import Any

from .logging import logger

# Okabe-Ito plus four high-contrast extensions for categorical figures.
COLORBLIND_PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
    "#6F4E7C",
    "#2E8B57",
    "#A05195",
    "#8C564B",
]

# Lifted from scanpy.plotting.palettes.
CUSTOM_PALETTES: dict[int, list[str]] = {
    10: [
        "#1f77b4",
        "#ff7f0e",
        "#279e68",
        "#d62728",
        "#aa40fc",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#b5bd61",
        "#17becf",
    ],
    20: [
        "#1f77b4",
        "#aec7e8",
        "#ff7f0e",
        "#ffbb78",
        "#2ca02c",
        "#98df8a",
        "#d62728",
        "#ff9896",
        "#9467bd",
        "#c5b0d5",
        "#8c564b",
        "#c49c94",
        "#e377c2",
        "#f7b6d2",
        "#7f7f7f",
        "#c7c7c7",
        "#bcbd22",
        "#dbdb8d",
        "#17becf",
        "#9edae5",
    ],
    28: [
        "#023fa5",
        "#7d87b9",
        "#bec1d4",
        "#d6bcc0",
        "#bb7784",
        "#8e063b",
        "#4a6fe3",
        "#8595e1",
        "#b5bbe3",
        "#e6afb9",
        "#e07b91",
        "#d33f6a",
        "#11c638",
        "#8dd593",
        "#c6dec7",
        "#ead3c6",
        "#f0b98d",
        "#ef9708",
        "#0fcfc0",
        "#9cded6",
        "#d5eae7",
        "#f3e1eb",
        "#f6c4e1",
        "#f79cd4",
        "#7f7f7f",
        "#c7c7c7",
        "#1CE6FF",
        "#336600",
    ],
    102: [
        "#FFFF00",
        "#1CE6FF",
        "#FF34FF",
        "#FF4A46",
        "#008941",
        "#006FA6",
        "#A30059",
        "#FFDBE5",
        "#7A4900",
        "#0000A6",
        "#63FFAC",
        "#B79762",
        "#004D43",
        "#8FB0FF",
        "#997D87",
        "#5A0007",
        "#809693",
        "#6A3A4C",
        "#1B4400",
        "#4FC601",
        "#3B5DFF",
        "#4A3B53",
        "#FF2F80",
        "#61615A",
        "#BA0900",
        "#6B7900",
        "#00C2A0",
        "#FFAA92",
        "#FF90C9",
        "#B903AA",
        "#D16100",
        "#DDEFFF",
        "#000035",
        "#7B4F4B",
        "#A1C299",
        "#300018",
        "#0AA6D8",
        "#013349",
        "#00846F",
        "#372101",
        "#FFB500",
        "#C2FFED",
        "#A079BF",
        "#CC0744",
        "#C0B9B2",
        "#C2FF99",
        "#001E09",
        "#00489C",
        "#6F0062",
        "#0CBD66",
        "#EEC3FF",
        "#456D75",
        "#B77B68",
        "#7A87A1",
        "#788D66",
        "#885578",
        "#FAD09F",
        "#FF8A9A",
        "#D157A0",
        "#BEC459",
        "#456648",
        "#0086ED",
        "#886F4C",
        "#34362D",
        "#B4A8BD",
        "#00A6AA",
        "#452C2C",
        "#636375",
        "#A3C8C9",
        "#FF913F",
        "#938A81",
        "#575329",
        "#00FECF",
        "#B05B6F",
        "#8CD0FF",
        "#3B9700",
        "#04F757",
        "#C8A1A1",
        "#1E6E00",
        "#7900D7",
        "#A77500",
        "#6367A9",
        "#A05837",
        "#6B002C",
        "#772600",
        "#D790FF",
        "#9B9700",
        "#549E79",
        "#FFF69F",
        "#201625",
        "#72418F",
        "#BC23FF",
        "#99ADC0",
        "#3A2465",
        "#922329",
        "#5B4534",
        "#FDE8DC",
        "#404E55",
        "#0089A3",
        "#CB7E98",
        "#A4E804",
        "#324E72",
    ],
}


def _seaborn() -> Any:
    """Import seaborn, whose husl palette spaces the hues of more colors than the palettes hold."""
    try:
        import seaborn
    except ImportError as error:
        raise ImportError(
            "Colors for more categories than CyteArc's palettes hold need seaborn. "
            "Install with: pip install 'cytearc[extra]'"
        ) from error
    return seaborn


def palette_for_n(
    n: int,
    *,
    palette_name: str = "default",
) -> list[str]:
    """Return ``n`` distinct colors of a generated palette, in its order.

    ``"default"`` takes the first ``n`` colors of the smallest of Scanpy's
    palettes of 10, 20, 28 and 102 colors that holds them, and
    ``"colorblind"`` the first of its 12 colors. Beyond them, the colors are
    evenly spaced hues of seaborn's husl palette, which needs seaborn.

    Raises:
        ValueError: If ``palette_name`` is neither ``"default"`` nor
            ``"colorblind"``.
        ImportError: If more colors than the palette holds are asked for and
            seaborn is not installed.
    """
    if palette_name not in ("default", "colorblind"):
        raise ValueError("palette_name must be 'default' or 'colorblind'")
    if palette_name == "colorblind":
        if n <= len(COLORBLIND_PALETTE):
            return list(COLORBLIND_PALETTE[:n])
        logger.warning(
            f"Requested {n} colorblind-safe colors but only "
            f"{len(COLORBLIND_PALETTE)} are available; "
            "falling back to evenly spaced hues that are distinct but not "
            "guaranteed colorblind safe"
        )
        sns = _seaborn()
        return list(sns.color_palette("husl", n_colors=n).as_hex())
    if n <= 10:
        return list(CUSTOM_PALETTES[10][:n])
    if n <= 20:
        return list(CUSTOM_PALETTES[20][:n])
    if n <= 28:
        return list(CUSTOM_PALETTES[28][:n])
    if n <= 102:
        return list(CUSTOM_PALETTES[102][:n])
    sns = _seaborn()
    return list(sns.color_palette("husl", n_colors=n).as_hex())
