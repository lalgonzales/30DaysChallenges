"""Map furniture shared by the day notebooks: scale bar and north arrow.

Both assume a projected CRS in metres (e.g. UTM) and an axes drawn with
`set_aspect("equal")`. Call them after the data is plotted and the limits set,
so the scale bar's length is picked from the final view.

Usage, from a notebook in maps/<year>/dayNN/:

    import sys
    sys.path.append("../..")  # maps/
    from cartography import add_north_arrow, add_scale_bar
"""

import math

import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def nice_length(width_m: float, share: float = 0.2) -> float:
    """A round length (1, 2 or 5 × 10^n m) close to `share` of the map width."""
    target = width_m * share
    power = 10 ** math.floor(math.log10(target))
    return max(step * power for step in (1, 2, 5) if step * power <= target)


def add_scale_bar(
    ax: plt.Axes,
    loc: str = "lower right",
    color: str = "black",
    background: str | None = None,
    share: float = 0.2,
) -> AnchoredSizeBar:
    """Add a scale bar in km (or m under 1 km), about `share` of the map width.

    `background` draws a box behind it, for maps whose base is not plain.
    """
    x0, x1 = ax.get_xlim()
    length = nice_length(abs(x1 - x0), share)
    label = f"{length / 1000:g} km" if length >= 1000 else f"{length:g} m"
    bar = AnchoredSizeBar(
        ax.transData,
        length,
        label,
        loc=loc,
        pad=0.6,
        borderpad=0.8,
        sep=4,
        color=color,
        frameon=background is not None,
        size_vertical=abs(x1 - x0) / 250,
        fontproperties=FontProperties(size=9),
    )
    if background is not None:
        bar.patch.set(facecolor=background, edgecolor="none", alpha=0.85)
    ax.add_artist(bar)
    return bar


def add_north_arrow(
    ax: plt.Axes,
    xy: tuple[float, float] = (0.95, 0.9),
    color: str = "black",
    halo: str | None = None,
    size: float = 0.06,
) -> None:
    """Add a north arrow with its tip at `xy` (axes fraction), `size` tall, N on top.

    The arrow points to grid north (up). In a UTM zone it differs from true
    north by under ~3°, too little to show at these scales.
    """
    x, y = xy
    effects = [patheffects.withStroke(linewidth=3, foreground=halo)] if halo else None
    ax.annotate(
        "",
        xy=(x, y),
        xytext=(x, y - size),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>,head_width=0.4,head_length=0.8", color=color, lw=1.5
        ),
    )
    ax.text(
        x,
        y + 0.005,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=color,
        path_effects=effects,
    )
