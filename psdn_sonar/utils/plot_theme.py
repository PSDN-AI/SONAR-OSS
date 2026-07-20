"""
Plotnine theme matching Swarm Lab visualization guidelines.

This module provides a custom plotnine theme that replicates the visual style
of the Swarm Lab's swarmviz package, ensuring consistent aesthetics across
all visualizations.

Design Guidelines:
- Dark grid background
- Bold, large typography
- High DPI (600) for publication quality
- Thick lines (width=3) and large markers
- Specific color palette from XKCD colors
"""

import logging
from typing import Optional

import plotnine as p9

logger = logging.getLogger(__name__)

# Swarm Lab Color Palette - Professional shades
SWARM_COLORS = [
    "#2E5090",  # deep blue
    "#2D8B57",  # forest green
    "#C84848",  # muted red
    "#D4A017",  # dark gold
    "#6B7280",  # slate gray
    "#7B5D8A",  # muted purple
]

# Additional colors for extended palettes (10+ distinct colors for multi-model plots)
SWARM_COLORS_EXTENDED = SWARM_COLORS + [
    "#D97020",  # burnt orange
    "#2B7A78",  # teal
    "#8B5A3C",  # brown
    "#6A7FDB",  # soft blue
    "#A85F8F",  # mauve
    "#4A7C7E",  # blue-gray
    "#9B7EBD",  # lavender
    "#CC8B3C",  # copper
]


def theme_swarm_lab(
    base_size: int = 20,
    figure_size: tuple[float, float] = (10, 10),
    grid_color: str = "#CCCCCC",
    grid_alpha: float = 0.5,
) -> p9.theme:
    """
    Create a custom plotnine theme matching Swarm Lab visualization guidelines.

    Parameters
    ----------
    base_size : int, default=20
        Base font size for text elements. Other sizes scale from this.
    figure_size : tuple[float, float], default=(10, 10)
        Figure dimensions in inches (width, height).
    grid_color : str, default="#CCCCCC"
        Color for grid lines.
    grid_alpha : float, default=0.5
        Transparency for grid lines (0=transparent, 1=opaque).

    Returns
    -------
    plotnine.theme
        A plotnine theme object with Swarm Lab styling.

    Examples
    --------
    >>> from plotnine import ggplot, aes, geom_point
    >>> from psdn_sonar.utils.plot_theme import theme_swarm_lab, SWARM_COLORS
    >>>
    >>> plot = (
    ...     ggplot(df, aes(x='x', y='y', color='group'))
    ...     + geom_point(size=5)
    ...     + theme_swarm_lab()
    ... )
    """
    return p9.theme_seaborn(style="darkgrid", context="talk") + p9.theme(
        # Typography - Bold and Large
        text=p9.element_text(family="sans-serif", weight="bold", size=base_size, color="black"),
        # Plot title
        plot_title=p9.element_text(
            size=base_size * 1.3,  # 26pt when base=20
            ha="center",
            weight="bold",
            margin={"b": 15},
        ),
        # Axis titles
        axis_title_x=p9.element_text(
            size=base_size,  # 20pt
            weight="bold",
            margin={"t": 10},
        ),
        axis_title_y=p9.element_text(
            size=base_size,  # 20pt
            weight="bold",
            margin={"r": 10},
        ),
        # Axis tick labels
        axis_text_x=p9.element_text(
            size=base_size * 0.7,  # 14pt when base=20
            weight="bold",
            rotation=0,
        ),
        axis_text_y=p9.element_text(
            size=base_size * 0.7,  # 14pt when base=20
            weight="bold",
        ),
        # Legend - Larger size
        legend_text=p9.element_text(
            size=base_size * 0.75,  # 15pt when base=20
            weight="normal",
        ),
        legend_title=p9.element_text(
            size=base_size * 0.85,  # 17pt when base=20
            weight="bold",
        ),
        legend_position="right",
        legend_background=p9.element_rect(fill="white", color="black", size=1),
        legend_key=p9.element_rect(fill="white", color="white"),
        legend_key_size=12,
        # Facet labels (for facet_wrap/facet_grid)
        strip_text=p9.element_text(size=base_size, weight="bold", color="black"),
        strip_background=p9.element_rect(fill="#E5E5E5", color="black", size=1),
        # Panel and background
        panel_background=p9.element_rect(fill="#EBEBEB"),
        panel_grid_major=p9.element_line(color=grid_color, size=0.8, alpha=grid_alpha),
        panel_grid_minor=p9.element_line(color=grid_color, size=0.4, alpha=grid_alpha * 0.5),
        panel_border=p9.element_rect(color="black", size=1.5, fill="none"),
        # Figure size
        figure_size=figure_size,
        # Plot background
        plot_background=p9.element_rect(fill="white"),
        # Aspect ratio
        aspect_ratio=None,  # Allow flexible aspect ratio
    )


def save_plot(
    plot: p9.ggplot,
    filename: str,
    dpi: int = 600,
    width: float = 10,
    height: float = 10,
    verbose: bool = True,
) -> None:
    """
    Save a plotnine plot with Swarm Lab standards (600 DPI, publication quality).

    Parameters
    ----------
    plot : plotnine.ggplot
        The plot object to save.
    filename : str
        Output filename (e.g., 'my_plot.png').
    dpi : int, default=600
        Dots per inch for output resolution. 600 is publication standard.
    width : float, default=10
        Figure width in inches.
    height : float, default=10
        Figure height in inches.
    verbose : bool, default=True
        If True, print confirmation message.

    Examples
    --------
    >>> from psdn_sonar.utils.plot_theme import save_plot
    >>> save_plot(my_plot, "results/figure1.png", dpi=600, width=12, height=8)
    """
    plot.save(filename=filename, dpi=dpi, width=width, height=height, verbose=False)
    if verbose:
        logger.info("✓ Saved plot: %s (%sx%s in, %s DPI)", filename, width, height, dpi)


def get_swarm_colors(n: Optional[int] = None) -> list[str]:
    """
    Get Swarm Lab color palette.

    Parameters
    ----------
    n : int, optional
        Number of colors needed. If None, returns all colors.
        If n > available colors, cycles through the palette.

    Returns
    -------
    list[str]
        List of hex color codes.

    Examples
    --------
    >>> colors = get_swarm_colors(3)
    >>> print(colors)
    ['#3B5B92', '#39AD48', '#D9544D']
    """
    if n is None:
        return SWARM_COLORS.copy()

    if n <= len(SWARM_COLORS_EXTENDED):
        return SWARM_COLORS_EXTENDED[:n]

    # Cycle through colors if more are needed
    colors = []
    for i in range(n):
        colors.append(SWARM_COLORS_EXTENDED[i % len(SWARM_COLORS_EXTENDED)])
    return colors


# Convenience function for common plot types
def create_line_plot_base(
    data,
    x: str,
    y: str,
    color: Optional[str] = None,
    title: str = "",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    line_size: float = 3.0,
) -> p9.ggplot:
    """
    Create a base line plot with Swarm Lab styling.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to plot.
    x : str
        Column name for x-axis.
    y : str
        Column name for y-axis.
    color : str, optional
        Column name for color grouping.
    title : str, default=""
        Plot title.
    x_label : str, optional
        X-axis label (defaults to column name).
    y_label : str, optional
        Y-axis label (defaults to column name).
    line_size : float, default=3.0
        Line width (Swarm Lab guideline: 3.0).

    Returns
    -------
    plotnine.ggplot
        A plotnine plot object ready for additional layers or saving.
    """
    aes_mapping = p9.aes(x=x, y=y)
    if color:
        aes_mapping = p9.aes(x=x, y=y, color=color)

    plot = p9.ggplot(data, aes_mapping)
    plot += p9.geom_line(size=line_size)

    if color:
        n_groups = data[color].nunique()
        plot += p9.scale_color_manual(values=get_swarm_colors(n_groups))

    plot += p9.labs(title=title, x=x_label or x, y=y_label or y)
    plot += theme_swarm_lab()

    return plot


def create_bar_plot_base(
    data,
    x: str,
    y: str,
    fill: Optional[str] = None,
    title: str = "",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    alpha: float = 0.85,
) -> p9.ggplot:
    """
    Create a base bar plot with Swarm Lab styling.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to plot.
    x : str
        Column name for x-axis (categories).
    y : str
        Column name for y-axis (values).
    fill : str, optional
        Column name for fill color grouping.
    title : str, default=""
        Plot title.
    x_label : str, optional
        X-axis label (defaults to column name).
    y_label : str, optional
        Y-axis label (defaults to column name).
    alpha : float, default=0.85
        Bar transparency (0=transparent, 1=opaque).

    Returns
    -------
    plotnine.ggplot
        A plotnine plot object ready for additional layers or saving.
    """
    aes_mapping = p9.aes(x=x, y=y)
    if fill:
        aes_mapping = p9.aes(x=x, y=y, fill=fill)

    plot = p9.ggplot(data, aes_mapping)
    plot += p9.geom_bar(stat="identity", alpha=alpha, color="black", size=0.5)

    if fill:
        n_groups = data[fill].nunique()
        plot += p9.scale_fill_manual(values=get_swarm_colors(n_groups))

    plot += p9.labs(title=title, x=x_label or x, y=y_label or y)
    plot += theme_swarm_lab()

    return plot


def create_scatter_plot_base(
    data,
    x: str,
    y: str,
    color: Optional[str] = None,
    size: Optional[str] = None,
    title: str = "",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    point_size: float = 5.0,
) -> p9.ggplot:
    """
    Create a base scatter plot with Swarm Lab styling.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to plot.
    x : str
        Column name for x-axis.
    y : str
        Column name for y-axis.
    color : str, optional
        Column name for color grouping.
    size : str, optional
        Column name for point size mapping.
    title : str, default=""
        Plot title.
    x_label : str, optional
        X-axis label (defaults to column name).
    y_label : str, optional
        Y-axis label (defaults to column name).
    point_size : float, default=5.0
        Point size (Swarm Lab guideline: large markers).

    Returns
    -------
    plotnine.ggplot
        A plotnine plot object ready for additional layers or saving.
    """
    aes_mapping = p9.aes(x=x, y=y)
    if color and size:
        aes_mapping = p9.aes(x=x, y=y, color=color, size=size)
    elif color:
        aes_mapping = p9.aes(x=x, y=y, color=color)
    elif size:
        aes_mapping = p9.aes(x=x, y=y, size=size)

    plot = p9.ggplot(data, aes_mapping)

    if size:
        plot += p9.geom_point(alpha=0.8)
    else:
        plot += p9.geom_point(size=point_size, alpha=0.8)

    if color:
        n_groups = data[color].nunique()
        plot += p9.scale_color_manual(values=get_swarm_colors(n_groups))

    plot += p9.labs(title=title, x=x_label or x, y=y_label or y)
    plot += theme_swarm_lab()

    return plot
