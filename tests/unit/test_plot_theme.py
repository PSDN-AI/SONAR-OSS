import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from plotnine import aes, geom_point, ggplot

from psdn_sonar.utils.plot_theme import (
    SWARM_COLORS,
    SWARM_COLORS_EXTENDED,
    get_swarm_colors,
    save_plot,
    theme_swarm_lab,
)


class TestPlotTheme:
    def test_swarm_colors_count(self):
        assert len(SWARM_COLORS) == 6
        assert len(SWARM_COLORS_EXTENDED) == 14

    def test_get_swarm_colors_basic(self):
        colors = get_swarm_colors(3)
        assert len(colors) == 3
        assert all(c.startswith("#") for c in colors)

    def test_get_swarm_colors_extended(self):
        colors = get_swarm_colors(10)
        assert len(colors) == 10
        assert len(set(colors)) == 10

    def test_get_swarm_colors_cycling(self):
        colors = get_swarm_colors(20)
        assert len(colors) == 20

    def test_theme_swarm_lab_returns_theme(self):
        theme = theme_swarm_lab()
        assert theme is not None

    def test_theme_swarm_lab_custom_size(self):
        theme = theme_swarm_lab(base_size=16, figure_size=(8, 6))
        assert theme is not None

    def test_save_plot_creates_file(self, tmp_path):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        plot = ggplot(df, aes("x", "y")) + geom_point()

        output_path = tmp_path / "test_plot.png"
        save_plot(plot, str(output_path), dpi=100, width=5, height=5)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_colors_are_hex_format(self):
        colors = get_swarm_colors(5)
        for color in colors:
            assert color.startswith("#")
            assert len(color) == 7
            int(color[1:], 16)
