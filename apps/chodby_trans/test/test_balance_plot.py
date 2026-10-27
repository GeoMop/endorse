"""Tests for the interactive Flow123d balance viewer."""

import shutil
from pathlib import Path

from bokeh.models import HoverTool, MultiChoice, Scatter, Select

from chodby_trans.balance_plot import create_balance_tabs, read_balance_yaml, write_balance_html


BALANCE_DIRECTORY = Path(__file__).parent / "balance_plot"
SAMPLE_A = BALANCE_DIRECTORY / "sample_a"
SAMPLE_B = BALANCE_DIRECTORY / "sample_b"


def test_read_supplied_water_and_mass_balance_yaml() -> None:
    """Read the supplied Flow123d YAML files and expose their columns and time series."""
    water = read_balance_yaml(SAMPLE_A / "water_balance.yaml")
    mass = read_balance_yaml(SAMPLE_A / "mass_balance.yaml")

    assert water.column_names[1] == "flux_in"
    assert mass.column_names == water.column_names
    assert "ALL" in water.series_labels
    assert "ALL" in mass.series_labels
    assert water.curves()["flux_in"]["ALL"][0][0] == 0.0
    assert mass.curves()["mass"]["ALL"][0][:3] == [0.0, 100.0, 200.0]


def test_balance_tabs_have_quantity_and_multi_region_controls() -> None:
    """Build one tab per supplied file with the requested interactive selectors."""
    tabs = create_balance_tabs(SAMPLE_A)

    assert [panel.title for panel in tabs.tabs] == ["Water balance", "Mass balance"]
    for panel in tabs.tabs:
        quantity_select = panel.child.select_one({"type": Select})
        region_select = panel.child.select_one({"type": MultiChoice})
        hover = panel.child.select_one({"type": HoverTool})
        assert quantity_select.value == "flux_in"
        assert "flux_in" in quantity_select.options
        assert region_select.value == ["ALL"]
        assert len(region_select.options) > 1
        assert hover.tooltips == [
            ("sample", "@sample"),
            ("region", "@region"),
            ("time", "@time"),
            ("value", "@value"),
        ]
        assert len(hover.renderers) == 1
        assert isinstance(hover.renderers[0].glyph, Scatter)


def test_balance_tabs_compare_two_samples_using_distinct_colors() -> None:
    """Plot matching quantities from both supplied samples with one color per sample."""
    tabs = create_balance_tabs([SAMPLE_A, SAMPLE_B])
    hover = tabs.tabs[0].child.select_one({"type": HoverTool})
    point_data = hover.renderers[0].data_source.data

    assert [panel.title for panel in tabs.tabs] == ["Water balance", "Mass balance"]
    assert set(point_data["sample"]) == {"sample_a", "sample_b"}
    assert len(set(point_data["color"])) == 2


def test_write_balance_html_when_only_water_balance_is_available(tmp_path: Path) -> None:
    """Generate a standalone viewer when one of the two optional balance files is absent."""
    shutil.copyfile(SAMPLE_A / "water_balance.yaml", tmp_path / "water_balance.yaml")
    output_path = write_balance_html(tmp_path)

    html = output_path.read_text(encoding="utf-8")
    assert output_path.name == "balance_plot.html"
    assert "Water balance" in html
    assert "Mass balance" not in html
