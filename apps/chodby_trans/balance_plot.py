"""Create an interactive standalone viewer for Flow123d balance YAML files."""

from __future__ import annotations

import argparse
import logging
import webbrowser
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml
from bokeh.embed import file_html
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CustomJS, HoverTool, MultiChoice, Select, TabPanel, Tabs
from bokeh.palettes import Category10, Turbo256
from bokeh.plotting import figure
from bokeh.resources import INLINE


BALANCE_FILES = (
    ("Water balance", "water_balance.yaml"),
    ("Mass balance", "mass_balance.yaml"),
)


@dataclass(frozen=True)
class BalanceRecord:
    """One Flow123d balance row with values aligned to the file's column names."""

    time: float
    region: str
    physical_quantity: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class BalanceData:
    """Validated balance columns and records read from one Flow123d YAML file."""

    path: Path
    column_names: tuple[str, ...]
    records: tuple[BalanceRecord, ...]

    @property
    def series_labels(self) -> tuple[str, ...]:
        """Return stable labels for every region and physical-quantity pair."""
        quantities = {record.physical_quantity for record in self.records}
        labels = dict.fromkeys(
            self._series_label(record.region, record.physical_quantity, len(quantities) > 1)
            for record in self.records
        )
        return tuple(labels)

    def curves(self) -> dict[str, dict[str, tuple[list[float], list[float]]]]:
        """Return time and value arrays indexed by balance column and series label."""
        multiple_quantities = len({record.physical_quantity for record in self.records}) > 1
        grouped: dict[str, dict[str, list[tuple[float, float]]]] = {
            column_name: defaultdict(list) for column_name in self.column_names
        }
        for record in self.records:
            label = self._series_label(record.region, record.physical_quantity, multiple_quantities)
            for column_name, value in zip(self.column_names, record.values, strict=True):
                grouped[column_name][label].append((record.time, value))

        return {
            column_name: {
                label: ([point[0] for point in sorted(points)], [point[1] for point in sorted(points)])
                for label, points in series.items()
            }
            for column_name, series in grouped.items()
        }

    @staticmethod
    def _series_label(region: str, physical_quantity: str, show_quantity: bool) -> str:
        return f"{region} [{physical_quantity}]" if show_quantity else region


def read_balance_yaml(path: Path) -> BalanceData:
    """Read and validate a Flow123d balance YAML file."""
    with path.open("r", encoding="utf-8") as stream:
        raw: Any = yaml.safe_load(stream)

    if not isinstance(raw, dict):
        raise ValueError(f"Balance file must contain a mapping: {path}")
    try:
        raw_columns = raw["column_names"]
        raw_records = raw["data"]
    except KeyError as exc:
        raise ValueError(f"Balance file is missing required key {exc.args[0]!r}: {path}") from exc

    if not isinstance(raw_columns, list) or not raw_columns:
        raise ValueError(f"Balance column_names must be a non-empty list: {path}")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError(f"Balance data must be a non-empty list: {path}")

    column_names = tuple(str(name).strip() for name in raw_columns)
    if len(set(column_names)) != len(column_names):
        raise ValueError(f"Balance column_names must be unique: {path}")

    records = []
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            raise ValueError(f"Balance data record {index} must be a mapping: {path}")
        try:
            values = tuple(float(value) for value in raw_record["data"])
            record = BalanceRecord(
                time=float(raw_record["time"]),
                region=str(raw_record["region"]),
                physical_quantity=str(raw_record["quantity"]),
                values=values,
            )
        except KeyError as exc:
            raise ValueError(
                f"Balance data record {index} is missing required key {exc.args[0]!r}: {path}"
            ) from exc
        if len(values) != len(column_names):
            raise ValueError(
                f"Balance data record {index} has {len(values)} values for {len(column_names)} columns: {path}"
            )
        records.append(record)

    return BalanceData(path=path, column_names=column_names, records=tuple(records))


def create_balance_panel(samples: Sequence[tuple[str, BalanceData]], title: str) -> TabPanel:
    """Create one interactive tab containing one or two parsed balance samples."""
    column_names = samples[0][1].column_names
    for sample_name, balance in samples[1:]:
        if balance.column_names != column_names:
            raise ValueError(
                f"Balance columns in sample {sample_name!r} do not match {samples[0][0]!r}."
            )

    labels = tuple(dict.fromkeys(label for _name, balance in samples for label in balance.series_labels))
    selected_metric = "flux_in" if "flux_in" in column_names else column_names[0]
    selected_regions = ["ALL"] if "ALL" in labels else [labels[0]]
    region_colors = {
        region: Turbo256[round(index * (len(Turbo256) - 1) / max(len(labels) - 1, 1))]
        for index, region in enumerate(labels)
    }
    sample_colors = {
        sample_name: Category10[10][index]
        for index, (sample_name, _balance) in enumerate(samples)
    }
    dashes = ("solid", "dashed", "dotted", "dotdash", "dashdot")
    markers = ("circle", "square", "triangle", "diamond", "inverted_triangle", "hex")
    dash_by_region = {region: dashes[index % len(dashes)] for index, region in enumerate(labels)}
    marker_by_region = {region: markers[index % len(markers)] for index, region in enumerate(labels)}
    sample_curves = [
        (sample_name, balance, balance.curves())
        for sample_name, balance in samples
    ]

    rows = [
        {
            "metric": metric,
            "sample": sample_name,
            "region": region,
            "times": curves[metric][region][0],
            "values": curves[metric][region][1],
            "color": sample_colors[sample_name] if len(samples) > 1 else region_colors[region],
            "dash": dash_by_region[region],
            "marker": marker_by_region[region],
            "legend": f"{sample_name}: {region}" if len(samples) > 1 else region,
        }
        for sample_name, balance, curves in sample_curves
        for metric in column_names
        for region in balance.series_labels
    ]

    all_data = ColumnDataSource(
        data={key: [item[key] for item in rows] for key in rows[0]}
    )

    selected_rows = [
        item
        for item in rows
        if item["metric"] == selected_metric and item["region"] in selected_regions
    ]

    line_source = ColumnDataSource(
        data={
            "xs": [item["times"] for item in selected_rows],
            "ys": [item["values"] for item in selected_rows],
            "sample": [item["sample"] for item in selected_rows],
            "region": [item["region"] for item in selected_rows],
            "color": [item["color"] for item in selected_rows],
            "dash": [item["dash"] for item in selected_rows],
            "legend": [item["legend"] for item in selected_rows],
        }
    )
    point_source = ColumnDataSource(
        data={
            "time": [time for item in selected_rows for time in item["times"]],
            "value": [value for item in selected_rows for value in item["values"]],
            "sample": [item["sample"] for item in selected_rows for _time in item["times"]],
            "region": [item["region"] for item in selected_rows for _time in item["times"]],
            "color": [item["color"] for item in selected_rows for _time in item["times"]],
            "marker": [item["marker"] for item in selected_rows for _time in item["times"]],
        }
    )
    plot = figure(
        title=f"{title}: {selected_metric}",
        x_axis_label="Time",
        y_axis_label=selected_metric,
        sizing_mode="stretch_width",
        height=620,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    plot.multi_line(
        xs="xs",
        ys="ys",
        source=line_source,
        color="color",
        line_dash="dash",
        line_width=2,
        legend_field="legend",
    )
    points = plot.scatter(
        x="time",
        y="value",
        source=point_source,
        color="color",
        marker="marker",
        line_color="white",
        line_width=0.75,
        size=7,
    )
    plot.add_tools(
        HoverTool(
            renderers=[points],
            tooltips=[
                ("sample", "@sample"),
                ("region", "@region"),
                ("time", "@time"),
                ("value", "@value"),
            ],
        )
    )
    plot.legend.click_policy = "hide"
    plot.legend.location = "top_left"

    metric_select = Select(title="Balance quantity", value=selected_metric, options=list(column_names))
    region_select = MultiChoice(title="Regions", value=selected_regions, options=list(labels))
    callback = CustomJS(
        args={
            "all_source": all_data,
            "metric_select": metric_select,
            "plot": plot,
            "region_select": region_select,
            "line_source": line_source,
            "point_source": point_source,
            "tab_title": title,
            "title_model": plot.title,
            "y_axis": plot.yaxis[0],
        },
        code="""
            const metric = metric_select.value;
            const selected = new Set(region_select.value);
            const data = all_source.data;
            const lines = {xs: [], ys: [], sample: [], region: [], color: [], dash: [], legend: []};
            const points = {time: [], value: [], sample: [], region: [], color: [], marker: []};
            for (let index = 0; index < data.metric.length; ++index) {
                if (data.metric[index] === metric && selected.has(data.region[index])) {
                    const region = data.region[index];
                    const color = data.color[index];
                    const sample = data.sample[index];
                    const times = data.times[index];
                    const values = data.values[index];
                    lines.xs.push(times);
                    lines.ys.push(values);
                    lines.sample.push(sample);
                    lines.region.push(region);
                    lines.color.push(color);
                    lines.dash.push(data.dash[index]);
                    lines.legend.push(data.legend[index]);
                    for (let point = 0; point < times.length; ++point) {
                        points.time.push(times[point]);
                        points.value.push(values[point]);
                        points.sample.push(sample);
                        points.region.push(region);
                        points.color.push(color);
                        points.marker.push(data.marker[index]);
                    }
                }
            }
            line_source.data = lines;
            point_source.data = points;
            y_axis.axis_label = metric;
            title_model.text = `${tab_title}: ${metric}`;
        """,
    )
    metric_select.js_on_change("value", callback)
    region_select.js_on_change("value", callback)

    controls = row(metric_select, region_select, sizing_mode="stretch_width")
    return TabPanel(child=column(controls, plot, sizing_mode="stretch_width"), title=title)


def _normalize_directories(balance_directories: Path | Sequence[Path]) -> tuple[Path, ...]:
    directories = (balance_directories,) if isinstance(balance_directories, Path) else tuple(balance_directories)
    if not 1 <= len(directories) <= 2:
        raise ValueError(f"Expected one or two balance directories, got {len(directories)}.")
    return directories


def create_balance_tabs(balance_directories: Path | Sequence[Path]) -> Tabs:
    """Create tabs comparing supported balance files from one or two directories."""
    directories = _normalize_directories(balance_directories)
    directory_names = [directory.name or str(directory) for directory in directories]
    if len(set(directory_names)) != len(directory_names):
        directory_names = [f"sample {index + 1}" for index in range(len(directories))]

    panels = []
    for title, filename in BALANCE_FILES:
        samples = [
            (sample_name, read_balance_yaml(path))
            for sample_name, directory in zip(directory_names, directories, strict=True)
            if (path := directory / filename).is_file()
        ]
        if samples:
            panels.append(create_balance_panel(samples, title))
    if not panels:
        expected = ", ".join(filename for _title, filename in BALANCE_FILES)
        searched = ", ".join(str(directory) for directory in directories)
        raise FileNotFoundError(f"No balance YAML file found in {searched}; expected {expected}.")
    return Tabs(tabs=panels, sizing_mode="stretch_width")


def write_balance_html(
    balance_directories: Path | Sequence[Path],
    output_path: Path | None = None,
) -> Path:
    """Write a self-contained interactive HTML viewer and return its path."""
    directories = _normalize_directories(balance_directories)
    output_path = output_path or directories[0] / "balance_plot.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tabs = create_balance_tabs(directories)
    output_path.write_text(file_html(tabs, INLINE, "Flow123d balance"), encoding="utf-8")
    logging.info("Wrote Flow123d balance viewer to %s", output_path)
    return output_path


def main() -> None:
    """Parse command-line arguments and write the balance viewer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "balance_directories",
        nargs="+",
        type=Path,
        metavar="BALANCE_DIRECTORY",
        help="One or two directories containing Flow123d balance YAML files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HTML path; defaults to BALANCE_DIRECTORY/balance_plot.html.",
    )
    parser.add_argument("--show", action="store_true", help="Open the generated viewer in the default browser.")
    args = parser.parse_args()
    if len(args.balance_directories) > 2:
        parser.error("at most two BALANCE_DIRECTORY arguments are supported")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output_path = write_balance_html(args.balance_directories, args.output)
    if args.show:
        webbrowser.open(output_path.resolve().as_uri())


if __name__ == "__main__":
    main()
