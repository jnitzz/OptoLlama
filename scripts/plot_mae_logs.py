#!/usr/bin/env python

from __future__ import annotations

import argparse
import html
import re

from pathlib import Path


MIN_PATTERN = re.compile(r"min test MAE:\s*([0-9.]+)")
LAST_PATTERN = re.compile(r"last test MAE:\s*([0-9.]+)")
VALIDATION_PATTERN = re.compile(
    r"\((?P<trigger>mid_epoch|epoch_end),\s*"
    r"epoch=(?P<epoch>\d+),\s*"
    r"epoch_samples=(?P<epoch_samples>\d+),\s*"
    r"total_samples=(?P<total_samples>\d+)\)"
)


COLORS = [
    ("#2563eb", "#f97316"),
    ("#16a34a", "#dc2626"),
    ("#7c3aed", "#0891b2"),
    ("#ca8a04", "#db2777"),
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot min and last test MAE values from training logs.")
    parser.add_argument("logs", nargs="+", type=Path, help="Training log file(s) to parse.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("plots/mae_min_last_plot.svg"),
        help="Output SVG path. Defaults to plots/mae_min_last_plot.svg.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels for the logs. Must match the number of log files.",
    )
    parser.add_argument("--title", default="Min and Last Test MAE by Epoch", help="Plot title.")
    return parser.parse_args()


def _position_from_metadata(point: dict[str, float], samples_per_epoch: int | None) -> float:
    epoch = int(point["epoch"])
    epoch_samples = int(point["epoch_samples"])
    if samples_per_epoch and samples_per_epoch > 0:
        progress = min(1.0, max(0.0, epoch_samples / samples_per_epoch))
        return (epoch - 1) + progress
    return float(epoch)


def parse_log(path: Path) -> tuple[list[float], list[float], list[float]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    points: list[dict[str, float]] = []
    pending: dict[str, float] | None = None
    pending_min: float | None = None

    for line in text.splitlines():
        validation_match = VALIDATION_PATTERN.search(line)
        if validation_match:
            pending = {
                "trigger": validation_match.group("trigger"),
                "epoch": float(validation_match.group("epoch")),
                "epoch_samples": float(validation_match.group("epoch_samples")),
                "total_samples": float(validation_match.group("total_samples")),
            }
            pending_min = None
            continue

        min_match = MIN_PATTERN.search(line)
        if min_match:
            pending_min = float(min_match.group(1))
            continue

        last_match = LAST_PATTERN.search(line)
        if last_match and pending is not None and pending_min is not None:
            point = dict(pending)
            point["min"] = pending_min
            point["last"] = float(last_match.group(1))
            points.append(point)
            pending = None
            pending_min = None

    if points:
        epoch_end_samples = [
            int(point["epoch_samples"])
            for point in points
            if point["trigger"] == "epoch_end" and int(point["epoch_samples"]) > 0
        ]
        samples_per_epoch = max(epoch_end_samples) if epoch_end_samples else None
        x_values = [_position_from_metadata(point, samples_per_epoch) for point in points]
        min_values = [point["min"] for point in points]
        last_values = [point["last"] for point in points]
        return x_values, min_values, last_values

    # Backward-compatible fallback for older logs that only printed the MAE
    # pairs once per epoch and did not include validation metadata.
    min_values = [float(match.group(1)) for match in MIN_PATTERN.finditer(text)]
    last_values = [float(match.group(1)) for match in LAST_PATTERN.finditer(text)]
    count = min(len(min_values), len(last_values))
    if count == 0:
        raise ValueError(f"No matching MAE pairs found in {path}")
    x_values = [float(index) for index in range(1, count + 1)]
    return x_values, min_values[:count], last_values[:count]


def line_path(x_values: list[float], values: list[float], x_scale, y_scale) -> str:
    points = []
    for index, (x_value, value) in enumerate(zip(x_values, values, strict=True)):
        command = "M" if index == 0 else "L"
        points.append(f"{command} {x_scale(x_value):.2f} {y_scale(value):.2f}")
    return " ".join(points)


def label_from_path(path: Path) -> str:
    name = path.name
    if name.startswith("log-"):
        name = name[4:]
    return path.with_name(name).stem


def render_svg(runs: list[dict], title: str, source_text: str) -> str:
    width = 1200
    height = 720
    margin = {"left": 82, "right": 250, "top": 72, "bottom": 78}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]
    max_epoch = max(max(run["x_values"]) for run in runs)
    values = [value for run in runs for value in run["min_values"] + run["last_values"]]
    y_min_raw = min(values)
    y_max_raw = max(values)
    y_pad = (y_max_raw - y_min_raw) * 0.08 or 0.01
    y_min = max(0.0, y_min_raw - y_pad)
    y_max = y_max_raw + y_pad

    def x_scale(epoch: float) -> float:
        denom = max(1.0, max_epoch - 1.0)
        return margin["left"] + ((epoch - 1.0) / denom) * plot_width

    def y_scale(value: float) -> float:
        return margin["top"] + (1 - (value - y_min) / (y_max - y_min)) * plot_height

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            f'<text x="{margin["left"]}" y="34" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="24" font-weight="700" fill="#111827">{html.escape(title)}</text>'
        ),
        (
            f'<text x="{margin["left"]}" y="58" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="13" fill="#4b5563">{html.escape(source_text)}</text>'
        ),
    ]

    y_ticks = 7
    for index in range(y_ticks + 1):
        value = y_min + (index / y_ticks) * (y_max - y_min)
        y_pos = y_scale(value)
        parts.append(
            f'<line x1="{margin["left"]}" x2="{margin["left"] + plot_width}" '
            f'y1="{y_pos:.2f}" y2="{y_pos:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin["left"] - 12}" y="{y_pos + 4:.2f}" text-anchor="end" '
            f'font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#374151">{value:.3f}</text>'
        )

    max_tick = int(max_epoch) if max_epoch == int(max_epoch) else int(max_epoch) + 1
    x_step = 1 if max_tick <= 20 else max(1, (max_tick + 9) // 10)
    for epoch in range(1, max_tick + 1, x_step):
        x_pos = x_scale(epoch)
        parts.append(
            f'<line x1="{x_pos:.2f}" x2="{x_pos:.2f}" y1="{margin["top"]}" '
            f'y2="{margin["top"] + plot_height}" stroke="#f3f4f6" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x_pos:.2f}" y="{margin["top"] + plot_height + 26}" text-anchor="middle" '
            f'font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#374151">{epoch}</text>'
        )

    parts.extend(
        [
            (
                f'<line x1="{margin["left"]}" x2="{margin["left"] + plot_width}" '
                f'y1="{margin["top"] + plot_height}" y2="{margin["top"] + plot_height}" '
                'stroke="#111827" stroke-width="1.5"/>'
            ),
            (
                f'<line x1="{margin["left"]}" x2="{margin["left"]}" y1="{margin["top"]}" '
                f'y2="{margin["top"] + plot_height}" stroke="#111827" stroke-width="1.5"/>'
            ),
            (
                f'<text x="{margin["left"] + plot_width / 2}" y="{height - 24}" text-anchor="middle" '
                'font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#111827">Epoch</text>'
            ),
            (
                f'<text x="22" y="{margin["top"] + plot_height / 2}" '
                f'transform="rotate(-90 22 {margin["top"] + plot_height / 2})" text-anchor="middle" '
                'font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#111827">Test MAE</text>'
            ),
        ]
    )

    for run in runs:
        parts.append(
            f'<path d="{line_path(run["x_values"], run["min_values"], x_scale, y_scale)}" fill="none" '
            f'stroke="{run["min_color"]}" stroke-width="2.5"/>'
        )
        parts.append(
            f'<path d="{line_path(run["x_values"], run["last_values"], x_scale, y_scale)}" fill="none" '
            f'stroke="{run["last_color"]}" stroke-width="2.5" stroke-dasharray="7 5"/>'
        )
        epoch = run["x_values"][-1]
        for key, color_key in (("min_values", "min_color"), ("last_values", "last_color")):
            value = run[key][-1]
            parts.append(
                f'<circle cx="{x_scale(epoch):.2f}" cy="{y_scale(value):.2f}" r="3.5" '
                f'fill="{run[color_key]}"/>'
            )

    legend_x = margin["left"] + plot_width + 28
    legend_y = margin["top"] + 8
    parts.append(
        f'<text x="{legend_x}" y="{legend_y}" font-family="Segoe UI, Arial, sans-serif" '
        'font-size="14" font-weight="700" fill="#111827">Legend</text>'
    )
    legend_y += 26
    for run in runs:
        entries = [
            (f'{run["label"]} min', run["min_color"], ""),
            (f'{run["label"]} last', run["last_color"], ' stroke-dasharray="7 5"'),
        ]
        for name, color, dash in entries:
            parts.append(
                f'<line x1="{legend_x}" x2="{legend_x + 34}" y1="{legend_y}" y2="{legend_y}" '
                f'stroke="{color}" stroke-width="2.5"{dash}/>'
            )
            parts.append(
                f'<text x="{legend_x + 44}" y="{legend_y + 4}" font-family="Segoe UI, Arial, sans-serif" '
                f'font-size="12.5" fill="#111827">{html.escape(name)}</text>'
            )
            legend_y += 22
        legend_y += 8

    legend_y += 12
    parts.append(
        f'<text x="{legend_x}" y="{legend_y}" font-family="Segoe UI, Arial, sans-serif" '
        'font-size="14" font-weight="700" fill="#111827">Final Values</text>'
    )
    legend_y += 22
    for run in runs:
        parts.append(
            f'<text x="{legend_x}" y="{legend_y}" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="12" fill="#374151">{html.escape(run["label"])}: epoch {run["x_values"][-1]:.2f}</text>'
        )
        legend_y += 18
        parts.append(
            f'<text x="{legend_x}" y="{legend_y}" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="12" fill="#374151">min {run["min_values"][-1]:.6f} | '
            f'last {run["last_values"][-1]:.6f}</text>'
        )
        legend_y += 26

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    args = parse_arguments()
    if args.labels is not None and len(args.labels) != len(args.logs):
        raise SystemExit("--labels must contain one label per log file")

    runs = []
    for index, log_path in enumerate(args.logs):
        x_values, min_values, last_values = parse_log(log_path)
        min_color, last_color = COLORS[index % len(COLORS)]
        runs.append(
            {
                "label": args.labels[index] if args.labels else label_from_path(log_path),
                "x_values": x_values,
                "min_values": min_values,
                "last_values": last_values,
                "min_color": min_color,
                "last_color": last_color,
            }
        )

    source_text = "Parsed from " + ", ".join(path.name for path in args.logs)
    svg = render_svg(runs, args.title, source_text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")

    for run in runs:
        print(
            f'{run["label"]}: points={len(run["x_values"])}, epoch={run["x_values"][-1]:.2f}, '
            f'min={run["min_values"][-1]:.6f}, last={run["last_values"][-1]:.6f}'
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
