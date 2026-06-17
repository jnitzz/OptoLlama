#!/usr/bin/env python

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Callable


COLORS = {
    "blue": "#2563eb",
    "orange": "#f97316",
    "green": "#16a34a",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "teal": "#0891b2",
    "amber": "#ca8a04",
    "pink": "#db2777",
    "gray": "#6b7280",
}


def parse_arguments() -> argparse.Namespace:
    default_input = Path.home() / "Downloads" / "depth-field-history.json"
    parser = argparse.ArgumentParser(description="Render an SVG overview of depth-field training history trends.")
    parser.add_argument(
        "history",
        nargs="?",
        type=Path,
        default=default_input,
        help=f"History JSON path. Defaults to {default_input}.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("plots/depth_field_history_overview.svg"),
        help="Output SVG path. Defaults to plots/depth_field_history_overview.svg.",
    )
    parser.add_argument("--title", default="Depth-Field Training Overview", help="SVG title.")
    parser.add_argument(
        "--samples-per-epoch",
        type=float,
        default=None,
        help="Override samples per epoch for x-axis placement. By default, epoch_end records are used.",
    )
    return parser.parse_args()


def nested_get(record: dict[str, Any], *keys: str) -> float | None:
    value: Any = record
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def infer_samples_per_epoch(records: list[dict[str, Any]], override: float | None) -> float | None:
    if override and override > 0:
        return float(override)
    epoch_end = [
        float(record.get("samples_seen_epoch") or 0.0)
        for record in records
        if str(record.get("trigger")) == "epoch_end" and float(record.get("samples_seen_epoch") or 0.0) > 0
    ]
    if epoch_end:
        return max(epoch_end)
    per_epoch_max: dict[int, float] = {}
    for record in records:
        epoch = int(record.get("epoch") or 0)
        samples = float(record.get("samples_seen_epoch") or 0.0)
        per_epoch_max[epoch] = max(per_epoch_max.get(epoch, 0.0), samples)
    values = [value for value in per_epoch_max.values() if value > 0]
    return max(values) if values else None


def record_x(record: dict[str, Any], samples_per_epoch: float | None) -> float:
    epoch = float(record.get("epoch") or 0.0)
    trigger = str(record.get("trigger") or "")
    samples = float(record.get("samples_seen_epoch") or 0.0)
    if trigger == "epoch_end":
        return epoch + 1.0
    if samples_per_epoch and samples_per_epoch > 0:
        return epoch + min(1.0, max(0.0, samples / samples_per_epoch))
    return epoch + 1.0


def series_from(
    records: list[dict[str, Any]],
    x_values: list[float],
    path: tuple[str, ...],
    *,
    scale: float = 1.0,
) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for record, x_value in zip(records, x_values, strict=True):
        value = nested_get(record, *path)
        if value is None:
            continue
        xs.append(x_value)
        ys.append(value * scale)
    return xs, ys


def fmt_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def fmt_percent(value: float) -> str:
    return f"{value:.1f}%"


def fmt_nm(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value / 1000.0:.2f}k nm"
    return f"{value:.0f} nm"


def line_path(xs: list[float], ys: list[float], x_scale: Callable[[float], float], y_scale: Callable[[float], float]) -> str:
    parts = []
    for index, (x_value, y_value) in enumerate(zip(xs, ys, strict=True)):
        command = "M" if index == 0 else "L"
        parts.append(f"{command} {x_scale(x_value):.2f} {y_scale(y_value):.2f}")
    return " ".join(parts)


def nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    if high <= low:
        return [low]
    return [low + (high - low) * index / float(count) for index in range(count + 1)]


def render_panel(
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    ylabel: str,
    x_max: float,
    series: list[dict[str, Any]],
    y_formatter: Callable[[float], str] = lambda value: f"{value:.3f}",
    y_min: float | None = None,
    y_max: float | None = None,
) -> list[str]:
    plot_left = x + 58
    plot_right = x + width - 20
    plot_top = y + 54
    plot_bottom = y + height - 42
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top
    values = [value for item in series for value in item["ys"]]

    if not values:
        values = [0.0, 1.0]
    raw_min = min(values) if y_min is None else float(y_min)
    raw_max = max(values) if y_max is None else float(y_max)
    if raw_max <= raw_min:
        raw_max = raw_min + 1.0
    pad = (raw_max - raw_min) * 0.08
    low = raw_min if y_min is not None else raw_min - pad
    high = raw_max if y_max is not None else raw_max + pad
    if low >= 0 and y_min is None:
        low = max(0.0, low)

    def x_scale(value: float) -> float:
        denom = max(1.0, x_max)
        return plot_left + (value / denom) * plot_width

    def y_scale(value: float) -> float:
        return plot_top + (1.0 - (value - low) / (high - low)) * plot_height

    parts = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="8" fill="#ffffff" stroke="#e5e7eb"/>',
        (
            f'<text x="{x + 18}" y="{y + 28}" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="17" font-weight="700" fill="#111827">{html.escape(title)}</text>'
        ),
        (
            f'<text x="{x + 18}" y="{y + 47}" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="11" fill="#6b7280">{html.escape(ylabel)}</text>'
        ),
    ]

    for tick in nice_ticks(low, high, 4):
        y_pos = y_scale(tick)
        parts.append(
            f'<line x1="{plot_left}" x2="{plot_right}" y1="{y_pos:.2f}" y2="{y_pos:.2f}" '
            'stroke="#f3f4f6" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{plot_left - 10}" y="{y_pos + 4:.2f}" text-anchor="end" '
            f'font-family="Segoe UI, Arial, sans-serif" font-size="10.5" fill="#4b5563">{html.escape(y_formatter(tick))}</text>'
        )

    max_tick = int(x_max) if x_max == int(x_max) else int(x_max) + 1
    step = 1 if max_tick <= 8 else max(1, (max_tick + 5) // 6)
    for epoch in range(0, max_tick + 1, step):
        x_pos = x_scale(float(epoch))
        parts.append(
            f'<line x1="{x_pos:.2f}" x2="{x_pos:.2f}" y1="{plot_top}" y2="{plot_bottom}" '
            'stroke="#f9fafb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x_pos:.2f}" y="{plot_bottom + 20}" text-anchor="middle" '
            f'font-family="Segoe UI, Arial, sans-serif" font-size="10.5" fill="#4b5563">{epoch}</text>'
        )

    parts.append(f'<line x1="{plot_left}" x2="{plot_right}" y1="{plot_bottom}" y2="{plot_bottom}" stroke="#9ca3af"/>')
    parts.append(f'<line x1="{plot_left}" x2="{plot_left}" y1="{plot_top}" y2="{plot_bottom}" stroke="#9ca3af"/>')

    legend_x = plot_left
    legend_y = y + height - 12
    for item in series:
        if len(item["xs"]) < 1:
            continue
        dash = ' stroke-dasharray="6 5"' if item.get("dash") else ""
        parts.append(
            f'<path d="{line_path(item["xs"], item["ys"], x_scale, y_scale)}" fill="none" '
            f'stroke="{item["color"]}" stroke-width="2.3"{dash}/>'
        )
        parts.append(f'<circle cx="{x_scale(item["xs"][-1]):.2f}" cy="{y_scale(item["ys"][-1]):.2f}" r="3" fill="{item["color"]}"/>')
        parts.append(f'<line x1="{legend_x}" x2="{legend_x + 22}" y1="{legend_y}" y2="{legend_y}" stroke="{item["color"]}" stroke-width="2.3"{dash}/>')
        parts.append(
            f'<text x="{legend_x + 28}" y="{legend_y + 4}" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="11" fill="#374151">{html.escape(item["label"])}</text>'
        )
        legend_x += max(110, len(str(item["label"])) * 7 + 46)

    return parts


def render_card(x: float, y: float, width: float, title: str, value: str, subtitle: str, color: str) -> list[str]:
    return [
        f'<rect x="{x}" y="{y}" width="{width}" height="92" rx="8" fill="#ffffff" stroke="#e5e7eb"/>',
        f'<rect x="{x}" y="{y}" width="5" height="92" rx="2.5" fill="{color}"/>',
        (
            f'<text x="{x + 18}" y="{y + 27}" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="12" font-weight="700" fill="#6b7280">{html.escape(title.upper())}</text>'
        ),
        (
            f'<text x="{x + 18}" y="{y + 57}" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="24" font-weight="750" fill="#111827">{html.escape(value)}</text>'
        ),
        (
            f'<text x="{x + 18}" y="{y + 78}" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="12" fill="#6b7280">{html.escape(subtitle)}</text>'
        ),
    ]


def build_overview(records: list[dict[str, Any]], title: str, source_name: str, samples_per_epoch: float | None) -> str:
    x_values = [record_x(record, samples_per_epoch) for record in records]
    x_max = max(x_values) if x_values else 1.0

    mae_x, mae_mean = series_from(records, x_values, ("val", "mae_mean"))
    _, mae_median = series_from(records, x_values, ("val", "mae_median"))
    loss_x, train_loss = series_from(records, x_values, ("train", "loss"))
    acc_x, acc = series_from(records, x_values, ("train", "acc"), scale=100.0)
    _, mat_acc = series_from(records, x_values, ("train", "mat_acc"), scale=100.0)
    _, void_acc = series_from(records, x_values, ("train", "void_acc"), scale=100.0)
    runs_x, material_runs = series_from(records, x_values, ("val", "material_runs_mean"))
    thickness_x, field_thickness = series_from(records, x_values, ("val", "field_total_thickness_nm_mean"))
    _, active_thickness = series_from(records, x_values, ("train", "mean_active_thickness_nm"))
    skip_x, skip_fraction = series_from(records, x_values, ("train", "overlimit_skip_fraction"), scale=100.0)
    _, full_depth = series_from(records, x_values, ("train", "full_depth_fraction"), scale=100.0)

    best_idx = min(range(len(mae_mean)), key=lambda index: mae_mean[index]) if mae_mean else None
    best_mae = mae_mean[best_idx] if best_idx is not None else None
    best_epoch = mae_x[best_idx] if best_idx is not None else None
    latest_mae = mae_mean[-1] if mae_mean else None
    first_mae = mae_mean[0] if mae_mean else None
    latest_loss = train_loss[-1] if train_loss else None
    latest_runs = material_runs[-1] if material_runs else None
    latest_skip = skip_fraction[-1] if skip_fraction else None

    width = 1500
    height = 1180
    margin = 42
    panel_gap = 24
    panel_w = (width - 2 * margin - panel_gap) / 2
    panel_h = 250
    card_gap = 18
    card_w = (width - 2 * margin - 4 * card_gap) / 5

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        (
            f'<text x="{margin}" y="38" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="28" font-weight="800" fill="#111827">{html.escape(title)}</text>'
        ),
        (
            f'<text x="{margin}" y="64" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="13" fill="#4b5563">Parsed from {html.escape(source_name)}; records={len(records)}, x-axis in epochs</text>'
        ),
    ]

    cards = [
        (
            "Best TMM MAE",
            "n/a" if best_mae is None else fmt_float(best_mae, 5),
            "n/a" if best_epoch is None else f"at epoch {best_epoch:.2f}",
            COLORS["blue"],
        ),
        (
            "Latest TMM MAE",
            "n/a" if latest_mae is None else fmt_float(latest_mae, 5),
            "n/a" if first_mae is None or latest_mae is None else f"{first_mae - latest_mae:+.5f} vs first",
            COLORS["green"],
        ),
        (
            "Train Loss",
            "n/a" if latest_loss is None else fmt_float(latest_loss, 5),
            "latest running metric",
            COLORS["orange"],
        ),
        (
            "Material Runs",
            "n/a" if latest_runs is None else fmt_float(latest_runs, 1),
            "latest validation mean",
            COLORS["purple"],
        ),
        (
            "Overlimit Skip",
            "n/a" if latest_skip is None else fmt_percent(latest_skip),
            "latest train fraction",
            COLORS["red"],
        ),
    ]
    card_y = 88
    for index, (card_title, value, subtitle, color) in enumerate(cards):
        parts.extend(render_card(margin + index * (card_w + card_gap), card_y, card_w, card_title, value, subtitle, color))

    top = 210
    panels = [
        {
            "title": "TMM Validation MAE",
            "ylabel": "lower is better",
            "series": [
                {"label": "mean", "xs": mae_x, "ys": mae_mean, "color": COLORS["blue"]},
                {"label": "median", "xs": mae_x, "ys": mae_median, "color": COLORS["teal"], "dash": True},
            ],
        },
        {
            "title": "Training Loss",
            "ylabel": "running train loss",
            "series": [{"label": "loss", "xs": loss_x, "ys": train_loss, "color": COLORS["orange"]}],
        },
        {
            "title": "Accuracy",
            "ylabel": "percent",
            "series": [
                {"label": "all", "xs": acc_x, "ys": acc, "color": COLORS["green"]},
                {"label": "material", "xs": acc_x, "ys": mat_acc, "color": COLORS["blue"], "dash": True},
                {"label": "void", "xs": acc_x, "ys": void_acc, "color": COLORS["gray"], "dash": True},
            ],
            "formatter": fmt_percent,
        },
        {
            "title": "Predicted Structure",
            "ylabel": "mean material runs",
            "series": [{"label": "runs", "xs": runs_x, "ys": material_runs, "color": COLORS["purple"]}],
        },
        {
            "title": "Thickness",
            "ylabel": "nm",
            "series": [
                {"label": "field val", "xs": thickness_x, "ys": field_thickness, "color": COLORS["pink"]},
                {"label": "active train", "xs": thickness_x, "ys": active_thickness, "color": COLORS["amber"], "dash": True},
            ],
            "formatter": fmt_nm,
        },
        {
            "title": "Filtering And Full-Depth",
            "ylabel": "percent",
            "series": [
                {"label": "overlimit skip", "xs": skip_x, "ys": skip_fraction, "color": COLORS["red"]},
                {"label": "full depth", "xs": skip_x, "ys": full_depth, "color": COLORS["gray"], "dash": True},
            ],
            "formatter": fmt_percent,
        },
    ]

    for index, panel in enumerate(panels):
        row = index // 2
        col = index % 2
        panel_x = margin + col * (panel_w + panel_gap)
        panel_y = top + row * (panel_h + panel_gap)
        parts.extend(
            render_panel(
                x=panel_x,
                y=panel_y,
                width=panel_w,
                height=panel_h,
                title=panel["title"],
                ylabel=panel["ylabel"],
                x_max=x_max,
                series=panel["series"],
                y_formatter=panel.get("formatter", lambda value: f"{value:.3f}"),
            )
        )

    footer_y = height - 32
    parts.append(
        f'<text x="{margin}" y="{footer_y}" font-family="Segoe UI, Arial, sans-serif" '
        'font-size="12" fill="#6b7280">Tip: pass --samples-per-epoch to override x-axis placement for partial-epoch records.</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    args = parse_arguments()
    records = json.loads(args.history.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise SystemExit(f"{args.history} must contain a non-empty JSON list")
    records = [record for record in records if isinstance(record, dict)]
    if not records:
        raise SystemExit(f"{args.history} contains no object records")

    samples_per_epoch = infer_samples_per_epoch(records, args.samples_per_epoch)
    svg = build_overview(records, args.title, args.history.name, samples_per_epoch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")

    mae_values = [nested_get(record, "val", "mae_mean") for record in records]
    mae_values = [value for value in mae_values if value is not None]
    best = min(mae_values) if mae_values else float("nan")
    latest = mae_values[-1] if mae_values else float("nan")
    print(f"records={len(records)} best_mae={best:.6f} latest_mae={latest:.6f}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
