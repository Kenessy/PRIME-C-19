#!/usr/bin/env python
from pathlib import Path

ACC_DATA = {
    "xor": {"c19": 1.0, "relu": 1.0, "silu": 1.0},
    "two_moons": {"c19": 1.0, "relu": 1.0, "silu": 1.0},
    "circles": {"c19": 0.9935, "relu": 0.9870, "silu": 0.9935},
    "spiral": {"c19": 0.9667, "relu": 0.8833, "silu": 0.9111},
}

MSE_DATA = {"c19": 0.0078, "relu": 0.0367, "silu": 0.0203}

COLORS = {"c19": "#4ade80", "relu": "#60a5fa", "silu": "#fbbf24"}


def svg_header(width, height):
    return [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='#0f172a'/>",
    ]


def svg_footer():
    return ["</svg>"]


def make_acc_svg(path: Path) -> None:
    width, height = 760, 300
    margin = {"l": 60, "r": 20, "t": 30, "b": 40}
    chart_w = width - margin["l"] - margin["r"]
    chart_h = height - margin["t"] - margin["b"]
    y_min, y_max = 0.0, 1.0
    datasets = list(ACC_DATA.keys())
    group_w = chart_w / len(datasets)
    bar_w = group_w / 4

    lines = svg_header(width, height)
    lines.append(
        "<text x='20' y='22' fill='#e2e8f0' font-size='14' "
        "font-family='Segoe UI, sans-serif'>Small Synthetic Bench: Accuracy</text>"
    )

    x0 = margin["l"]
    y0 = margin["t"] + chart_h
    lines.append(
        f"<line x1='{x0}' y1='{margin['t']}' x2='{x0}' y2='{y0}' "
        "stroke='#475569' stroke-width='1' />"
    )
    lines.append(
        f"<line x1='{x0}' y1='{y0}' x2='{x0 + chart_w}' y2='{y0}' "
        "stroke='#475569' stroke-width='1' />"
    )

    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = margin["t"] + chart_h * (1 - (t - y_min) / (y_max - y_min))
        lines.append(
            f"<line x1='{x0 - 4}' y1='{y}' x2='{x0}' y2='{y}' "
            "stroke='#64748b' stroke-width='1' />"
        )
        lines.append(
            f"<text x='{x0 - 8}' y='{y + 4}' fill='#94a3b8' font-size='10' "
            f"font-family='Segoe UI, sans-serif' text-anchor='end'>{t:.2f}</text>"
        )

    for i, ds in enumerate(datasets):
        group_x = x0 + i * group_w
        for j, act in enumerate(("c19", "relu", "silu")):
            val = ACC_DATA[ds][act]
            bar_h = chart_h * (val - y_min) / (y_max - y_min)
            x = group_x + (j + 0.5) * bar_w
            y = y0 - bar_h
            lines.append(
                f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_w * 0.8:.2f}' "
                f"height='{bar_h:.2f}' fill='{COLORS[act]}' />"
            )
        label_x = group_x + group_w / 2
        lines.append(
            f"<text x='{label_x:.2f}' y='{y0 + 18}' fill='#cbd5f5' "
            "font-size='11' font-family='Segoe UI, sans-serif' "
            f"text-anchor='middle'>{ds}</text>"
        )

    legend_x = width - 210
    legend_y = 20
    for idx, act in enumerate(("c19", "relu", "silu")):
        y = legend_y + idx * 18
        lines.append(
            f"<rect x='{legend_x}' y='{y}' width='10' height='10' fill='{COLORS[act]}' />"
        )
        lines.append(
            f"<text x='{legend_x + 16}' y='{y + 9}' fill='#e2e8f0' font-size='11' "
            f"font-family='Segoe UI, sans-serif'>{act.upper()}</text>"
        )

    lines.extend(svg_footer())
    path.write_text("\n".join(lines), encoding="utf-8")


def make_mse_svg(path: Path) -> None:
    width, height = 420, 240
    margin = {"l": 60, "r": 20, "t": 30, "b": 40}
    chart_w = width - margin["l"] - margin["r"]
    chart_h = height - margin["t"] - margin["b"]
    y_min, y_max = 0.0, 0.04
    acts = ("c19", "relu", "silu")
    bar_w = chart_w / (len(acts) * 1.6)

    lines = svg_header(width, height)
    lines.append(
        "<text x='20' y='22' fill='#e2e8f0' font-size='14' "
        "font-family='Segoe UI, sans-serif'>Sine Regression (MSE, lower is better)</text>"
    )

    x0 = margin["l"]
    y0 = margin["t"] + chart_h
    lines.append(
        f"<line x1='{x0}' y1='{margin['t']}' x2='{x0}' y2='{y0}' "
        "stroke='#475569' stroke-width='1' />"
    )
    lines.append(
        f"<line x1='{x0}' y1='{y0}' x2='{x0 + chart_w}' y2='{y0}' "
        "stroke='#475569' stroke-width='1' />"
    )

    for t in (0.0, 0.01, 0.02, 0.03, 0.04):
        y = margin["t"] + chart_h * (1 - (t - y_min) / (y_max - y_min))
        lines.append(
            f"<line x1='{x0 - 4}' y1='{y}' x2='{x0}' y2='{y}' "
            "stroke='#64748b' stroke-width='1' />"
        )
        lines.append(
            f"<text x='{x0 - 8}' y='{y + 4}' fill='#94a3b8' font-size='10' "
            f"font-family='Segoe UI, sans-serif' text-anchor='end'>{t:.02f}</text>"
        )

    for i, act in enumerate(acts):
        val = MSE_DATA[act]
        bar_h = chart_h * (val - y_min) / (y_max - y_min)
        x = x0 + i * (bar_w * 1.6) + bar_w * 0.3
        y = y0 - bar_h
        lines.append(
            f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_w:.2f}' "
            f"height='{bar_h:.2f}' fill='{COLORS[act]}' />"
        )
        lines.append(
            f"<text x='{x + bar_w / 2:.2f}' y='{y0 + 18}' fill='#cbd5f5' "
            f"font-size='11' font-family='Segoe UI, sans-serif' "
            f"text-anchor='middle'>{act.upper()}</text>"
        )
        lines.append(
            f"<text x='{x + bar_w / 2:.2f}' y='{y - 4:.2f}' fill='#e2e8f0' "
            f"font-size='10' font-family='Segoe UI, sans-serif' "
            f"text-anchor='middle'>{val:.4f}</text>"
        )

    lines.extend(svg_footer())
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "docs"
    root.mkdir(parents=True, exist_ok=True)
    make_acc_svg(root / "bench_small_prime_acc.svg")
    make_mse_svg(root / "bench_small_prime_sine.svg")


if __name__ == "__main__":
    main()
