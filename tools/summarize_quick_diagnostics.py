"""Summarize quick diagnostic experiment outputs."""

import argparse
import json
import shutil
from pathlib import Path


def read_jsonl(path):
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def latest_sample_dir(experiment_dir):
    sample_root = experiment_dir / "samples"
    if not sample_root.is_dir():
        return None
    candidates = sorted(path for path in sample_root.glob("epoch_*") if path.is_dir())
    return candidates[-1] if candidates else None


def best_metric(rows, key, mode="min"):
    values = [(row.get("epoch"), row.get(key)) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return min(values, key=lambda item: item[1]) if mode == "min" else max(values, key=lambda item: item[1])


def tail_text(path, lines=40):
    if not path.is_file():
        return []
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return content[-lines:]


def summarize_experiment(name, experiment_dir, output_root, max_panels):
    metrics = read_jsonl(experiment_dir / "metrics.jsonl")
    val_metrics = read_jsonl(experiment_dir / "validation_metrics.jsonl")
    sample_dir = latest_sample_dir(experiment_dir)
    copied_panels = []
    if sample_dir is not None:
        panel_out = output_root / "latest_panels" / name
        panel_out.mkdir(parents=True, exist_ok=True)
        for panel in sorted(sample_dir.glob("*_panel.png"))[:max_panels]:
            destination = panel_out / panel.name
            shutil.copy2(panel, destination)
            copied_panels.append(str(destination))
    log_path = output_root / "logs" / f"{name}.log"
    return {
        "name": name,
        "dir": str(experiment_dir),
        "latest_metrics": metrics[-1] if metrics else None,
        "best_val_diffusion": best_metric(metrics, "val_diffusion"),
        "latest_validation_metrics": val_metrics[-1] if val_metrics else None,
        "best_gen_vs_real_l1": best_metric(val_metrics, "gen_vs_real_l1"),
        "latest_sample_dir": str(sample_dir) if sample_dir else None,
        "copied_panels": copied_panels,
        "log_tail": tail_text(log_path),
    }


def write_markdown(summary, path):
    lines = ["# Quick Diagnostic Summary", ""]
    for experiment in summary["experiments"]:
        lines.append(f"## {experiment['name']}")
        lines.append("")
        lines.append(f"- dir: `{experiment['dir']}`")
        lines.append(f"- latest_sample_dir: `{experiment['latest_sample_dir']}`")
        if experiment["latest_metrics"]:
            lines.append("- latest metrics:")
            lines.append("```json")
            lines.append(json.dumps(experiment["latest_metrics"], ensure_ascii=False, indent=2))
            lines.append("```")
        if experiment["best_val_diffusion"]:
            epoch, value = experiment["best_val_diffusion"]
            lines.append(f"- best val_diffusion: epoch {epoch}, {value}")
        if experiment["latest_validation_metrics"]:
            lines.append("- latest validation metrics:")
            lines.append("```json")
            lines.append(json.dumps(experiment["latest_validation_metrics"], ensure_ascii=False, indent=2))
            lines.append("```")
        if experiment["best_gen_vs_real_l1"]:
            epoch, value = experiment["best_gen_vs_real_l1"]
            lines.append(f"- best gen_vs_real_l1: epoch {epoch}, {value}")
        if experiment["copied_panels"]:
            lines.append("- copied latest panels:")
            for panel in experiment["copied_panels"]:
                lines.append(f"  - `{panel}`")
        if experiment["log_tail"]:
            lines.append("- log tail:")
            lines.append("```text")
            lines.extend(experiment["log_tail"])
            lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-panels", type=int, default=4)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    experiments = []
    for path in sorted(output_root.iterdir() if output_root.is_dir() else []):
        if path.is_dir() and (path / "metrics.jsonl").is_file():
            experiments.append(summarize_experiment(path.name, path, output_root, args.max_panels))
    summary = {
        "output_root": str(output_root),
        "experiments": experiments,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, output_root / "SUMMARY.md")
    print(json.dumps({
        "summary": str(output_root / "SUMMARY.md"),
        "json": str(output_root / "summary.json"),
        "experiments": [item["name"] for item in experiments],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
