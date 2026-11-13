"""Reproducible cross-model plots from existing benchmark CSV files."""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

_COLORS = ("#315E9B", "#865C9A", "#C58A35", "#3D8A87", "#B65D6D", "#66718B", "#4E814A", "#A45C32", "#3A7FA1", "#A472A3", "#8C7250", "#487A98", "#9A5470", "#6C8540", "#222222", "#0072B2", "#D55E00", "#009E73", "#CC79A7")
_MARKERS = ("o", "D", "s", "^", "v", "P", "X", "<", ">", "h", "p", "*", "8", "d", "H", "o", "s", "^", "D")


def _model_label(model, rows):
    """Describe the actual saved model configuration, never just the model ID."""
    configs = {r.get("model_config", "") for r in rows if r["model_id"] == model and r.get("model_config")}
    if not configs:
        return model
    try:
        parsed = [json.loads(value) for value in configs]
        identities = {(v.get("type"), str(v.get("parameters", {}).get("model_path", "")),
                       bool(v.get("parameters", {}).get("enable_symmetrix")),
                       bool(v.get("parameters", {}).get("enable_flash"))) for v in parsed}
        if len(identities) != 1:
            return model
        config = parsed[0]
        params = config.get("parameters", {})
        filename = str(params.get("model_path", "")).replace("\\", "/").rsplit("/", 1)[-1]
        kind = config.get("type", "")
        if kind == "chgnet" and filename.startswith("chgnet_0.3.0_"):
            return "CHGNet"
        if kind == "mace" and filename == "mace-mh-1.model":
            return "MACE-MH1"
        if kind == "mace" and filename == "mace-omat-0-medium.model":
            return "MACE-OMAT0" + ("-Symmetrix" if params.get("enable_symmetrix") else "")
        if kind == "sevennet" and filename == "checkpoint_7net_nano_6.0.pth":
            return "SevenNet-Nano6" + ("-Flash" if params.get("enable_flash") else "")
        if kind == "sevennet" and filename == "checkpoint_sevennet_omni.pth":
            return "SevenNet-Omni" + ("-Flash" if params.get("enable_flash") else "")
        if kind == "mattersim" and filename == "mattersim-v1.0.0-1M.pth":
            return "MatterSim-1M"
        if kind == "mattersim" and filename == "mattersim-v1.0.0-5M.verified.pth":
            return "MatterSim-5M"
        if kind == "orb" and params.get("model_name") == "orb-v3-direct-inf-omat":
            return "ORB-v3-Direct" + ("-Compile" if params.get("compile") else "")
        if kind == "orb" and params.get("model_name") == "orb-v3-conservative-inf-omat":
            return "ORB-v3-Cons" + ("-Compile" if params.get("compile") else "")
        if kind == "grace" and filename == "GRACE-2L-SMAX-OMAT-medium":
            return "GRACE-2L-OMAT"
        if kind == "grace" and filename == "GRACE-3L-OMAT-large-ft-AM":
            return "GRACE-3L-OMAT"
        if kind == "nequip" and params.get("enable_openequivariance") and filename in {f"NequIP-OAM-{size}-0.2-oeq.nequip.pth" for size in ("S", "M", "L", "XL")}:
            return filename.removesuffix("-0.2-oeq.nequip.pth") + "-0.2-OEQ"
        if kind == "nep" and filename == "nep89_20250409.txt" and params.get("backend") == "gpu":
            if config.get("md_backend", "ase") == "gpumd_native":
                return "NEB89-GPUMD-ASE"
            return "NEP89-ASE-GPUMD"
    except (ValueError, TypeError, AttributeError):
        pass
    return model


def _hardware(rows):
    devices = set()
    for row in rows:
        try:
            env = json.loads(row.get("environment") or "{}")
            gpu = env.get("gpu")
            if gpu:
                devices.add(str(gpu).split(", GPU-")[0].strip())
            elif row.get("device"):
                devices.add(row["device"] + " (hardware not recorded)")
        except (ValueError, TypeError, AttributeError):
            devices.add("hardware not recorded")
    return "Hardware: " + (", ".join(sorted(devices)) if devices else "not recorded")


def _style_axes(ax, title, letter):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#555555")
    ax.spines[["bottom", "left"]].set_linewidth(.7)
    ax.tick_params(direction="out", length=3, width=.7, colors="#333333", labelsize=8)
    ax.grid(axis="y", which="major", color="#e5e5e5", linewidth=.6)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=10, fontweight="medium", pad=9)
    ax.text(-.16, 1.04, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


def _save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


@matplotlib.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                        "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 9})
def plot_comparison(result_dirs, cps_csv, output_dir, structures=None):
    """Plot MD speed, speed versus CPS and relaxation time from benchmark results.

    result_dirs contain speed/md.csv and speed/relax.csv. structures maps material
    labels to exact structure IDs selected for CPS; omitted selects LCO_756/NVP_720.
    Existing files in output_dir with these plot names are overwritten.
    """
    default_structures = structures is None
    structures = structures or {"LCO": "LCO_756", "NVP": "NVP_720"}
    rows = {"md": [], "relax": []}
    seen = set()
    for directory in map(Path, result_dirs):
        for stage in rows:
            path = directory / "speed" / f"{stage}.csv"
            with path.open(encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    config = json.loads(row.get("model_config") or "{}")
                    if config.get("type") == "orb" and config.get("parameters", {}).get("compile"):
                        row["model_id"] += "_compile"
                    key = (stage, row["model_id"], row["structure_id"])
                    if key in seen:
                        raise ValueError(f"Duplicate benchmark row: {key}")
                    seen.add(key)
                    row["natoms"] = int(row["natoms"])
                    rows[stage].append(row)
    if not seen:
        raise ValueError("No benchmark rows")
    scores = {}
    with Path(cps_csv).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            model = row["model_id"]
            if model in scores:
                raise ValueError(f"Duplicate CPS model: {model}")
            value = row["cps"].strip()
            score = float(value) if value else None
            if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError(f"Invalid CPS for {model}: {value}")
            scores[model] = score
    models = sorted({r["model_id"] for rs in rows.values() for r in rs})
    preferred = {"chgnet": 0, "mace_mh1": 1, "sevennet_nano": 2,
                 "mattersim_1m": 3, "orb_direct": 4, "sevennet_omni": 5,
                 "mattersim_5m": 6, "orb_conservative": 7, "mace_omat": 8,
                 "grace_2l": 9, "grace_3l": 10, "nep89": 11,
                 "orb_direct_compile": 12, "orb_conservative_compile": 13, "nep89_native": 14,
                 "nequip_oam_s_oeq": 15, "nequip_oam_m_oeq": 16,
                 "nequip_oam_l_oeq": 17, "nequip_oam_xl_oeq": 18}
    colors = {m: _COLORS[preferred.get(m, i) % len(_COLORS)] for i, m in enumerate(models)}
    markers = {m: _MARKERS[preferred.get(m, i) % len(_MARKERS)] for i, m in enumerate(models)}
    linestyles = {m: ("-", "--", "-.", ":")[preferred.get(m, i) % 4] for i, m in enumerate(models)}
    labels = {m: _model_label(m, rows["md"] + rows["relax"]) for m in models}
    hardware = _hardware(rows["md"] + rows["relax"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for stage, metric, ylabel, name in [("md", "steps_per_second", "MD speed (steps/s)", "md_speed"),
                                        ("relax", "elapsed_seconds", "Relaxation time (s)", "relax_time")]:
        fig, axes = plt.subplots(1, len(structures), figsize=(3.55 * len(structures), 3.5), squeeze=False)
        fig.subplots_adjust(left=.10, right=.98, bottom=.40, top=.84, wspace=.3)
        fig.text(.10, .96, hardware, fontsize=8, color="#555555", va="top")
        has_unconverged = False
        for index, (ax, material) in enumerate(zip(axes[0], structures)):
            for model in models:
                points = [r for r in rows[stage] if r["model_id"] == model and r["structure_id"].startswith(material + "_") and r["status"] == "completed" and r.get(metric)]
                points.sort(key=lambda r: r["natoms"])
                if stage == "relax":
                    valid = [r for r in points if r.get("converged", "").lower() == "true"]
                    excluded = [r for r in points if r.get("converged", "").lower() != "true"]
                else:
                    valid, excluded = points, []
                if valid:
                    ax.plot([r["natoms"] for r in valid], [float(r[metric]) for r in valid], marker=markers[model], markersize=4.5, markerfacecolor="white", markeredgewidth=.9, linewidth=.9, linestyle=linestyles[model], color=colors[model], label=labels[model])
                if excluded:
                    has_unconverged = True
                    ax.scatter([r["natoms"] for r in excluded], [float(r[metric]) for r in excluded], color=colors[model], marker="x", label=f"{model} (not converged)")
            _style_axes(ax, material, chr(97 + index))
            ax.set(xlabel="Number of atoms", ylabel=ylabel, yscale="log")
            ax.xaxis.label.set_size(9)
            ax.yaxis.label.set_size(9)
        handles = [Line2D([], [], color=colors[m], marker=markers[m], markerfacecolor="white", markeredgewidth=.9, linewidth=.9, linestyle=linestyles[m], markersize=4.5, label=labels[m]) for m in models]
        if has_unconverged:
            handles.append(Line2D([], [], color="#555555", marker="x", linestyle="none", label="Not converged"))
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), frameon=False, fontsize=6.3, bbox_to_anchor=(.5, .015))
        _save(fig, output, name)

    panels = list(structures.items())
    if default_structures:
        panels += [("LCO", "LCO_378"), ("NVP", "NVP_480")]
    ncols = len(structures)
    nrows = 2 if default_structures else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.55 * ncols, 5.8 if nrows == 2 else 3.5), squeeze=False)
    fig.subplots_adjust(left=.15, right=.98, bottom=.25 if nrows == 2 else .39, top=.90 if nrows == 2 else .82, wspace=.48, hspace=.65)
    fig.text(.10, .97, hardware, fontsize=8, color="#555555", va="top")
    missing = []
    plotted = set()
    for index, (ax, (material, structure)) in enumerate(zip(axes.flat, panels)):
        for model in models:
            match = [r for r in rows["md"] if r["model_id"] == model and r["structure_id"] == structure and r["status"] == "completed" and r.get("steps_per_second")]
            score = scores.get(model, scores.get(model.removesuffix("_compile")))
            if not match:
                missing.append((structure, model, "missing speed"))
                continue
            speed = float(match[0]["steps_per_second"])
            if score is None:
                missing.append((structure, model, "missing CPS"))
            ax.scatter(speed, score if score is not None else 1.15, facecolors="white", edgecolors=colors[model], linewidths=1.4, marker=markers[model], label=labels[model], s=46, zorder=3)
            plotted.add(model)
        _style_axes(ax, structure.replace("_", " · ") + " atoms", chr(97 + index))
        ax.axhline(1.05, color="#bbbbbb", linewidth=.6, linestyle="--")
        ax.set(xlabel="MD speed (steps/s)", ylabel="CPS", xscale="log", ylim=(0, 1.25))
        ax.set_yticks([0, .2, .4, .6, .8, 1, 1.15], ["0", "0.2", "0.4", "0.6", "0.8", "1.0", "Unknown"])
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)
    if plotted:
        handles = [Line2D([], [], color=colors[m], marker=markers[m], markerfacecolor="white", markeredgewidth=1.4, linestyle="none", markersize=6, label=labels[m]) for m in models if m in plotted]
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), frameon=False, fontsize=6.3, bbox_to_anchor=(.5, .015))
    _save(fig, output, "speed_vs_cps")
    for suffix in ("png", "pdf", "svg", "alignment.json"):
        (output / f"speed_vs_cps_around_400.{suffix}").unlink(missing_ok=True)
    with (output / "missing_cps_or_speed.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["structure_id", "model_id", "reason"])
        writer.writerows(missing)
    return {"output_dir": str(output), "missing": missing}
