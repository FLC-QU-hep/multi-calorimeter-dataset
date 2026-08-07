"""Generate 1 zero-shot interpolation config: sf=0.035, num_layers=35.

This combination never appears exactly in the training set:
  - Training samples sf and num_layers continuously.
  - (nl=35, sf=0.035) is a true zero-shot interpolation point.
"""
import os
import json
import pickle
import numpy as np
from pathlib import Path
from string import Template

REPO_ROOT = Path(os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[1])))
BASE = REPO_ROOT / "calo_configs"

CONFIG = {
    "model_path": BASE / "par04/SimpleBox/sf_nlayers/SiW_iter1/model.pkl",
    "output_dir": BASE / "par04/SimpleBox/sf_nlayers_angles/SiW_zeroshot",
    "template_path": BASE / "base/SimpleBox_template.xml",
    "x0_si": 93.6,
    "x0_w": 3.5,
    "n_x0_total": 30,
}

# Zero-shot interpolation point
TARGET_SF = 0.035
TARGET_NL = 35


def compute_thicknesses(model, sf, nl, cfg):
    a = model.predict([[sf, nl]])[0]
    t_active = (cfg["n_x0_total"] * cfg["x0_si"] * a) / nl
    t_passive = (cfg["n_x0_total"] * cfg["x0_w"] * (1 - a)) / nl
    return a, t_active, t_passive


def main():
    cfg = CONFIG
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)

    template = Template(cfg["template_path"].read_text())
    model = pickle.load(open(cfg["model_path"], "rb"))

    a, t_active, t_passive = compute_thicknesses(model, TARGET_SF, TARGET_NL, cfg)

    metadata = [{
        "id": 0,
        "branch": "zeroshot",
        "target_sf": float(TARGET_SF),
        "num_layers": int(TARGET_NL),
        "predicted_a": float(a),
        "active_mm": float(t_active),
        "passive_mm": float(t_passive),
    }]

    xml = template.substitute(
        num_layers=TARGET_NL,
        sampling_fraction=f"{TARGET_SF:.10f}",
        active_thickness=f"{t_active:.10f}",
        passive_thickness=f"{t_passive:.10f}",
        elements_path="../../../base/elements.xml",
        materials_path="../../../base/materials.xml",
    )
    (cfg["output_dir"] / "SimpleBox_config_000.xml").write_text(xml)

    with open(cfg["output_dir"] / "zeroshot_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated zero-shot config:")
    print(f"  SF={TARGET_SF:.3f}, layers={TARGET_NL}")
    print(f"  predicted_a={a:.6f}")
    print(f"  active_thickness={t_active:.4f} mm")
    print(f"  passive_thickness={t_passive:.4f} mm")
    print(f"  -> {cfg['output_dir']}")


if __name__ == "__main__":
    main()
