"""End-to-end OptiSat command-line pipeline."""

import argparse
import os
import warnings

import rasterio

from src.config import OpticalSensorConfig
from src.ingestion import load_reference_l2a
from src.processor import OpticalProcessingPipeline
from src.simulator import ForwardSensorSimulator
from src.validation import compute_validation_metrics, export_geotiff, render_dashboard

# Suppress only Rasterio's harmless warning for in-memory/non-georeferenced arrays.
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate a satellite optical sensor and process L0 data to L2A."
    )
    parser.add_argument(
        "--source", choices=("auto", "real", "synthetic"), default="auto",
        help="Reference L2A input: try AWS then fallback, require real data, or use synthetic data.",
    )
    parser.add_argument("--lon", type=float, default=72.945, help="Longitude for Sentinel-2 lookup.")
    parser.add_argument("--lat", type=float, default=18.950, help="Latitude for Sentinel-2 lookup.")
    parser.add_argument("--patch-size", type=int, default=512, help="Square input patch size in pixels.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for output products.")
    parser.add_argument("--scene-seed", type=int, default=42, help="Seed for synthetic reference scene.")
    parser.add_argument("--simulator-seed", type=int, default=101, help="Seed for sensor noise simulation.")
    return parser


def run_pipeline(args: argparse.Namespace) -> dict:
    if args.patch_size < 64:
        raise ValueError("--patch-size must be at least 64 pixels for the GCP grid and validation crop.")

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = OpticalSensorConfig()
    cache_path = os.path.join(args.output_dir, "real_s2_l2a_reference.tif")

    print("[1/5] Loading 4-band reference L2A surface reflectance...")
    gt_l2a = load_reference_l2a(
        cfg, source=args.source, lon=args.lon, lat=args.lat,
        patch_size=args.patch_size, cache_path=cache_path, seed=args.scene_seed,
    )

    print("[2/5] Running physics-based forward sensor simulator (L2A -> raw 12-bit L0)...")
    simulator = ForwardSensorSimulator(cfg, seed=args.simulator_seed)
    l0_dn, telemetry = simulator.simulate_l0(gt_l2a)

    print("[3/5] Running inverse processing pipeline (L0 -> L1A -> L1C -> L2A)...")
    pipeline = OpticalProcessingPipeline(cfg)
    l1a_rad = pipeline.process_l0_to_l1a(l0_dn, telemetry)
    l1c_toa, H_inv_est = pipeline.process_l1a_to_l1c(l1a_rad, telemetry)
    rec_l2a = pipeline.process_l1c_to_l2a(l1c_toa)

    print("[4/5] Exporting georeferenced L1C and L2A GeoTIFF products...")
    l1c_path = os.path.join(args.output_dir, "product_L1C_TOA_Reflectance.tif")
    l2a_path = os.path.join(args.output_dir, "product_L2A_BOA_Reflectance.tif")
    export_geotiff(l1c_path, l1c_toa, cfg, "L1C_TOA")
    export_geotiff(l2a_path, rec_l2a, cfg, "L2A_BOA")

    print("[5/5] Computing validation metrics and rendering visual QA dashboard...\n")
    metrics = compute_validation_metrics(
        gt_l2a, l0_dn, l1c_toa, rec_l2a, telemetry, H_inv_est, cfg
    )

    print("=" * 68)
    print("  OPTISAT L0-TO-L2A VALIDATION & QUALITY ASSURANCE REPORT")
    print("=" * 68)
    for label, key, fmt in (
        ("L1C TOA Reflectance RMSE", "L1C_Reflectance_RMSE", ".4f"),
        ("L2A BOA Reflectance RMSE", "L2A_Reflectance_RMSE", ".4f"),
        ("L2A Reconstruction PSNR", "L2A_PSNR_dB", ".2f"),
        ("L2A Mean Structural Similarity (SSIM)", "L2A_Mean_SSIM", ".4f"),
        ("Spectral Angle Mapper (SAM Error)", "Spectral_Angle_Mapper_Deg", ".2f"),
        ("PRNU Destriping Gain (dB)", "PRNU_Destriping_Gain_dB", ".2f"),
        ("Geometric RMSE (pixels)", "Geometric_RMSE_px", ".3f"),
        ("Geometric RMSE (meters)", "Geometric_RMSE_m", ".2f"),
    ):
        print(f"  {label:<42}: {metrics[key]:{fmt}}")
    print("=" * 68)

    dashboard_path = os.path.join(args.output_dir, "optisat_pipeline_dashboard.png")
    render_dashboard(gt_l2a, l0_dn, l1a_rad, l1c_toa, rec_l2a, metrics, dashboard_path)
    print(f"\nSaved dashboard: {dashboard_path}")
    print(f"Saved GeoTIFFs: {l1c_path} and {l2a_path}")
    return metrics


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
