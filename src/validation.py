"""Quality metrics, GeoTIFF export, RGB display, and visual QA dashboard."""

import os
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage

from .config import OpticalSensorConfig

try:
    from osgeo import gdal
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False


def export_geotiff(
    filepath: str, data: np.ndarray, config: OpticalSensorConfig, product_level: str
) -> None:
    """Writes a multi-band georeferenced GeoTIFF with CRS and band metadata via Rasterio & GDAL."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    num_bands, h, w = data.shape
    transform = from_origin(
        config.origin_utm_x, config.origin_utm_y, config.pixel_size_m, config.pixel_size_m
    )

    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=num_bands,
        dtype=data.dtype,
        crs=config.crs_epsg,
        transform=transform,
    ) as dst:
        dst.update_tags(
            PRODUCT_LEVEL=product_level,
            SOLAR_ZENITH_DEG=str(config.solar_zenith_deg),
            PROCESSOR="OptiSat-L0toL2A-Pipeline",
        )
        for idx, band_name in enumerate(config.bands, start=1):
            dst.write(data[idx - 1], idx)
            dst.set_band_description(idx, band_name)

    if HAS_GDAL:
        ds = gdal.Open(filepath, gdal.GA_ReadOnly)
        if ds is not None:
            _ = ds.GetProjection()
            ds = None


def compute_validation_metrics(
    gt_l2a: np.ndarray,
    l0_dn: np.ndarray,
    l1c_toa: np.ndarray,
    rec_l2a: np.ndarray,
    telemetry: Dict,
    H_inv_est: np.ndarray,
    config: OpticalSensorConfig,
) -> Dict[str, float]:
    """
    Computes radiometric, spectral, noise, and geometric accuracy metrics on the valid interior ROI.
    """
    margin = 14  # Exclude border reflection padding from warp metrics
    gt = gt_l2a[:, margin:-margin, margin:-margin]
    l1c = l1c_toa[:, margin:-margin, margin:-margin]
    l2a = rec_l2a[:, margin:-margin, margin:-margin]
    l0_crop = l0_dn[:, margin:-margin, margin:-margin].astype(np.float32)

    # 1. Radiometric RMSE & PSNR (L2A Recovered vs Ground Truth)
    rmse_l1c = float(np.sqrt(np.mean((l1c - gt) ** 2)))
    rmse_l2a = float(np.sqrt(np.mean((l2a - gt) ** 2)))
    psnr_l2a = float(20.0 * np.log10(1.0 / (rmse_l2a + 1e-8)))

    # 2. Structural Similarity Index (SSIM - analytical windowed mean/variance formulation)
    c1, c2 = (0.01 * 1.0) ** 2, (0.03 * 1.0) ** 2
    ssim_vals = []
    for b in range(gt.shape[0]):
        mu_x = ndimage.uniform_filter(gt[b], size=7)
        mu_y = ndimage.uniform_filter(l2a[b], size=7)
        sig_x2 = ndimage.uniform_filter(gt[b] ** 2, size=7) - mu_x ** 2
        sig_y2 = ndimage.uniform_filter(l2a[b] ** 2, size=7) - mu_y ** 2
        sig_xy = ndimage.uniform_filter(gt[b] * l2a[b], size=7) - mu_x * mu_y
        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)) / (
            (mu_x ** 2 + mu_y ** 2 + c1) * (sig_x2 + sig_y2 + c2)
        )
        ssim_vals.append(np.mean(ssim_map))
    mean_ssim = float(np.mean(ssim_vals))

    # 3. Spectral Angle Mapper (SAM in degrees) - evaluates multi-band spectral fidelity
    dot_prod = np.sum(gt * l2a, axis=0)
    norm_gt = np.linalg.norm(gt, axis=0) + 1e-8
    norm_l2a = np.linalg.norm(l2a, axis=0) + 1e-8
    cos_sam = np.clip(dot_prod / (norm_gt * norm_l2a), -1.0, 1.0)
    sam_deg = float(np.mean(np.rad2deg(np.arccos(cos_sam))))

    # 4. Cross-Track Destriping Improvement (High-frequency column variation reduction)
    l0_norm = l0_crop[2] / (np.mean(l0_crop[2]) + 1e-6)
    l2a_norm = l2a[2] / (np.mean(l2a[2]) + 1e-6)
    col_diff_l0 = np.std(np.diff(np.mean(l0_norm, axis=0)))
    col_diff_l2a = np.std(np.diff(np.mean(l2a_norm, axis=0)))
    destripe_db = float(20.0 * np.log10((col_diff_l0 + 1e-8) / (col_diff_l2a + 1e-8)))

    # 5. Sub-Pixel Geometric Registration RMSE (in pixels and meters)
    gcp_obs = telemetry["gcp_obs_l0_px"]
    gcp_ref = telemetry["gcp_ref_px"]
    gcp_reproj = cv2.perspectiveTransform(gcp_obs[None, :, :], H_inv_est)[0]
    geom_rmse_px = float(np.sqrt(np.mean(np.sum((gcp_reproj - gcp_ref) ** 2, axis=1))))
    geom_rmse_m = geom_rmse_px * config.pixel_size_m

    return {
        "L1C_Reflectance_RMSE": rmse_l1c,
        "L2A_Reflectance_RMSE": rmse_l2a,
        "L2A_PSNR_dB": psnr_l2a,
        "L2A_Mean_SSIM": mean_ssim,
        "Spectral_Angle_Mapper_Deg": sam_deg,
        "PRNU_Destriping_Gain_dB": destripe_db,
        "Geometric_RMSE_px": geom_rmse_px,
        "Geometric_RMSE_m": geom_rmse_m,
    }


def make_rgb(cube: np.ndarray, stretch_max: float = 0.32) -> np.ndarray:
    """Extracts [Red=B04, Green=B03, Blue=B02] into a gamma-stretched RGB display array."""
    rgb = np.stack([cube[2], cube[1], cube[0]], axis=-1)
    rgb_norm = np.clip(rgb / stretch_max, 0.0, 1.0)
    return np.power(rgb_norm, 1.0 / 1.4)


def render_dashboard(
    gt_l2a: np.ndarray,
    l0_dn: np.ndarray,
    l1a_rad: np.ndarray,
    l1c_toa: np.ndarray,
    rec_l2a: np.ndarray,
    metrics: Dict[str, float],
    output_path: str = "outputs/optisat_pipeline_dashboard.png",
) -> None:
    """Render and save the six-panel visual QA dashboard."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), dpi=140)

    axes[0, 0].imshow(make_rgb(gt_l2a))
    axes[0, 0].set_title("1. Ground Truth L2A (BOA RGB)", fontweight="bold")

    l0_rgb = np.stack([l0_dn[2], l0_dn[1], l0_dn[0]], axis=-1).astype(np.float32) / 4095.0
    axes[0, 1].imshow(np.clip(l0_rgb * 2.2, 0, 1))
    axes[0, 1].set_title("2. Raw L0 (Warped, Striped, Hazy, Bad Px)", fontweight="bold")

    axes[0, 2].imshow(make_rgb(l1a_rad / 160.0, stretch_max=0.45))
    axes[0, 2].set_title("3. L1A (Destriped + Wiener MTF Restored)", fontweight="bold")

    axes[1, 0].imshow(make_rgb(l1c_toa))
    axes[1, 0].set_title("4. L1C Orthorectified TOA (Atmospheric Haze)", fontweight="bold")

    axes[1, 1].imshow(make_rgb(rec_l2a))
    axes[1, 1].set_title(
        f"5. Recovered L2A BOA (PSNR: {metrics['L2A_PSNR_dB']:.1f} dB, "
        f"SSIM: {metrics['L2A_Mean_SSIM']:.3f})",
        fontweight="bold",
    )

    err_map = np.mean(np.abs(rec_l2a - gt_l2a), axis=0)
    im = axes[1, 2].imshow(err_map, cmap="magma", vmin=0.0, vmax=0.06)
    axes[1, 2].set_title("6. Absolute L2A Reflectance Error Map", fontweight="bold")
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04, label="|Δ Reflectance|")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
