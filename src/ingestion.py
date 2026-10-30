"""Reference-scene ingestion from AWS Earth Search or a synthetic fallback."""

import json
import os
import urllib.request
import warnings
from typing import Optional

import numpy as np
import rasterio
from rasterio.warp import transform as transform_coords
from rasterio.windows import Window
from scipy import ndimage

from .config import OpticalSensorConfig
from .validation import export_geotiff


def generate_reference_l2a_scene(height: int = 256, width: int = 256, seed: int = 42) -> np.ndarray:
    """
    Generates a physically realistic 4-band (Blue, Green, Red, NIR) Surface Reflectance
    (L2A BOA, values in [0.01, 0.65]) scene containing vegetation fields, a meandering
    river, urban high-albedo pads, and bare basaltic soil.
    Shape: (4, H, W)
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]

    # Base bare soil / basaltic terrain with smooth spatial texture
    soil_texture = ndimage.gaussian_filter(rng.normal(0, 1, (height, width)), sigma=6.0)
    soil_texture = (soil_texture - soil_texture.min()) / (np.ptp(soil_texture) + 1e-8)
    # soil_texture = (soil_texture - soil_texture.min()) / (soil_texture.ptp() + 1e-8)

    # Spectral signatures [Blue, Green, Red, NIR] in surface reflectance units (0.0 - 1.0)
    sig_soil = np.array([0.09, 0.13, 0.18, 0.26])[:, None, None]
    sig_veg = np.array([0.03, 0.07, 0.04, 0.48])[:, None, None]
    sig_water = np.array([0.05, 0.04, 0.025, 0.01])[:, None, None]
    sig_urban = np.array([0.22, 0.25, 0.28, 0.31])[:, None, None]

    scene = sig_soil * (0.85 + 0.3 * soil_texture[None, :, :])

    # Agricultural vegetation parcels (high NIR, low Red)
    for r0, r1, c0, c1 in [(25, 110, 20, 120), (140, 230, 130, 235), (30, 95, 150, 225)]:
        sub_tex = 0.9 + 0.2 * np.sin(xx[r0:r1, c0:c1] * 0.25) * np.cos(yy[r0:r1, c0:c1] * 0.25)
        scene[:, r0:r1, c0:c1] = sig_veg * sub_tex[None, :, :]

    # Urban / industrial infrastructure blocks (sharp high-frequency edges for MTF testing)
    for r0, r1, c0, c1 in [(135, 195, 35, 95), (50, 80, 60, 90)]:
        scene[:, r0:r1, c0:c1] = sig_urban
        # Add high-contrast road grid inside urban block
        scene[:, r0:r1:12, c0:c1] = 0.12
        scene[:, r0:r1, c0:c1:12] = 0.12

    # Meandering river channel across the scene
    river_center = width * 0.52 + 28.0 * np.sin(yy[:, 0] * 2.0 * np.pi / height)
    river_mask = np.abs(xx - river_center[:, None]) < 11.0
    scene[:, river_mask] = sig_water.reshape(4, 1)
    # scene[:, river_mask] = sig_water

    return np.clip(scene.astype(np.float32), 0.005, 0.85)


def fetch_real_sentinel2_l2a_patch(
    cfg: OpticalSensorConfig,
    lon: float = 72.945,   # Default: JNPT Port & Mumbai Harbour (Maritime + Urban + Mangroves)
    lat: float = 18.950,   # Try Nashik Gangapur Dam: lon=73.675, lat=20.035
    patch_size: int = 512,
    cache_path: str = "outputs/real_s2_l2a_reference.tif",
) -> np.ndarray:
    """
    Queries AWS Earth Search STAC API (no login/keys needed) for a cloud-free Sentinel-2 L2A
    scene, streams a 512x512 4-band (B02, B03, B04, B08) window via Rasterio COG range requests,
    converts uint16 DNs to physical BOA Surface Reflectance, and syncs real orbital metadata.
    """
    if os.path.exists(cache_path):
        print(f"  -> Loading cached real Sentinel-2 L2A patch from {cache_path}")
        with rasterio.open(cache_path) as src:
            cfg.crs_epsg = src.crs.to_string()
            cfg.origin_utm_x = src.transform.c
            cfg.origin_utm_y = src.transform.f
            cfg.pixel_size_m = src.transform.a
            cfg.solar_zenith_deg = float(src.tags().get("SOLAR_ZENITH_DEG", 28.0))
            return src.read().astype(np.float32)

    print(f"  -> Querying AWS Earth Search STAC API at Lon={lon}, Lat={lat} (Cloud Cover < 5%)...")
    stac_url = "https://earth-search.aws.element84.com/v1/search"
    payload = {
        "collections": ["sentinel-2-l2a"],
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "query": {"eo:cloud_cover": {"lt": 5.0}},
        "limit": 5,
    }
    req = urllib.request.Request(
        stac_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        features = json.loads(resp.read().decode("utf-8"))["features"]

    if not features:
        raise RuntimeError("No cloud-free Sentinel-2 scene found for coordinates.")

    # Pick the clearest scene
    item = sorted(features, key=lambda f: f["properties"].get("eo:cloud_cover", 100.0))[0]
    props = item["properties"]
    scene_id = item["id"]
    cfg.solar_zenith_deg = float(props.get("view:sun_zenith", 28.0))
    boa_offset_applied = bool(props.get("earthsearch:boa_offset_applied", False))
    print(f"  -> Selected Scene: {scene_id} | Cloud Cover: {props.get('eo:cloud_cover'):.2f}% | Solar Zenith: {cfg.solar_zenith_deg:.2f} deg")

    asset_keys = ["blue", "green", "red", "nir"]  # Corresponds to 10m B02, B03, B04, B08
    bands_boa = []

    for idx, key in enumerate(asset_keys):
        cog_url = item["assets"][key]["href"]
        with rasterio.open(cog_url) as src:
            if idx == 0:
                # Project (lon, lat) from EPSG:4326 into the satellite tile's native UTM CRS
                utm_x, utm_y = transform_coords("EPSG:4326", src.crs, [lon], [lat])
                center_row, center_col = src.index(utm_x[0], utm_y[0])
                half = patch_size // 2
                row_off = int(np.clip(center_row - half, 0, src.height - patch_size))
                col_off = int(np.clip(center_col - half, 0, src.width - patch_size))
                win = Window(col_off, row_off, patch_size, patch_size)
                win_transform = src.window_transform(win)

                cfg.crs_epsg = src.crs.to_string()
                cfg.origin_utm_x = win_transform.c
                cfg.origin_utm_y = win_transform.f
                cfg.pixel_size_m = win_transform.a

            raw_dn = src.read(1, window=win).astype(np.float32)
            # Apply ESA Sentinel-2 L2A radiometric scaling to physical reflectance [0, 1]
            if boa_offset_applied:
                rho = raw_dn / 10000.0
            else:
                # ESA Baseline 04.00+ shifts DNs by +1000 (offset = -0.1 in reflectance units)
                offset = -0.1 if np.percentile(raw_dn, 1) > 800 else 0.0
                rho = (raw_dn * 0.0001) + offset

            bands_boa.append(np.clip(rho, 0.005, 0.95))

    gt_l2a = np.stack(bands_boa, axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    export_geotiff(cache_path, gt_l2a, cfg, f"REF_L2A_{scene_id}")
    return gt_l2a


def load_reference_l2a(
    cfg: OpticalSensorConfig,
    source: str = "auto",
    lon: float = 72.945,
    lat: float = 18.950,
    patch_size: int = 512,
    cache_path: str = "outputs/real_s2_l2a_reference.tif",
    seed: int = 42,
) -> np.ndarray:
    """Load a real Sentinel-2 L2A patch, synthetic scene, or auto-fallback.

    source='auto' tries AWS Earth Search first and falls back to a synthetic
    scene if the request or data access fails.
    """
    if source not in {"auto", "real", "synthetic"}:
        raise ValueError("source must be 'auto', 'real', or 'synthetic'")

    if source == "synthetic":
        print("  -> Using synthetic 4-band L2A reference scene")
        return generate_reference_l2a_scene(patch_size, patch_size, seed=seed)

    try:
        return fetch_real_sentinel2_l2a_patch(
            cfg, lon=lon, lat=lat, patch_size=patch_size, cache_path=cache_path
        )
    except Exception as exc:
        if source == "real":
            raise
        warnings.warn(
            f"Real Sentinel-2 ingestion failed ({exc!r}); falling back to a synthetic scene.",
            RuntimeWarning,
        )
        return generate_reference_l2a_scene(patch_size, patch_size, seed=seed)
