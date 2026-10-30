"""Forward optical sensor simulator: clean L2A reflectance to noisy raw L0 DNs."""

from typing import Dict, Tuple

import cv2
import numpy as np
from scipy import ndimage

from .config import OpticalSensorConfig


class ForwardSensorSimulator:
    """
    Physics-based forward simulator converting clean BOA Surface Reflectance (L2A)
    into raw uncalibrated, geometrically warped, noisy 12-bit L0 Digital Numbers.
    """

    def __init__(self, config: OpticalSensorConfig, seed: int = 101):
        self.cfg = config
        self.rng = np.random.default_rng(seed)

    def simulate_l0(self, rho_boa: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Runs the full forward degradation chain and returns (raw_l0_dn, telemetry_meta)."""
        num_bands, h, w = rho_boa.shape
        theta_s_rad = np.deg2rad(self.cfg.solar_zenith_deg)
        cos_theta_s = np.cos(theta_s_rad)
        d2 = self.cfg.earth_sun_dist_au ** 2

        # Define off-nadir platform viewing homography (simulating roll/pitch/yaw + perspective)
        src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dst_pts = np.float32([
            [6.5, 4.0],
            [w - 9.0, 7.5],
            [w - 4.5, h - 6.0],
            [8.0, h - 3.5],
        ])
        H_forward = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Simulate pushbroom high-frequency attitude micro-jitter along-track (per scanline)
        scanline_jitter_px = 0.45 * np.sin(np.linspace(0, 10 * np.pi, h)) + self.rng.normal(0, 0.15, h)

        # Generate persistent detector column PRNU (Photo-Response Non-Uniformity) & Dark frame
        prnu_gains = self.rng.normal(0.0, self.cfg.prnu_stripe_std, size=(num_bands, 1, w)).astype(np.float32)
        dark_frame = (
            self.cfg.dark_current_dn
            + self.rng.normal(0.0, 1.2, size=(num_bands, h, w)).astype(np.float32)
        )

        # Generate defective (dead = 0, saturated hot = 4095) pixel mask
        defect_probs = self.rng.random((num_bands, h, w))
        hot_mask = defect_probs < (self.cfg.dead_pixel_ratio / 2.0)
        dead_mask = (defect_probs >= (self.cfg.dead_pixel_ratio / 2.0)) & (
            defect_probs < self.cfg.dead_pixel_ratio
        )

        l0_dn = np.zeros((num_bands, h, w), dtype=np.uint16)
        toa_radiance_true = np.zeros_like(rho_boa)

        for b in range(num_bands):
            # Step 1: Radiative transfer (BOA Reflectance -> At-Sensor TOA Radiance)
            esun_b = self.cfg.esun[b]
            tau_b = self.cfg.atm_transmittance[b]
            l_path_b = self.cfg.path_radiance[b]
            l_toa = (rho_boa[b] * esun_b * cos_theta_s * tau_b) / (np.pi * d2) + l_path_b
            toa_radiance_true[b] = l_toa

            # Step 2: Optical PSF Blur (Diffraction + detector aperture MTF degradation)
            l_opt = ndimage.gaussian_filter(l_toa, sigma=self.cfg.psf_sigma_px[b], mode="reflect")

            # Step 3: Off-nadir projective viewing geometry warp
            l_geom = cv2.warpPerspective(
                l_opt, H_forward, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            )

            # Apply line-by-line pushbroom attitude jitter
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            xx_jittered = xx + scanline_jitter_px[:, None].astype(np.float32)
            l_geom = cv2.remap(
                l_geom, xx_jittered, yy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            )

            # Step 4: Photon Shot Noise (Poisson statistics scaled by photoelectron conversion)
            photo_scale = 25.0  # electrons per unit radiance
            l_shot = self.rng.poisson(np.maximum(l_geom * photo_scale, 0.0)) / photo_scale

            # Step 5: Detector PRNU striping, Dark Current, Read Noise, and 12-bit ADC Quantization
            gain_b = self.cfg.radiometric_gain[b]
            read_noise = self.rng.normal(0.0, self.cfg.read_noise_std_dn, size=(h, w))
            dn_float = (
                gain_b * l_shot * (1.0 + prnu_gains[b])
                + dark_frame[b]
                + self.cfg.radiometric_bias
                + read_noise
            )

            # Inject defective hot/dead detector elements
            max_dn = (1 << self.cfg.adc_bits) - 1
            dn_quant = np.clip(np.floor(dn_float), 0, max_dn).astype(np.uint16)
            dn_quant[hot_mask[b]] = max_dn
            dn_quant[dead_mask[b]] = 0
            l0_dn[b] = dn_quant

        # Simulate noisy star-tracker / GPS Ground Control Points (GCPs) for orthorectification
        gcp_grid_y, gcp_grid_x = np.meshgrid(np.linspace(25, h - 25, 5), np.linspace(25, w - 25, 5))
        gcp_ref = np.column_stack([gcp_grid_x.ravel(), gcp_grid_y.ravel()]).astype(np.float32)
        gcp_warped = cv2.perspectiveTransform(gcp_ref[None, :, :], H_forward)[0]
        # Add realistic sub-pixel attitude measurement noise to GCPs
        gcp_obs = gcp_warped + self.rng.normal(0, 0.18, gcp_warped.shape).astype(np.float32)

        telemetry = {
            "H_forward_true": H_forward,
            "gcp_ref_px": gcp_ref,
            "gcp_obs_l0_px": gcp_obs,
            "dark_frame_calib": np.mean(dark_frame, axis=(1, 2)),  # Onboard calibration shutter mean
            "toa_radiance_true": toa_radiance_true,
        }
        return l0_dn, telemetry
