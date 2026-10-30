"""Inverse optical processing: raw L0 to L1A, L1C, and L2A products."""

from typing import Dict, Tuple

import cv2
import numpy as np
from scipy import ndimage
from scipy.fft import fft2, ifft2, ifftshift

from .config import OpticalSensorConfig


class OpticalProcessingPipeline:
    """
    Operational ground-segment processor converting raw L0 detector DNs into:
      - L1A: Radiometrically calibrated, destriped, MTF-restored sensor-geometry radiance
      - L1C: Orthorectified Top-of-Atmosphere (TOA) Reflectance in map projection
      - L2A: Atmospherically corrected Bottom-of-Atmosphere (BOA) Surface Reflectance
    """

    def __init__(self, config: OpticalSensorConfig):
        self.cfg = config

    @staticmethod
    def _inpaint_defective_pixels(band_dn: np.ndarray, max_dn: int = 4095) -> np.ndarray:
        """Detects dead/hot impulse pixels via local median deviation and inpaints them."""
        band_f = band_dn.astype(np.float32)
        med = ndimage.median_filter(band_f, size=3, mode="reflect")
        mad = ndimage.median_filter(np.abs(band_f - med), size=5, mode="reflect") + 1e-3
        bad_mask = (band_dn == 0) | (band_dn == max_dn) | (np.abs(band_f - med) > 6.0 * mad)
        cleaned = band_f.copy()
        cleaned[bad_mask] = med[bad_mask]
        return cleaned

    @staticmethod
    def _destripe_prnu_columns(radiance_band: np.ndarray) -> np.ndarray:
        """
        Mitigates pushbroom vertical PRNU striping by separating low-frequency scene
        structure from high-frequency column gain anomalies in the cross-track profile.
        """
        col_means = np.median(radiance_band, axis=0, keepdims=True)
        smooth_col_trend = ndimage.gaussian_filter1d(col_means, sigma=9.0, axis=1, mode="reflect")
        stripe_gain = col_means / (smooth_col_trend + 1e-6)
        return radiance_band / np.clip(stripe_gain, 0.85, 1.15)

    @staticmethod
    def _wiener_mtf_deconvolution(
        img: np.ndarray, psf_sigma: float, nsr: float = 0.015
    ) -> np.ndarray:
        """
        Frequency-domain Wiener deconvolution to restore optical PSF blur while
        suppressing high-frequency read/shot noise amplification.
        """
        h, w = img.shape
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = h // 2, w // 2
        psf = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * psf_sigma ** 2))
        psf /= psf.sum()

        # Pad image to avoid circular FFT boundary ringing artefacts
        pad = 16
        img_pad = np.pad(img, pad, mode="reflect")
        hp, wp = img_pad.shape
        yyp, xxp = np.mgrid[0:hp, 0:wp]
        psf_pad = np.exp(-((xxp - wp // 2) ** 2 + (yyp - hp // 2) ** 2) / (2.0 * psf_sigma ** 2))
        psf_pad /= psf_pad.sum()

        H_otf = fft2(ifftshift(psf_pad))
        W_filter = np.conj(H_otf) / (np.abs(H_otf) ** 2 + nsr)
        restored_pad = np.real(ifft2(fft2(img_pad) * W_filter))
        return np.maximum(restored_pad[pad:-pad, pad:-pad], 0.0)

    def process_l0_to_l1a(self, l0_dn: np.ndarray, telemetry: Dict) -> np.ndarray:
        """
        Step 1 (L0 -> L1A): Defective pixel interpolation, dark current & bias subtraction,
        DN-to-Radiance conversion, PRNU column destriping, and Wiener MTF restoration.
        """
        num_bands, h, w = l0_dn.shape
        max_dn = (1 << self.cfg.adc_bits) - 1
        l1a_radiance = np.zeros((num_bands, h, w), dtype=np.float32)

        for b in range(num_bands):
            # 1. Bad pixel mitigation
            clean_dn = self._inpaint_defective_pixels(l0_dn[b], max_dn=max_dn)

            # 2. Radiometric calibration (DN -> Physical At-Sensor Radiance)
            dark_offset = telemetry["dark_frame_calib"][b] + self.cfg.radiometric_bias
            rad = np.maximum((clean_dn - dark_offset) / self.cfg.radiometric_gain[b], 0.0)

            # 3. Pushbroom PRNU column destriping
            rad_destriped = self._destripe_prnu_columns(rad)

            # 4. Optical MTF restoration via Wiener filter
            rad_sharp = self._wiener_mtf_deconvolution(
                rad_destriped, psf_sigma=self.cfg.psf_sigma_px[b], nsr=0.018
            )
            l1a_radiance[b] = rad_sharp

        return l1a_radiance

    def process_l1a_to_l1c(
        self, l1a_radiance: np.ndarray, telemetry: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 2 (L1A -> L1C): RANSAC geometric orthorectification using GCPs and conversion
        from At-Sensor Radiance to dimensionless Top-of-Atmosphere (TOA) Reflectance.
        """
        num_bands, h, w = l1a_radiance.shape
        gcp_obs = telemetry["gcp_obs_l0_px"]
        gcp_ref = telemetry["gcp_ref_px"]

        # Estimate inverse projective transformation via RANSAC
        H_inv_est, _ = cv2.findHomography(gcp_obs, gcp_ref, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if H_inv_est is None:
            raise RuntimeError("Could not estimate the orthorectification homography from the supplied GCPs.")

        theta_s_rad = np.deg2rad(self.cfg.solar_zenith_deg)
        cos_theta_s = np.cos(theta_s_rad)
        d2 = self.cfg.earth_sun_dist_au ** 2

        l1c_toa_reflectance = np.zeros((num_bands, h, w), dtype=np.float32)
        l1c_ortho_radiance = np.zeros((num_bands, h, w), dtype=np.float32)

        for b in range(num_bands):
            # Geometric warp onto orthorectified map grid
            rad_ortho = cv2.warpPerspective(
                l1a_radiance[b],
                H_inv_est,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT,
            )
            l1c_ortho_radiance[b] = rad_ortho

            # Convert TOA Radiance to TOA Reflectance (L1C standard product)
            esun_b = self.cfg.esun[b]
            rho_toa = (np.pi * rad_ortho * d2) / (esun_b * cos_theta_s)
            l1c_toa_reflectance[b] = np.clip(rho_toa, 0.0, 1.0)

        return l1c_toa_reflectance, H_inv_est

    def process_l1c_to_l2a(self, l1c_toa_reflectance: np.ndarray) -> np.ndarray:
        """
        Step 3 (L1C -> L2A): Atmospheric Correction via Dark Object Subtraction (DOS1)
         coupled with Rayleigh scattering transmittance compensation to recover BOA Reflectance.
        """
        num_bands, h, w = l1c_toa_reflectance.shape
        l2a_boa_reflectance = np.zeros((num_bands, h, w), dtype=np.float32)

        # Estimate empirical path reflectance from 1st-percentile dark targets (deep water/shadows)
        # while subtracting theoretical minimum 1% dark object surface reflectance
        for b in range(num_bands):
            rho_toa_b = l1c_toa_reflectance[b]
            dark_val = np.percentile(rho_toa_b, 0.5)
            rho_path_est = max(dark_val - 0.012, 0.0)

            # Estimate two-way transmittance from wavelength-dependent Rayleigh optical depth
            wl_um = self.cfg.wavelengths_nm[b] / 1000.0
            tau_rayleigh_est = np.exp(-0.008569 * (wl_um ** -4) * 2.2)
            tau_total_est = np.clip(tau_rayleigh_est * 0.94, 0.72, 0.96)

            rho_boa_b = (rho_toa_b - rho_path_est) / tau_total_est
            l2a_boa_reflectance[b] = np.clip(rho_boa_b, 0.001, 1.0)

        return l2a_boa_reflectance
