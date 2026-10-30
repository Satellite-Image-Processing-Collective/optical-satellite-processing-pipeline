"""Sensor, orbital, radiometric, and geospatial configuration."""

from dataclasses import dataclass
from typing import Tuple


class OpticalSensorConfig:
    """Physics and electronics parameters for a 4-band pushbroom optical payload."""
    bands: Tuple[str, ...] = ("B02_Blue", "B03_Green", "B04_Red", "B08_NIR")
    wavelengths_nm: Tuple[float, ...] = (490.0, 560.0, 665.0, 842.0)
    # Exo-atmospheric solar irradiance E_sun (W / m^2 / um)
    esun: Tuple[float, ...] = (1941.0, 1822.0, 1517.0, 1036.0)
    solar_zenith_deg: float = 28.0
    earth_sun_dist_au: float = 1.002
    # Two-way atmospheric transmittance (Rayleigh + aerosol) per band
    atm_transmittance: Tuple[float, ...] = (0.78, 0.83, 0.88, 0.93)
    # Atmospheric path radiance L_path (W / m^2 / sr / um) - stronger in Blue (Rayleigh)
    path_radiance: Tuple[float, ...] = (32.0, 21.0, 12.0, 4.5)
    # Optics & Detector electronics (12-bit ADC: 0 - 4095 DN)
    adc_bits: int = 12
    radiometric_gain: Tuple[float, ...] = (8.5, 9.0, 10.5, 13.0)#(11.5, 12.0, 13.5, 16.0)  # DN per unit radiance
    radiometric_bias: float = 120.0  # Dark offset pedestal in DN
    dark_current_dn: float = 18.0
    read_noise_std_dn: float = 3.5
    prnu_stripe_std: float = 0.035  # 3.5% column-to-column gain non-uniformity
    dead_pixel_ratio: float = 0.002  # 0.2% defective/hot pixels
    psf_sigma_px: Tuple[float, ...] = (1.05, 1.00, 0.95, 1.15)  # Optical PSF blur per band
    # Geospatial grid parameters (Nashik / Deccan Trap test site coordinates in UTM 43N)
    crs_epsg: str = "EPSG:32643"
    pixel_size_m: float = 10.0
    origin_utm_x: float = 370000.0
    origin_utm_y: float = 2212000.0
