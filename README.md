# Optical Satellite Image Preprocessing (Raw L0 data to L2A) pipeline.
**Closed-loop satellite optical sensor simulator and L0-to-L2A processing prototype.**

OptiSat demonstrates a four-band pushbroom optical payload, a physics-inspired forward degradation model, an inverse image-processing chain, geospatial export, and quantitative quality checks.

## What the project does

1. Loads a four-band Sentinel-2 L2A reference patch from the AWS Earth Search STAC API, or creates a synthetic reference scene.
2. Simulates a forward sensor chain from surface reflectance (L2A) to raw 12-bit L0 digital numbers, including optical blur, viewing-geometry warp, line jitter, shot/read noise, PRNU striping, dark current, and defective pixels.
3. Processes simulated L0 data through L1A radiometric calibration and destriping, L1C geometric correction and TOA reflectance conversion, and L2A atmospheric correction.
4. Exports L1C and L2A multi-band GeoTIFFs and saves a six-panel visual quality dashboard.
5. Calculates reflectance RMSE, PSNR, SSIM, SAM, destriping gain, and geometric RMSE.

## Project structure

```text
OptiSat-L0toL2A/
├── src/
│   ├── __init__.py
│   ├── config.py          # Sensor, orbital, radiometric, and geospatial parameters
│   ├── ingestion.py       # AWS Earth Search fetcher and synthetic reference generator
│   ├── simulator.py       # Forward L2A -> raw L0 physics/noise simulator
│   ├── processor.py       # Inverse L0 -> L1A -> L1C -> L2A pipeline
│   ├── validation.py      # Metrics, GeoTIFF export, and QA dashboard
├── main.py                # End-to-end command-line runner
├── tests/
│   ├── test_simulator.py
│   └── test_processor.py
├── outputs/               # Generated products; GeoTIFFs are ignored by Git
├── requirements.txt
├── .gitignore
└── README.md
```

## 1. Set up the project (Windows)

Open PowerShell in the extracted project folder. Check that Python is installed:

```powershell
py --version
```

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt instead and run:

```bat
.venv\Scripts\activate.bat
```

Install the libraries:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Run the pipeline

For a first run, choose the synthetic source. It does not need internet access for the satellite-data download:

```powershell
python main.py --source synthetic --patch-size 256
```

To try a real Sentinel-2 L2A patch from AWS Earth Search (internet required):

```powershell
python main.py --source real --lon 72.945 --lat 18.950 --patch-size 512
```

To try real data and automatically fall back to synthetic data if ingestion fails:

```powershell
python main.py --source auto --patch-size 512
```

The `--source` options are:

- `synthetic`: use a reproducible generated four-band reference scene.
- `real`: require a real Sentinel-2 L2A patch; errors are reported instead of falling back.
- `auto`: try real data first, then use the synthetic scene if the download or data access fails.

The runner writes these files to `outputs/`:

- `product_L1C_TOA_Reflectance.tif`
- `product_L2A_BOA_Reflectance.tif`
- `real_s2_l2a_reference.tif` (when real data are fetched and cached)
- `optisat_pipeline_dashboard.png`

## 3. Run the tests

```powershell
python -m pytest -q
```

The tests cover the 12-bit output range, defective-pixel interpolation, PRNU destriping, Wiener restoration, and GCP-based geometric registration.


## Important scientific / scope notes

This repository is a **demonstration prototype**, not a validated operational Sentinel-2 processor. The forward sensor, PRNU, MTF, attitude, and noise models are simplified simulations. Atmospheric correction uses a DOS1-style dark-object estimate with simplified wavelength-dependent transmittance; it is not a full radiative-transfer solution such as 6S. Geometric correction uses simulated GCPs and a projective transform; it does not use a DEM or a real satellite ephemeris/attitude solution. Metrics compare the reconstruction with the known input reference scene, so they are simulation validation metrics rather than independent validation against field measurements.

The real-data ingestion path reads Sentinel-2 L2A COG assets from AWS Earth Search. The forward simulator then degrades that input using the project's **simulated** sensor parameters; the result should not be described as raw Sentinel-2 L0 data.

Two validation scores should be interpreted cautiously in the current implementation: `L1C_Reflectance_RMSE` compares L1C TOA reflectance with an L2A BOA reference, so those are different product levels; and `PRNU_Destriping_Gain_dB` compares column variation between the raw L0 input and the final L2A output rather than isolating the destriping step. In one synthetic test run, that destriping score was negative (-1.14 dB), indicating the measured variation was higher after the full chain. These metrics are retained from the original script for transparency and should be refined before making quantitative performance claims.

GDAL Python bindings are optional. Rasterio performs the GeoTIFF export; when `osgeo.gdal` is installed, the code also opens the output for a basic projection metadata check.
