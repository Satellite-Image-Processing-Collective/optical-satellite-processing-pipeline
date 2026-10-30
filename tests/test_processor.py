import numpy as np

from src.config import OpticalSensorConfig
from src.ingestion import generate_reference_l2a_scene
from src.processor import OpticalProcessingPipeline
from src.simulator import ForwardSensorSimulator
from src.validation import compute_validation_metrics


def test_defective_pixel_inpainting_replaces_impulse():
    image = np.ones((9, 9), dtype=np.uint16) * 100
    image[4, 4] = 4095

    cleaned = OpticalProcessingPipeline._inpaint_defective_pixels(image, max_dn=4095)

    assert cleaned[4, 4] == 100
    assert cleaned.shape == image.shape


def test_prnu_destriping_reduces_alternating_column_gain():
    height, width = 80, 80
    gains = np.where(np.arange(width) % 2 == 0, 1.08, 0.92)
    striped = np.tile((100.0 * gains)[None, :], (height, 1)).astype(np.float32)

    corrected = OpticalProcessingPipeline._destripe_prnu_columns(striped)

    before = np.std(np.mean(striped, axis=0))
    after = np.std(np.mean(corrected, axis=0))
    assert after < before * 0.25


def test_wiener_mtf_restoration_returns_finite_image():
    rng = np.random.default_rng(5)
    image = rng.random((64, 64), dtype=np.float32)

    restored = OpticalProcessingPipeline._wiener_mtf_deconvolution(
        image, psf_sigma=1.0, nsr=0.018
    )

    assert restored.shape == image.shape
    assert np.isfinite(restored).all()
    assert restored.min() >= 0.0


def test_l0_to_l1c_processing_estimates_orthorectification():
    cfg = OpticalSensorConfig()
    gt_l2a = generate_reference_l2a_scene(height=96, width=96, seed=13)
    l0_dn, telemetry = ForwardSensorSimulator(cfg, seed=17).simulate_l0(gt_l2a)

    pipeline = OpticalProcessingPipeline(cfg)
    l1a = pipeline.process_l0_to_l1a(l0_dn, telemetry)
    l1c, h_inv = pipeline.process_l1a_to_l1c(l1a, telemetry)
    rec_l2a = pipeline.process_l1c_to_l2a(l1c)
    metrics = compute_validation_metrics(
        gt_l2a, l0_dn, l1c, rec_l2a, telemetry, h_inv, cfg
    )

    assert l1a.shape == gt_l2a.shape
    assert l1c.shape == gt_l2a.shape
    assert h_inv.shape == (3, 3)
    assert np.isfinite(l1a).all()
    assert np.isfinite(l1c).all()
    assert rec_l2a.shape == gt_l2a.shape
    assert np.isfinite(rec_l2a).all()
    assert np.isfinite(metrics["L2A_PSNR_dB"])
    assert "PRNU_Destriping_Gain_dB" in metrics

    import cv2
    observed = telemetry["gcp_obs_l0_px"]
    reference = telemetry["gcp_ref_px"]
    projected = cv2.perspectiveTransform(observed[None, :, :], h_inv)[0]
    rmse_px = np.sqrt(np.mean(np.sum((projected - reference) ** 2, axis=1)))
    assert rmse_px < 1.0
