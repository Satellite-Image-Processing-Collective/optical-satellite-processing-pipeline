import numpy as np

from src.config import OpticalSensorConfig
from src.ingestion import generate_reference_l2a_scene
from src.simulator import ForwardSensorSimulator


def test_forward_simulator_returns_4_band_12_bit_l0():
    cfg = OpticalSensorConfig()
    l2a = generate_reference_l2a_scene(height=96, width=96, seed=7)

    simulator = ForwardSensorSimulator(cfg, seed=11)
    l0_dn, telemetry = simulator.simulate_l0(l2a)

    assert l0_dn.shape == l2a.shape
    assert l0_dn.dtype == np.uint16
    assert int(l0_dn.min()) >= 0
    assert int(l0_dn.max()) <= (1 << cfg.adc_bits) - 1
    assert "gcp_ref_px" in telemetry
    assert "gcp_obs_l0_px" in telemetry
    assert telemetry["gcp_ref_px"].shape == (25, 2)


def test_reference_scene_has_expected_band_order_and_range():
    scene = generate_reference_l2a_scene(height=80, width=72, seed=3)

    assert scene.shape == (4, 80, 72)
    assert scene.dtype == np.float32
    assert np.isfinite(scene).all()
    assert scene.min() >= 0.005
    assert scene.max() <= 0.85
