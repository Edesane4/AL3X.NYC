import math
import pytest
from al3x.forecaster import _compute_forecast_bands, _Z_80


def test_bands_with_gefs_only():
    """When GEFS provides sigma but no model disagreement, sigma_total = gefs_sigma."""
    bands = _compute_forecast_bands(
        final_f=55.0,
        gefs_sigma=2.0,
        contributing_sources=[("hrrr", 55.0, 0.5)],
    )
    assert bands["sigma_f"] == pytest.approx(2.0, abs=0.01)
    assert bands["p50"] == 55.0
    assert bands["p10"] == pytest.approx(55.0 - _Z_80 * 2.0, abs=0.05)
    assert bands["p90"] == pytest.approx(55.0 + _Z_80 * 2.0, abs=0.05)


def test_bands_with_model_disagreement_only():
    """When GEFS sigma is None but models disagree, sigma_total reflects disagreement."""
    bands = _compute_forecast_bands(
        final_f=55.0,
        gefs_sigma=None,
        contributing_sources=[
            ("hrrr", 53.0, 0.4),
            ("nws_point", 57.0, 0.6),
        ],
    )
    # Weighted mean = 0.4*53 + 0.6*57 = 55.4
    # Weighted var  = 0.4*(53-55.4)^2 + 0.6*(57-55.4)^2 = 0.4*5.76 + 0.6*2.56 = 3.84
    # sigma         = sqrt(3.84) ≈ 1.96
    assert bands["sigma_f"] == pytest.approx(1.96, abs=0.05)


def test_bands_combines_both_independent_sources():
    """sigma_total² = sigma_gefs² + sigma_model²"""
    bands = _compute_forecast_bands(
        final_f=55.0,
        gefs_sigma=2.0,
        contributing_sources=[
            ("hrrr", 53.0, 0.4),
            ("nws_point", 57.0, 0.6),
        ],
    )
    # Expected: sqrt(2.0² + 1.96²) = sqrt(4 + 3.84) = sqrt(7.84) ≈ 2.80
    assert bands["sigma_f"] == pytest.approx(2.80, abs=0.05)


def test_bands_excludes_gfs_ensemble_from_disagreement():
    """The gfs_ensemble source must not contribute to model-disagreement variance."""
    with_gefs = _compute_forecast_bands(
        final_f=55.0,
        gefs_sigma=2.0,
        contributing_sources=[
            ("hrrr", 55.0, 0.5),
            ("gfs_ensemble", 60.0, 0.5),  # outlier, excluded from disagreement
        ],
    )
    # model_disagreement_var should be 0 (only hrrr remains, single contributor)
    # sigma_total = sqrt(4 + 0) = 2.0
    assert with_gefs["sigma_f"] == pytest.approx(2.0, abs=0.05)


def test_bands_floors_at_half_a_degree():
    """sigma_total is floored at 0.5 to prevent overconfident bands."""
    bands = _compute_forecast_bands(
        final_f=55.0,
        gefs_sigma=0.1,
        contributing_sources=[("hrrr", 55.0, 0.5)],
    )
    assert bands["sigma_f"] >= 0.5


def test_bands_zero_contributors_still_emits_floor_band():
    """If everything is missing, we still emit a floored sigma rather than crashing."""
    bands = _compute_forecast_bands(
        final_f=55.0,
        gefs_sigma=None,
        contributing_sources=[],
    )
    assert bands["sigma_f"] == 0.5
    assert bands["p10"] == pytest.approx(55.0 - _Z_80 * 0.5, abs=0.05)


def test_bands_method_marker_present():
    """The bands payload includes a method marker for future migration."""
    bands = _compute_forecast_bands(55.0, 1.0, [])
    assert bands["method"] == "gaussian_variance_decomposition_v1"
