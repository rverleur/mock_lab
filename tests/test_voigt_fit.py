import numpy as np

from mock_lab.spectroscopy.voigt import (
    VoigtFitParameters,
    evaluate_voigt_spectrum,
    fit_voigt_spectrum,
)


def test_fit_voigt_spectrum_recovers_synthetic_parameters() -> None:
    frequency_cm_inv = np.linspace(-0.32, 0.05, 500)
    true_parameters = VoigtFitParameters(
        temperature_k=2400.0,
        line_centers_relative_cm_inv=np.array([-0.140, -0.240, -0.112]),
        collisional_hwhm_cm_inv=np.array([0.030, 0.045, 0.035]),
        line_areas=np.array([0.010, 0.002, 0.0008]),
        baseline_offset=0.002,
        baseline_slope=-0.01,
    )
    absorbance, _, _, _ = evaluate_voigt_spectrum(frequency_cm_inv, true_parameters)

    fit_result = fit_voigt_spectrum(frequency_cm_inv, absorbance)

    assert fit_result.success
    assert np.isclose(fit_result.parameters.temperature_k, true_parameters.temperature_k, rtol=0.35)
    assert np.allclose(
        fit_result.parameters.line_centers_relative_cm_inv,
        true_parameters.line_centers_relative_cm_inv,
        atol=5.0e-3,
    )
    assert np.allclose(
        fit_result.parameters.collisional_hwhm_cm_inv,
        true_parameters.collisional_hwhm_cm_inv,
        rtol=0.3,
    )
    assert np.allclose(fit_result.parameters.line_areas, true_parameters.line_areas, rtol=0.25)
    assert fit_result.rmse_absorbance < 1.0e-4
