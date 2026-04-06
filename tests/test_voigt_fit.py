import numpy as np

from mock_lab.spectroscopy.voigt import (
    DEFAULT_CO_TRANSITIONS,
    VoigtFitParameters,
    evaluate_voigt_spectrum,
    fit_voigt_spectrum,
    transition_relative_offsets,
    transition_strength_ratios,
)


def test_fit_voigt_spectrum_recovers_synthetic_parameters() -> None:
    frequency_cm_inv = np.linspace(-0.32, 0.05, 500)
    temperature_k = 2400.0
    anchor_center = -0.140
    offsets = transition_relative_offsets(DEFAULT_CO_TRANSITIONS, anchor_index=0)
    strength_ratios = transition_strength_ratios(
        temperature_k,
        transitions=DEFAULT_CO_TRANSITIONS,
        anchor_index=0,
    )
    strongest_line_area = 0.010
    true_parameters = VoigtFitParameters(
        temperature_k=temperature_k,
        line_centers_relative_cm_inv=np.array(
            [
                anchor_center,
                anchor_center + offsets[1] + 0.006,
                anchor_center + offsets[2],
            ]
        ),
        collisional_hwhm_cm_inv=np.array([0.030, 0.045, 0.045]),
        line_areas=strongest_line_area * strength_ratios,
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
    fitted_strength_ratios = transition_strength_ratios(
        fit_result.parameters.temperature_k,
        transitions=DEFAULT_CO_TRANSITIONS,
        anchor_index=0,
    )
    assert np.allclose(
        fit_result.parameters.line_areas / fit_result.parameters.line_areas[0],
        fitted_strength_ratios,
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    assert fit_result.parameters.line_areas[0] > fit_result.parameters.line_areas[1]
    assert fit_result.parameters.line_areas[1] > fit_result.parameters.line_areas[2]
    assert fit_result.rmse_absorbance < 1.0e-4
