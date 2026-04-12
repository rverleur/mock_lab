"""Monte Carlo refit pipeline for spectroscopy-driven state uncertainty."""

from __future__ import annotations

import csv
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.format import open_memmap
from numpy.typing import NDArray

from mock_lab.io.matlab_loader import DEFAULT_PHASE_START_S, DEFAULT_REFERENCE_FREQUENCY_HZ
from mock_lab.plotting.figures import plot_state_history, save_figure
from mock_lab.spectroscopy.collisional_broadening import (
    BATH_GAS_REFERENCE_SPECIES,
    COLLISION_PARTNERS,
    DEFAULT_MIXTURE_COMPOSITION_MODEL,
    NON_CO_COLLISION_PARTNERS,
)
from mock_lab.spectroscopy.state_estimation import (
    DEFAULT_OPTICAL_PATH_LENGTH_CM,
    StateHistory,
    build_state_history,
    estimate_state_model_uncertainty,
    evaluate_state_from_fit_parameters,
)
from mock_lab.spectroscopy.voigt import (
    DEFAULT_CO_TRANSITIONS,
    Transition,
    VoigtFitParameters,
    VoigtFitResult,
    expand_constrained_parameters,
    finite_difference_jacobian,
    fit_voigt_spectra,
    hitran_line_strength_to_cm2_atm,
)


Array1D = NDArray[np.float64]
Array2D = NDArray[np.float64]
Array3D = NDArray[np.float64]
Bool2D = NDArray[np.bool_]

MONTE_CARLO_VERSION = 5
STATE_COMPONENTS = ("temperature_k", "pressure_atm", "co_mole_fraction")
SAMPLED_TRANSITION_PARAMETER_NAMES = (
    "center_cm_inv",
    "line_strength_ref_cm2_atm",
    "CO_gamma_ref_cm_inv_atm",
    "CO_temperature_exponent",
    "bath_gamma_ref_cm_inv_atm",
    "bath_temperature_exponent",
)


@dataclass(frozen=True)
class MonteCarloStateHistoryResult:
    """Summary arrays produced by the Monte Carlo refit pipeline."""

    scan_index: Array1D
    scan_time_s: Array1D
    temperature_k: Array1D
    pressure_atm: Array1D
    co_mole_fraction: Array1D
    temperature_total_lower: Array1D
    temperature_total_upper: Array1D
    pressure_total_lower: Array1D
    pressure_total_upper: Array1D
    co_mole_fraction_total_lower: Array1D
    co_mole_fraction_total_upper: Array1D
    completed_trials: int
    confidence_level: float


def _finite_uncertainty(value: float) -> float:
    """Return a finite uncertainty half-width, treating unavailable values as zero."""

    return float(value) if np.isfinite(value) else 0.0


def _parameter_half_widths(
    transitions: tuple[Transition, ...],
) -> tuple[dict[str, Array1D], dict[str, Array1D]]:
    """Return half-widths for line-position/strength and active broadening parameters."""

    line_half_widths = {
        "center_cm_inv": np.asarray(
            [
                _finite_uncertainty(
                    transition.uncertainties["wavenumber_cm_inv"].upper_bound_absolute
                )
                for transition in transitions
            ],
            dtype=float,
        ),
        "line_strength_ref": np.asarray(
            [
                _finite_uncertainty(
                    transition.uncertainties["line_strength"].upper_bound_absolute
                )
                for transition in transitions
            ],
            dtype=float,
        ),
    }
    broadening_half_widths = {
        "co_gamma_ref_cm_inv_atm": np.asarray(
            [
                transition.broadening_by_species["CO"].gamma_ref_cm_inv_atm
                * transition.broadening_by_species["CO"].gamma_relative_uncertainty
                for transition in transitions
            ],
            dtype=float,
        ),
        "co_temperature_exponent": np.asarray(
            [
                transition.broadening_by_species["CO"].temperature_exponent
                * transition.broadening_by_species["CO"].exponent_relative_uncertainty
                for transition in transitions
            ],
            dtype=float,
        ),
        "bath_gamma_ref_cm_inv_atm": np.asarray(
            [
                transition.broadening_by_species[BATH_GAS_REFERENCE_SPECIES].gamma_ref_cm_inv_atm
                * transition.broadening_by_species[BATH_GAS_REFERENCE_SPECIES].gamma_relative_uncertainty
                for transition in transitions
            ],
            dtype=float,
        ),
        "bath_temperature_exponent": np.asarray(
            [
                transition.broadening_by_species[BATH_GAS_REFERENCE_SPECIES].temperature_exponent
                * transition.broadening_by_species[BATH_GAS_REFERENCE_SPECIES].exponent_relative_uncertainty
                for transition in transitions
            ],
            dtype=float,
        ),
    }
    return line_half_widths, broadening_half_widths


def sample_transitions_uniform(
    rng: np.random.Generator,
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
) -> tuple[Transition, ...]:
    """Sample one independent spectroscopy realization from uniform bounds."""

    line_half_widths, broadening_half_widths = _parameter_half_widths(transitions)
    sampled: list[Transition] = []

    for transition_index, transition in enumerate(transitions):
        sampled_center = float(
            transition.center_cm_inv
            + rng.uniform(
                -line_half_widths["center_cm_inv"][transition_index],
                line_half_widths["center_cm_inv"][transition_index],
            )
        )
        sampled_line_strength = float(
            transition.line_strength_ref
            + rng.uniform(
                -line_half_widths["line_strength_ref"][transition_index],
                line_half_widths["line_strength_ref"][transition_index],
            )
        )
        sampled_line_strength = max(sampled_line_strength, np.finfo(float).tiny)
        sampled_broadening = dict(transition.broadening_by_species)
        sampled_co_gamma = float(
            transition.broadening_by_species["CO"].gamma_ref_cm_inv_atm
            + rng.uniform(
                -broadening_half_widths["co_gamma_ref_cm_inv_atm"][transition_index],
                broadening_half_widths["co_gamma_ref_cm_inv_atm"][transition_index],
            )
        )
        sampled_co_exponent = float(
            transition.broadening_by_species["CO"].temperature_exponent
            + rng.uniform(
                -broadening_half_widths["co_temperature_exponent"][transition_index],
                broadening_half_widths["co_temperature_exponent"][transition_index],
            )
        )
        sampled_bath_gamma = float(
            transition.broadening_by_species[BATH_GAS_REFERENCE_SPECIES].gamma_ref_cm_inv_atm
            + rng.uniform(
                -broadening_half_widths["bath_gamma_ref_cm_inv_atm"][transition_index],
                broadening_half_widths["bath_gamma_ref_cm_inv_atm"][transition_index],
            )
        )
        sampled_bath_exponent = float(
            transition.broadening_by_species[BATH_GAS_REFERENCE_SPECIES].temperature_exponent
            + rng.uniform(
                -broadening_half_widths["bath_temperature_exponent"][transition_index],
                broadening_half_widths["bath_temperature_exponent"][transition_index],
            )
        )
        sampled_broadening["CO"] = replace(
            transition.broadening_by_species["CO"],
            gamma_ref_cm_inv_atm=max(sampled_co_gamma, np.finfo(float).tiny),
            temperature_exponent=max(sampled_co_exponent, np.finfo(float).tiny),
        )
        for species in NON_CO_COLLISION_PARTNERS:
            sampled_broadening[species] = replace(
                transition.broadening_by_species[species],
                gamma_ref_cm_inv_atm=max(sampled_bath_gamma, np.finfo(float).tiny),
                temperature_exponent=max(sampled_bath_exponent, np.finfo(float).tiny),
            )

        sampled_line_strength_hitran = (
            sampled_line_strength / hitran_line_strength_to_cm2_atm(1.0)
        )

        sampled.append(
            replace(
                transition,
                center_cm_inv=sampled_center,
                line_strength_hitran_ref=sampled_line_strength_hitran,
                line_strength_ref=sampled_line_strength,
                broadening_by_species=sampled_broadening,
            )
        )

    return tuple(sampled)


def _transition_sample_vector(transitions: tuple[Transition, ...]) -> Array2D:
    """Pack the sampled transition values saved for audit and restart context."""

    return np.asarray(
        [
            [
                transition.center_cm_inv,
                transition.line_strength_ref,
                transition.broadening_by_species["CO"].gamma_ref_cm_inv_atm,
                transition.broadening_by_species["CO"].temperature_exponent,
                transition.broadening_by_species[BATH_GAS_REFERENCE_SPECIES].gamma_ref_cm_inv_atm,
                transition.broadening_by_species[BATH_GAS_REFERENCE_SPECIES].temperature_exponent,
            ]
            for transition in transitions
        ],
        dtype=float,
    )


def _state_fit_half_width(
    fit_result: VoigtFitResult,
    *,
    transitions: tuple[Transition, ...],
    optical_path_length_cm: float,
) -> Array1D:
    """Propagate one fit result's covariance into `[T, P, X_CO]` half-widths."""

    if not (
        np.all(np.isfinite(fit_result.reduced_parameter_vector))
        and np.all(np.isfinite(fit_result.reduced_parameter_covariance))
        and np.isfinite(fit_result.confidence_scale)
    ):
        return np.full(len(STATE_COMPONENTS), np.nan, dtype=float)

    def state_vector(parameter_vector: Array1D) -> Array1D:
        parameters = expand_constrained_parameters(
            parameter_vector,
            transitions=transitions,
        )
        return evaluate_state_from_fit_parameters(
            parameters,
            optical_path_length_cm=optical_path_length_cm,
            transitions=transitions,
            transition=transitions[0],
            composition_model=DEFAULT_MIXTURE_COMPOSITION_MODEL,
        )

    state_jacobian = finite_difference_jacobian(
        state_vector,
        np.asarray(fit_result.reduced_parameter_vector, dtype=float),
    )
    state_covariance = (
        state_jacobian
        @ np.asarray(fit_result.reduced_parameter_covariance, dtype=float)
        @ state_jacobian.T
    )
    state_standard_error = np.sqrt(
        np.clip(np.diag(np.asarray(state_covariance, dtype=float)), a_min=0.0, a_max=None)
    )
    return np.asarray(fit_result.confidence_scale * state_standard_error, dtype=float)


def _trial_outputs_from_fit_results(
    fit_results: tuple[VoigtFitResult | None, ...],
    *,
    transitions: tuple[Transition, ...],
    optical_path_length_cm: float,
) -> tuple[Array2D, Array2D, NDArray[np.bool_]]:
    """Convert one trial's fit stack into nominal states and within-trial half-widths."""

    scan_count = len(fit_results)
    temperature_k = np.full(scan_count, np.nan, dtype=float)
    collisional_hwhm_cm_inv = np.full((scan_count, len(transitions)), np.nan, dtype=float)
    strongest_line_area_cm_inv = np.full(scan_count, np.nan, dtype=float)
    fit_half_widths = np.full((scan_count, len(STATE_COMPONENTS)), np.nan, dtype=float)
    success = np.zeros(scan_count, dtype=bool)

    for scan_index, result in enumerate(fit_results):
        if result is None:
            continue

        temperature_k[scan_index] = result.parameters.temperature_k
        collisional_hwhm_cm_inv[scan_index] = result.parameters.collisional_hwhm_cm_inv
        strongest_line_area_cm_inv[scan_index] = result.parameters.line_areas[0]
        fit_half_width = _state_fit_half_width(
            result,
            transitions=transitions,
            optical_path_length_cm=optical_path_length_cm,
        )
        model_form_half_width = estimate_state_model_uncertainty(
            result.parameters,
            optical_path_length_cm=optical_path_length_cm,
            transitions=transitions,
            transition=transitions[0],
            composition_model=DEFAULT_MIXTURE_COMPOSITION_MODEL,
            include_broadening_parameters=False,
            include_bath_gas_model=False,
            include_line_consistency=True,
        )
        fit_half_widths[scan_index] = np.sqrt(
            np.clip(fit_half_width, a_min=0.0, a_max=None) ** 2
            + np.clip(model_form_half_width, a_min=0.0, a_max=None) ** 2
        )
        success[scan_index] = bool(result.success)

    state_history = build_state_history(
        temperature_k,
        collisional_hwhm_cm_inv,
        strongest_line_area_cm_inv,
        sweep_frequency_hz=DEFAULT_REFERENCE_FREQUENCY_HZ,
        optical_path_length_cm=optical_path_length_cm,
        transition=transitions[0],
        transitions=transitions,
        composition_model=DEFAULT_MIXTURE_COMPOSITION_MODEL,
    )
    nominal_state = np.stack(
        [
            state_history.temperature_k,
            state_history.pressure_atm,
            state_history.co_mole_fraction,
        ],
        axis=1,
    )
    return nominal_state, fit_half_widths, success


def _fit_trial(
    *,
    frequency_cm_inv: Array1D,
    absorbance_sweeps: Array2D,
    trial_seed: int,
    confidence_level: float,
    optical_path_length_cm: float,
    minimum_peak_absorbance: float,
) -> tuple[Array2D, Array2D, NDArray[np.bool_], Array2D]:
    """Run one full trial: sample transitions, refit all sweeps, and reduce state."""

    rng = np.random.default_rng(trial_seed)
    sampled_transitions = sample_transitions_uniform(rng)
    fit_results = fit_voigt_spectra(
        frequency_cm_inv,
        absorbance_sweeps,
        transitions=sampled_transitions,
        minimum_peak_absorbance=minimum_peak_absorbance,
        confidence_level=confidence_level,
    )
    nominal_state, fit_half_widths, success = _trial_outputs_from_fit_results(
        fit_results,
        transitions=sampled_transitions,
        optical_path_length_cm=optical_path_length_cm,
    )
    return (
        nominal_state,
        fit_half_widths,
        success,
        _transition_sample_vector(sampled_transitions),
    )


def _compute_chunk_worker(
    specification: dict[str, object],
) -> tuple[int, int, Array3D, Array3D, Bool2D, Array3D]:
    """Run a contiguous chunk of full MC trials in one worker."""

    start = int(specification["start"])
    stop = int(specification["stop"])
    voigt_fit_data = Path(specification["voigt_fit_data"])
    confidence_level = float(specification["confidence_level"])
    optical_path_length_cm = float(specification["optical_path_length_cm"])
    minimum_peak_absorbance = float(specification["minimum_peak_absorbance"])
    seed = int(specification["seed"])

    with np.load(voigt_fit_data) as data:
        frequency_cm_inv = np.asarray(data["frequency_cm_inv"], dtype=float)
        absorbance_sweeps = np.asarray(data["absorbance_sweeps"], dtype=float)

    scan_count = absorbance_sweeps.shape[0]
    chunk_size = stop - start
    nominal_state_chunk = np.full(
        (chunk_size, scan_count, len(STATE_COMPONENTS)),
        np.nan,
        dtype=float,
    )
    fit_half_width_chunk = np.full_like(nominal_state_chunk, np.nan)
    success_chunk = np.zeros((chunk_size, scan_count), dtype=bool)
    sampled_transition_chunk = np.full(
        (
            chunk_size,
            len(DEFAULT_CO_TRANSITIONS),
            len(SAMPLED_TRANSITION_PARAMETER_NAMES),
        ),
        np.nan,
        dtype=float,
    )

    for local_trial_index, trial_index in enumerate(range(start, stop)):
        (
            nominal_state_chunk[local_trial_index],
            fit_half_width_chunk[local_trial_index],
            success_chunk[local_trial_index],
            sampled_transition_chunk[local_trial_index],
        ) = _fit_trial(
            frequency_cm_inv=frequency_cm_inv,
            absorbance_sweeps=absorbance_sweeps,
            trial_seed=seed + trial_index,
            confidence_level=confidence_level,
            optical_path_length_cm=optical_path_length_cm,
            minimum_peak_absorbance=minimum_peak_absorbance,
        )

    return (
        start,
        stop,
        nominal_state_chunk,
        fit_half_width_chunk,
        success_chunk,
        sampled_transition_chunk,
    )


def _metadata_path(output_dir: Path) -> Path:
    """Return the path to the resumable MC metadata JSON file."""

    return output_dir / "state_history_refit_mc_metadata.json"


def _state_samples_path(output_dir: Path) -> Path:
    """Return the path to the per-trial nominal-state memmap."""

    return output_dir / "state_history_refit_trial_states.npy"


def _fit_half_widths_path(output_dir: Path) -> Path:
    """Return the path to the per-trial fit half-width memmap."""

    return output_dir / "state_history_refit_trial_fit_half_widths.npy"


def _success_path(output_dir: Path) -> Path:
    """Return the path to the per-trial fit-success mask memmap."""

    return output_dir / "state_history_refit_trial_success.npy"


def _sampled_transition_path(output_dir: Path) -> Path:
    """Return the path to the sampled-transition audit memmap."""

    return output_dir / "state_history_refit_sampled_transitions.npy"


def _summary_path(output_dir: Path) -> Path:
    """Return the path to the MC summary NPZ file."""

    return output_dir / "state_history_monte_carlo_summary.npz"


def _initial_metadata(
    *,
    trial_count: int,
    scan_count: int,
    seed: int,
    confidence_level: float,
    optical_path_length_cm: float,
    minimum_peak_absorbance: float,
) -> dict[str, object]:
    """Build metadata for a new resumable MC refit run."""

    return {
        "version": MONTE_CARLO_VERSION,
        "trial_count": int(trial_count),
        "scan_count": int(scan_count),
        "state_components": list(STATE_COMPONENTS),
        "sampled_transition_parameter_names": list(SAMPLED_TRANSITION_PARAMETER_NAMES),
        "completed_trials": 0,
        "seed": int(seed),
        "confidence_level": float(confidence_level),
        "optical_path_length_cm": float(optical_path_length_cm),
        "minimum_peak_absorbance": float(minimum_peak_absorbance),
        "transition_source_csv_rows": [
            int(transition.source_csv_row)
            for transition in DEFAULT_CO_TRANSITIONS
        ],
        "sampling_note": (
            "Uniform line-center/linestrength sampling from HiTEMP bounds, "
            "uniform sampling of CO self-broadening plus the N2-equivalent "
            "bath-gas broadening within their 5% parameter bounds, "
            "and a full Voigt refit for every Monte Carlo trial. The saved "
            "within-trial half-widths also include the line-to-line pressure "
            "consistency term."
        ),
    }


def _load_or_create_storage(
    output_dir: Path,
    *,
    trial_count: int,
    scan_count: int,
    seed: int,
    confidence_level: float,
    optical_path_length_cm: float,
    minimum_peak_absorbance: float,
    force: bool,
) -> tuple[Array3D, Array3D, Bool2D, Array3D, dict[str, object]]:
    """Open or create all resumable MC refit memmaps plus metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = _metadata_path(output_dir)
    state_samples_path = _state_samples_path(output_dir)
    fit_half_widths_path = _fit_half_widths_path(output_dir)
    success_path = _success_path(output_dir)
    sampled_transition_path = _sampled_transition_path(output_dir)

    if force:
        metadata_path.unlink(missing_ok=True)
        state_samples_path.unlink(missing_ok=True)
        fit_half_widths_path.unlink(missing_ok=True)
        success_path.unlink(missing_ok=True)
        sampled_transition_path.unlink(missing_ok=True)

    required_paths = (
        metadata_path,
        state_samples_path,
        fit_half_widths_path,
        success_path,
        sampled_transition_path,
    )

    if any(path.exists() for path in required_paths):
        if not all(path.exists() for path in required_paths):
            raise RuntimeError(
                "Incomplete Monte Carlo refit files found. Use force=True to restart."
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "version": MONTE_CARLO_VERSION,
            "trial_count": int(trial_count),
            "scan_count": int(scan_count),
            "seed": int(seed),
            "confidence_level": float(confidence_level),
            "optical_path_length_cm": float(optical_path_length_cm),
            "minimum_peak_absorbance": float(minimum_peak_absorbance),
            "transition_source_csv_rows": [
                int(transition.source_csv_row)
                for transition in DEFAULT_CO_TRANSITIONS
            ],
        }
        for key, expected_value in expected.items():
            if metadata.get(key) != expected_value:
                raise RuntimeError(
                    f"Existing Monte Carlo metadata has {key}={metadata.get(key)!r}, "
                    f"expected {expected_value!r}. Use force=True to restart."
                )

        state_samples = open_memmap(
            state_samples_path,
            mode="r+",
            dtype=float,
            shape=(trial_count, scan_count, len(STATE_COMPONENTS)),
        )
        fit_half_widths = open_memmap(
            fit_half_widths_path,
            mode="r+",
            dtype=float,
            shape=(trial_count, scan_count, len(STATE_COMPONENTS)),
        )
        success = open_memmap(
            success_path,
            mode="r+",
            dtype=np.bool_,
            shape=(trial_count, scan_count),
        )
        sampled_transitions = open_memmap(
            sampled_transition_path,
            mode="r+",
            dtype=float,
            shape=(
                trial_count,
                len(DEFAULT_CO_TRANSITIONS),
                len(SAMPLED_TRANSITION_PARAMETER_NAMES),
            ),
        )
        return state_samples, fit_half_widths, success, sampled_transitions, metadata

    metadata = _initial_metadata(
        trial_count=trial_count,
        scan_count=scan_count,
        seed=seed,
        confidence_level=confidence_level,
        optical_path_length_cm=optical_path_length_cm,
        minimum_peak_absorbance=minimum_peak_absorbance,
    )
    state_samples = open_memmap(
        state_samples_path,
        mode="w+",
        dtype=float,
        shape=(trial_count, scan_count, len(STATE_COMPONENTS)),
    )
    fit_half_widths = open_memmap(
        fit_half_widths_path,
        mode="w+",
        dtype=float,
        shape=(trial_count, scan_count, len(STATE_COMPONENTS)),
    )
    success = open_memmap(
        success_path,
        mode="w+",
        dtype=np.bool_,
        shape=(trial_count, scan_count),
    )
    sampled_transitions = open_memmap(
        sampled_transition_path,
        mode="w+",
        dtype=float,
        shape=(
            trial_count,
            len(DEFAULT_CO_TRANSITIONS),
            len(SAMPLED_TRANSITION_PARAMETER_NAMES),
        ),
    )
    state_samples[:] = np.nan
    fit_half_widths[:] = np.nan
    success[:] = False
    sampled_transitions[:] = np.nan
    state_samples.flush()
    fit_half_widths.flush()
    success.flush()
    sampled_transitions.flush()
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return state_samples, fit_half_widths, success, sampled_transitions, metadata


def _write_trial_chunk(
    state_samples: Array3D,
    fit_half_widths: Array3D,
    success: Bool2D,
    sampled_transitions: Array3D,
    metadata: dict[str, object],
    metadata_path: Path,
    start: int,
    stop: int,
    nominal_state_chunk: Array3D,
    fit_half_width_chunk: Array3D,
    success_chunk: Bool2D,
    sampled_transition_chunk: Array3D,
) -> int:
    """Write one complete trial chunk and checkpoint the completed count."""

    state_samples[start:stop] = nominal_state_chunk
    fit_half_widths[start:stop] = fit_half_width_chunk
    success[start:stop] = success_chunk
    sampled_transitions[start:stop] = sampled_transition_chunk
    state_samples.flush()
    fit_half_widths.flush()
    success.flush()
    sampled_transitions.flush()
    metadata["completed_trials"] = int(stop)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return int(stop)


def summarize_monte_carlo_samples(
    state_samples: Array3D,
    fit_half_widths: Array3D,
    state_history_data: Path | str,
    output_dir: Path | str,
    *,
    confidence_level: float,
) -> MonteCarloStateHistoryResult:
    """Summarize trial states and combine refit MC with fit covariance by RSS."""

    state_history_data = Path(state_history_data)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(state_history_data) as data:
        scan_index = np.asarray(data["scan_index"], dtype=float)
        scan_time_s = np.asarray(data["scan_time_s"], dtype=float)
        deterministic_temperature = np.asarray(data["temperature_k"], dtype=float)
        deterministic_pressure = np.asarray(data["pressure_atm"], dtype=float)
        deterministic_co_mole_fraction = np.asarray(data["co_mole_fraction"], dtype=float)

    lower_percentile = 50.0 * (1.0 - confidence_level)
    upper_percentile = 100.0 - lower_percentile
    mc_mean = np.nanmean(state_samples, axis=0)
    mc_lower = np.nanpercentile(state_samples, lower_percentile, axis=0)
    mc_upper = np.nanpercentile(state_samples, upper_percentile, axis=0)
    fit_half_width_rms = np.sqrt(np.nanmean(fit_half_widths**2, axis=0))

    mc_lower_half_width = np.abs(mc_mean - mc_lower)
    mc_upper_half_width = np.abs(mc_upper - mc_mean)
    total_lower = mc_mean - np.sqrt(mc_lower_half_width**2 + fit_half_width_rms**2)
    total_upper = mc_mean + np.sqrt(mc_upper_half_width**2 + fit_half_width_rms**2)
    fit_lower = mc_mean - fit_half_width_rms
    fit_upper = mc_mean + fit_half_width_rms

    np.savez(
        _summary_path(output_dir),
        scan_index=scan_index,
        scan_time_s=scan_time_s,
        deterministic_temperature_k=deterministic_temperature,
        deterministic_pressure_atm=deterministic_pressure,
        deterministic_co_mole_fraction=deterministic_co_mole_fraction,
        mc_mean_temperature_k=mc_mean[:, 0],
        mc_mean_pressure_atm=mc_mean[:, 1],
        mc_mean_co_mole_fraction=mc_mean[:, 2],
        mc_refit_lower_temperature_k=mc_lower[:, 0],
        mc_refit_upper_temperature_k=mc_upper[:, 0],
        mc_refit_lower_pressure_atm=mc_lower[:, 1],
        mc_refit_upper_pressure_atm=mc_upper[:, 1],
        mc_refit_lower_co_mole_fraction=mc_lower[:, 2],
        mc_refit_upper_co_mole_fraction=mc_upper[:, 2],
        fit_lower_temperature_k=fit_lower[:, 0],
        fit_upper_temperature_k=fit_upper[:, 0],
        fit_lower_pressure_atm=fit_lower[:, 1],
        fit_upper_pressure_atm=fit_upper[:, 1],
        fit_lower_co_mole_fraction=fit_lower[:, 2],
        fit_upper_co_mole_fraction=fit_upper[:, 2],
        total_lower_temperature_k=total_lower[:, 0],
        total_upper_temperature_k=total_upper[:, 0],
        total_lower_pressure_atm=total_lower[:, 1],
        total_upper_pressure_atm=total_upper[:, 1],
        total_lower_co_mole_fraction=total_lower[:, 2],
        total_upper_co_mole_fraction=total_upper[:, 2],
        confidence_level=confidence_level,
        completed_trials=state_samples.shape[0],
    )
    _write_monte_carlo_csv(
        output_dir / "state_history_monte_carlo_summary.csv",
        scan_index=scan_index,
        scan_time_s=scan_time_s,
        deterministic_state=np.stack(
            [deterministic_temperature, deterministic_pressure, deterministic_co_mole_fraction],
            axis=1,
        ),
        mc_mean=mc_mean,
        mc_lower=mc_lower,
        mc_upper=mc_upper,
        fit_lower=fit_lower,
        fit_upper=fit_upper,
        total_lower=total_lower,
        total_upper=total_upper,
    )

    return MonteCarloStateHistoryResult(
        scan_index=scan_index,
        scan_time_s=scan_time_s,
        temperature_k=mc_mean[:, 0],
        pressure_atm=mc_mean[:, 1],
        co_mole_fraction=mc_mean[:, 2],
        temperature_total_lower=total_lower[:, 0],
        temperature_total_upper=total_upper[:, 0],
        pressure_total_lower=total_lower[:, 1],
        pressure_total_upper=total_upper[:, 1],
        co_mole_fraction_total_lower=total_lower[:, 2],
        co_mole_fraction_total_upper=total_upper[:, 2],
        completed_trials=state_samples.shape[0],
        confidence_level=confidence_level,
    )


def _write_monte_carlo_csv(
    path: Path,
    *,
    scan_index: Array1D,
    scan_time_s: Array1D,
    deterministic_state: Array2D,
    mc_mean: Array2D,
    mc_lower: Array2D,
    mc_upper: Array2D,
    fit_lower: Array2D,
    fit_upper: Array2D,
    total_lower: Array2D,
    total_upper: Array2D,
) -> None:
    """Write the summarized MC refit state history as a CSV table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "scan_index",
        "scan_time_s",
        "deterministic_temperature_k",
        "deterministic_pressure_atm",
        "deterministic_co_mole_fraction",
        "temperature_k",
        "temperature_refit_lower",
        "temperature_refit_upper",
        "temperature_fit_lower",
        "temperature_fit_upper",
        "temperature_total_lower",
        "temperature_total_upper",
        "pressure_atm",
        "pressure_refit_lower",
        "pressure_refit_upper",
        "pressure_fit_lower",
        "pressure_fit_upper",
        "pressure_total_lower",
        "pressure_total_upper",
        "co_mole_fraction",
        "co_mole_fraction_refit_lower",
        "co_mole_fraction_refit_upper",
        "co_mole_fraction_fit_lower",
        "co_mole_fraction_fit_upper",
        "co_mole_fraction_total_lower",
        "co_mole_fraction_total_upper",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)

        for row_index in range(scan_index.size):
            writer.writerow(
                [
                    f"{scan_index[row_index]:.8g}",
                    f"{scan_time_s[row_index]:.8g}",
                    f"{deterministic_state[row_index, 0]:.8g}",
                    f"{deterministic_state[row_index, 1]:.8g}",
                    f"{deterministic_state[row_index, 2]:.8g}",
                    f"{mc_mean[row_index, 0]:.8g}",
                    f"{mc_lower[row_index, 0]:.8g}",
                    f"{mc_upper[row_index, 0]:.8g}",
                    f"{fit_lower[row_index, 0]:.8g}",
                    f"{fit_upper[row_index, 0]:.8g}",
                    f"{total_lower[row_index, 0]:.8g}",
                    f"{total_upper[row_index, 0]:.8g}",
                    f"{mc_mean[row_index, 1]:.8g}",
                    f"{mc_lower[row_index, 1]:.8g}",
                    f"{mc_upper[row_index, 1]:.8g}",
                    f"{fit_lower[row_index, 1]:.8g}",
                    f"{fit_upper[row_index, 1]:.8g}",
                    f"{total_lower[row_index, 1]:.8g}",
                    f"{total_upper[row_index, 1]:.8g}",
                    f"{mc_mean[row_index, 2]:.8g}",
                    f"{mc_lower[row_index, 2]:.8g}",
                    f"{mc_upper[row_index, 2]:.8g}",
                    f"{fit_lower[row_index, 2]:.8g}",
                    f"{fit_upper[row_index, 2]:.8g}",
                    f"{total_lower[row_index, 2]:.8g}",
                    f"{total_upper[row_index, 2]:.8g}",
                ]
            )


def run_monte_carlo_state_history_pipeline(
    voigt_fit_data: Path | str,
    state_history_data: Path | str,
    output_dir: Path | str,
    *,
    figure_output_dir: Path | str | None = None,
    trial_count: int = 1000,
    chunk_size: int = 100,
    seed: int = 617,
    confidence_level: float = 0.95,
    optical_path_length_cm: float = DEFAULT_OPTICAL_PATH_LENGTH_CM,
    minimum_peak_absorbance: float = 0.02,
    force: bool = False,
    workers: int = 1,
) -> MonteCarloStateHistoryResult:
    """Run or resume the full-refit spectroscopy MC pipeline."""

    if trial_count <= 0:
        raise ValueError("Trial count must be positive.")
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")
    if workers <= 0:
        raise ValueError("Worker count must be positive.")

    voigt_fit_data = Path(voigt_fit_data)
    state_history_data = Path(state_history_data)
    output_dir = Path(output_dir)
    figure_output_dir = output_dir if figure_output_dir is None else Path(figure_output_dir)
    figure_output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(voigt_fit_data) as data:
        scan_count = int(np.asarray(data["absorbance_sweeps"], dtype=float).shape[0])

    (
        state_samples,
        fit_half_widths,
        success,
        sampled_transitions,
        metadata,
    ) = _load_or_create_storage(
        output_dir,
        trial_count=trial_count,
        scan_count=scan_count,
        seed=seed,
        confidence_level=confidence_level,
        optical_path_length_cm=optical_path_length_cm,
        minimum_peak_absorbance=minimum_peak_absorbance,
        force=force,
    )
    completed_trials = int(metadata["completed_trials"])
    metadata_path = _metadata_path(output_dir)

    chunk_specs = [
        {
            "start": start,
            "stop": min(start + chunk_size, trial_count),
            "seed": seed,
            "voigt_fit_data": str(voigt_fit_data),
            "confidence_level": confidence_level,
            "optical_path_length_cm": optical_path_length_cm,
            "minimum_peak_absorbance": minimum_peak_absorbance,
        }
        for start in range(completed_trials, trial_count, chunk_size)
    ]

    if workers == 1:
        for chunk_result in map(_compute_chunk_worker, chunk_specs):
            completed_trials = _write_trial_chunk(
                state_samples,
                fit_half_widths,
                success,
                sampled_transitions,
                metadata,
                metadata_path,
                *chunk_result,
            )
    else:
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for chunk_result in executor.map(_compute_chunk_worker, chunk_specs):
                    completed_trials = _write_trial_chunk(
                        state_samples,
                        fit_half_widths,
                        success,
                        sampled_transitions,
                        metadata,
                        metadata_path,
                        *chunk_result,
                    )
        except (OSError, PermissionError) as exc:
            warnings.warn(
                "ProcessPoolExecutor is unavailable; falling back to single-process "
                f"Monte Carlo refit chunks. Original error: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            remaining_specs = [
                spec
                for spec in chunk_specs
                if int(spec["stop"]) > completed_trials
            ]
            for chunk_result in map(_compute_chunk_worker, remaining_specs):
                completed_trials = _write_trial_chunk(
                    state_samples,
                    fit_half_widths,
                    success,
                    sampled_transitions,
                    metadata,
                    metadata_path,
                    *chunk_result,
                )

    completed_state_samples = np.asarray(state_samples[:completed_trials], dtype=float)
    completed_fit_half_widths = np.asarray(fit_half_widths[:completed_trials], dtype=float)
    result = summarize_monte_carlo_samples(
        completed_state_samples,
        completed_fit_half_widths,
        state_history_data,
        output_dir,
        confidence_level=confidence_level,
    )
    plot_monte_carlo_state_history(
        _summary_path(output_dir),
        figure_output_dir=figure_output_dir,
    )
    return result


def plot_monte_carlo_state_history(
    summary_data: Path | str,
    *,
    figure_output_dir: Path | str,
) -> Path:
    """Plot total MC-refit plus fit-covariance uncertainty from saved summary data."""

    summary_data = Path(summary_data)
    figure_output_dir = Path(figure_output_dir)
    figure_output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(summary_data) as data:
        confidence_level = float(np.asarray(data["confidence_level"]).item())
        figure = plot_state_history(
            1.0e6 * (
                np.asarray(data["scan_time_s"], dtype=float) + DEFAULT_PHASE_START_S
            ),
            np.asarray(data["mc_mean_temperature_k"], dtype=float),
            np.asarray(data["mc_mean_pressure_atm"], dtype=float),
            np.asarray(data["mc_mean_co_mole_fraction"], dtype=float),
            temperature_lower=np.asarray(data["total_lower_temperature_k"], dtype=float),
            temperature_upper=np.asarray(data["total_upper_temperature_k"], dtype=float),
            pressure_lower=np.asarray(data["total_lower_pressure_atm"], dtype=float),
            pressure_upper=np.asarray(data["total_upper_pressure_atm"], dtype=float),
            co_mole_fraction_lower=np.asarray(data["total_lower_co_mole_fraction"], dtype=float),
            co_mole_fraction_upper=np.asarray(data["total_upper_co_mole_fraction"], dtype=float),
            uncertainty_label=(
                f"{int(round(100.0 * confidence_level))}% total uncertainty"
            ),
            xlabel=r"Time [$\mu$s]",
        )

    output_path = figure_output_dir / "state_history_monte_carlo.png"
    save_figure(figure, output_path)
    plt.close(figure)
    return output_path
