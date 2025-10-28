####################################################################################################
# File:     signal_processing.py
# Purpose:  Main functions for onset extraction and signal processing
#
# Author:   Manuel Anglada-Tort, Peter Harrison, Nori Jacoby
####################################################################################################
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

from .utils import (
    read_audio_file,
    simple_resample,
    band_pass_sharp,
    cheap_hilbert,
    fade_in_out,
    ms_to_samples
)

@dataclass
class AudioSignal:
    """Container for audio signal data and metadata"""
    samples: np.ndarray
    sampling_rate: int
    duration: float
    timeline: np.ndarray

@dataclass 
class OnsetData:
    """Container for onset detection results"""
    timestamps: np.ndarray
    amplitudes: Optional[np.ndarray] = None
    
def extract_onsets(
    audio_signals: Dict[str, np.ndarray], 
    config: object
) -> Dict[str, np.ndarray]:
    """
    Extract onsets from audio signals.

    Args:
        audio_signals: Dictionary containing extracted signals including markers and tapping response
        config: Configuration parameters for the experiment

    Returns:
        Dictionary containing detected onsets for tapping and markers
        
    Raises:
        ValueError: If required signals are missing from audio_signals
    """
    if not {'rec_markers_clean', 'rec_tapping_clean'}.issubset(audio_signals.keys()):
        raise ValueError("Missing required signals: markers or tapping")

    return {
        'markers_detected_onsets': detect_onsets(
            audio_signals['rec_markers_clean'],
            config.EXTRACT_THRESH[0],
            config.EXTRACT_FIRST_WINDOW[0], 
            config.EXTRACT_SECOND_WINDOW[0],
            config.FS0
        ),
        'tapping_detected_onsets': detect_onsets(
            audio_signals['rec_tapping_clean'],
            config.EXTRACT_THRESH[1],
            config.EXTRACT_FIRST_WINDOW[1],
            config.EXTRACT_SECOND_WINDOW[1], 
            config.FS0
        )
    }


########################################
# Functions supporting onset extraction
########################################


def extract_onsets_only_tapping(
    audio_signals: Dict[str, np.ndarray],
    config: object
) -> Dict[str, Union[np.ndarray, int]]:
    """
    Extract onsets from tapping signal only.

    Args:
        audio_signals: Dictionary containing extracted signals from raw recording
        config: Configuration parameters for the experiment

    Returns:
        Dictionary containing detected tapping onsets and count
        
    Raises:
        ValueError: If tapping signal is missing from audio_signals
    """
    if 'rec_tapping_clean' not in audio_signals:
        raise ValueError("Missing required tapping signal")

    tapping_detected_onsets = detect_onsets(
        audio_signals['rec_tapping_clean'],
        config.EXTRACT_THRESH[1],
        config.EXTRACT_FIRST_WINDOW[1],
        config.EXTRACT_SECOND_WINDOW[1],
        config.FS0
    )

    return {
        'tapping_detected_onsets': tapping_detected_onsets,
        'num_tapping_detected_onsets': len(tapping_detected_onsets)
    }

def detect_onsets(
    samples: np.ndarray,
    threshold: float,
    first_window_ms: float,
    second_window_ms: float,
    sampling_rate: int
) -> np.ndarray:
    """
    Detect onsets in audio signal using amplitude threshold and timing constraints.
    
    Uses a two-stage approach:
    1. Detect samples above threshold
    2. Filter detected onsets based on minimum time windows between onsets

    Args:
        samples: Audio samples to analyze
        threshold: Relative amplitude threshold for onset detection
        first_window_ms: Minimum time between any onsets (ms)
        second_window_ms: Minimum time between accepted onsets (ms) 
        sampling_rate: Audio sampling rate in Hz

    Returns:
        Array of detected onset times in milliseconds
    """
    # Convert windows from ms to samples
    first_window_samples = ms_to_samples(first_window_ms, sampling_rate)
    second_window_samples = ms_to_samples(second_window_ms, sampling_rate)

    # Get absolute values of samples
    abs_samples = np.abs(samples)
    
    # Find potential onsets above threshold
    potential_onsets = []
    last_onset = 0
    
    for current_sample, amplitude in enumerate(abs_samples):
        if (amplitude > threshold and 
            (current_sample - last_onset) > first_window_samples):
            if (current_sample - last_onset) > second_window_samples:
                potential_onsets.append(current_sample)
            last_onset = current_sample

    # Filter onsets that are too close together
    filtered_onsets = filter_too_close_onsets(
        potential_onsets, 
        second_window_ms,
        sampling_rate
    )

    # Convert to milliseconds
    return (1000.0 * np.array(filtered_onsets)) / sampling_rate

def filter_too_close_onsets(
    onsets: List[int],
    min_interval_ms: float,
    sampling_rate: int
) -> np.ndarray:
    """
    Filter out onsets that are too close together.

    Args:
        onsets: List of onset times in samples
        min_interval_ms: Minimum allowed time between onsets in ms
        sampling_rate: Audio sampling rate in Hz

    Returns:
        Array of filtered onset times in samples
    """
    if not onsets:
        return np.array([])
        
    onsets = np.array(onsets)
    min_interval_samples = ms_to_samples(min_interval_ms, sampling_rate)
    
    # Calculate intervals between consecutive onsets
    intervals = np.diff(onsets)
    
    # Add infinity at start to keep first onset
    intervals = np.concatenate([[np.inf], intervals])
    
    # Keep onsets with sufficient interval from previous onset
    valid_onsets = intervals > min_interval_samples
    
    return onsets[valid_onsets]


def extract_audio_signals(
    rec_filename: str,
    config: object
) -> Dict[str, Union[np.ndarray, int, float]]:
    """
    Separate channels from mono recording and clean signals.

    Args:
        rec_filename: Path to the audio recording file
        config: Configuration parameters for the experiment

    Returns:
        Dictionary containing separated channels and signal processing metadata

    Raises:
        FileNotFoundError: If recording file doesn't exist
        ValueError: If audio file is invalid
    """
    # Load and downsample audio
    audio_data = downsample_audio(rec_filename, config.FS0)
    samples, fs_rec, rec_downsampled, time_line = audio_data
    
    # Separate channels
    channels = channel_separation(
        rec_downsampled,
        config.FS0,
        config.EXTRACT_COMPRESS_FACTOR,
        config.EXTRACT_FADE_IN,
        config.TAPPING_RANGE,
        config.TEST_RANGE,
        config.MARKERS_RANGE
    )

    # Clean signals
    cleaned_signals = signal_cleaning(
        channels["rec_tapping"],
        channels["rec_test"],
        channels["rec_markers"],
        config.FS0,
        config.CLEAN_BIN_WINDOW,
        config.CLEAN_MAX_RATIO,
        config.CLEAN_LOCATION_RATIO,
        config.CLEAN_NORMALIZE_FACTOR
    )

    return {
        'fs': config.FS0,
        'rec_original': samples,
        'fs_recording': fs_rec,
        'time_line_for_sample': time_line,
        'rec_downsampled': rec_downsampled,
        **channels,
        **cleaned_signals
    }

def downsample_audio(
    rec_filename: str, 
    target_fs: int
) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """
    Load and downsample audio file.

    Args:
        rec_filename: Path to audio file
        target_fs: Target sampling frequency

    Returns:
        Tuple containing:
        - Original samples
        - Original sampling rate
        - Downsampled signal
        - Time axis for samples

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If audio file is invalid
    """
    samples, fs_rec = read_audio_file(rec_filename)
    rec_downsampled = simple_resample(samples, fs_rec, target_fs)
    
    # Create timeline for plotting
    timeline = np.linspace(
        0, 
        len(rec_downsampled) / target_fs,
        len(rec_downsampled)
    )
    
    # Normalize downsampled signal
    rec_downsampled = rec_downsampled / np.max(np.abs(rec_downsampled))
    
    return samples, fs_rec, rec_downsampled, timeline

def channel_separation(
    recording: np.ndarray,
    fs: int,
    extract_compress_factor: float,
    fade_in: float,
    tapping_range: List[float],
    test_range: List[float],
    markers_range: List[float]
) -> Dict[str, np.ndarray]:
    """
    Separate mono recording into tapping, test and markers channels.

    Args:
        recording: Input audio signal
        fs: Sampling frequency
        extract_compress_factor: Compression factor for envelope extraction
        fade_in: Fade in duration in ms
        tapping_range: Frequency range for tapping channel
        test_range: Frequency range for test channel  
        markers_range: Frequency range for markers channel

    Returns:
        Dictionary containing separated channels
    """
    # Filter channels based on frequency ranges
    tapping_channel = filter_recording(recording, tapping_range, fs)
    test_channel = filter_recording(recording, test_range, fs)
    markers_channel = filter_recording(recording, markers_range, fs)

    # Normalize channels
    tapping_norm = 0.5 * tapping_channel / np.max(np.abs(tapping_channel))
    test_norm = 0.5 * test_channel / np.max(np.abs(markers_channel))
    markers_norm = 0.5 * markers_channel / np.max(np.abs(markers_channel))

    # Extract envelopes
    tapping_env = extract_envelope(tapping_norm, fs, extract_compress_factor, fade_in)
    test_env = extract_envelope(test_norm, fs, extract_compress_factor, fade_in)
    markers_env = extract_envelope(markers_norm, fs, extract_compress_factor, fade_in)

    return {
        "rec_tapping": normalize_max(tapping_env),
        "rec_test": normalize_max(test_env, ref=markers_env),
        "rec_markers": normalize_max(markers_env)
    }

def extract_envelope(
    signal: np.ndarray,
    fs: int,
    compress_factor: float,
    fade_ms: float
) -> np.ndarray:
    """
    Extract amplitude envelope from signal.

    Args:
        signal: Input audio signal
        fs: Sampling frequency
        compress_factor: Envelope compression factor
        fade_ms: Fade duration in milliseconds

    Returns:
        Signal envelope
    """
    envelope = np.power(
        cheap_hilbert(signal, fs),
        compress_factor
    )
    return fade_in_out(envelope, fs, fade_ms)

def normalize_max(signal: np.ndarray, ref: Optional[np.ndarray] = None) -> np.ndarray:
    """Normalize signal to maximum amplitude."""
    if ref is None:
        ref = signal
    return signal / np.max(np.abs(ref))

def filter_recording(
    recording: np.ndarray,
    freq_range: List[float],
    fs: int
) -> np.ndarray:
    """
    Apply bandpass filtering to recording.

    Args:
        recording: Input audio signal
        freq_range: List of frequency bands [min1, max1, min2, max2, ...]
        fs: Sampling frequency

    Returns:
        Filtered signal
    """
    if len(freq_range) > 2:
        filtered = np.zeros_like(recording)
        for i in range(0, len(freq_range), 2):
            band = band_pass_sharp(
                recording, 
                fs,
                freq_range[i],
                freq_range[i + 1]
            )
            filtered += band
    else:
        filtered = band_pass_sharp(
            recording,
            fs,
            min(freq_range),
            max(freq_range)
        )
    return filtered

def signal_cleaning(
    rec_tapping: np.ndarray,
    rec_test: np.ndarray,
    rec_markers: np.ndarray,
    fs: int,
    clean_bin_window: float,
    clean_max_ratio: float,
    clean_location_ratio: List[float],
    clean_normalize_factor: float
) -> Dict[str, Union[np.ndarray, Dict[str, float]]]:
    """
    Clean markers and tapping channels using adaptive filtering.

    Args:
        rec_tapping: Tapping channel recording
        rec_test: Test channel recording
        rec_markers: Markers channel recording
        fs: Sampling frequency
        clean_bin_window: Window size for cleaning (ms)
        clean_max_ratio: Maximum allowed cleaning ratio
        clean_location_ratio: [start, end] ratios for signal normalization
        clean_normalize_factor: Normalization factor

    Returns:
        Dictionary containing cleaned signals and SNR information
    """
    # Clean markers by comparing with test channel
    rec_test_max = cheap_hilbert(rec_test, fs, window_length=clean_bin_window)
    rec_markers_max = cheap_hilbert(rec_markers, fs, window_length=clean_bin_window)
    
    # Calculate cleaning ratio
    ratio_clean_seed = rec_markers_max / (rec_test_max + 1e-10)
    ratio_clean = np.clip(
        ratio_clean_seed,
        1.0 / clean_max_ratio,
        clean_max_ratio
    )
    
    # Apply cleaning
    rec_markers_clean = rec_markers * ratio_clean
    max_markers = np.max(np.abs(rec_markers_clean))
    
    # Normalize channels
    rec_test_final = rec_test / max_markers
    rec_markers_final = rec_markers / max_markers
    rec_markers_clean = rec_markers_clean / max_markers

    # Clean tapping channel
    rec_tapping_clean, start_include, end_include = signal_cleaning_tapping(
        rec_tapping,
        clean_location_ratio,
        clean_normalize_factor
    )

    # Calculate SNR metrics
    snr_info = calculate_snr_metrics(
        rec_markers_clean,
        rec_tapping_clean,
        rec_markers,
        rec_test,
        ratio_clean_seed,
        start_include,
        end_include
    )

    return {
        "rec_tapping_clean": rec_tapping_clean,
        "rec_test_final": rec_test_final,
        "rec_markers_final": rec_markers_final,
        "rec_markers_clean": rec_markers_clean,
        "ratio_clean": ratio_clean,
        "snr_info": snr_info
    }

def signal_cleaning_tapping(
    rec_tapping: np.ndarray,
    clean_location_ratio: List[float],
    clean_normalize_factor: float
) -> Tuple[np.ndarray, int, int]:
    """
    Clean tapping signal using adaptive normalization.

    Args:
        rec_tapping: Tapping channel recording
        clean_location_ratio: [start, end] ratios for normalization window
        clean_normalize_factor: Normalization factor

    Returns:
        Tuple containing:
        - Cleaned tapping signal
        - Start sample of cleaning window
        - End sample of cleaning window
    """
    # Calculate normalization window
    start_include = round(len(rec_tapping) * min(clean_location_ratio))
    end_include = round(len(rec_tapping) * max(clean_location_ratio))

    # Apply selective normalization
    rec_tapping_normalized = np.copy(rec_tapping)
    rec_tapping_normalized[:start_include] *= clean_normalize_factor
    rec_tapping_normalized[end_include:] *= clean_normalize_factor

    # Normalize and clip
    rec_tapping_clean = rec_tapping / np.max(np.abs(rec_tapping_normalized))
    rec_tapping_clean = np.clip(rec_tapping_clean, -1, 1)

    return rec_tapping_clean, start_include, end_include

def calculate_snr_metrics(
    rec_markers_clean: np.ndarray,
    rec_tapping_clean: np.ndarray,
    rec_markers: np.ndarray,
    rec_test: np.ndarray,
    ratio_clean_seed: np.ndarray,
    start_include: int,
    end_include: int
) -> Dict[str, float]:
    """
    Calculate signal-to-noise ratio metrics.

    Args:
        rec_markers_clean: Cleaned markers signal
        rec_tapping_clean: Cleaned tapping signal
        rec_markers: Original markers signal
        rec_test: Original test signal
        ratio_clean_seed: Original cleaning ratio
        start_include: Start of analysis window
        end_include: End of analysis window

    Returns:
        Dictionary containing SNR metrics
    """
    # Calculate amplitude ratios
    mid2begratio_markers = np.max(rec_markers_clean[start_include:end_include]) / max(
        np.max(rec_markers_clean[:start_include]),
        np.max(rec_markers_clean[end_include:])
    )

    mid2begratio_tapping = np.max(rec_tapping_clean[start_include:end_include]) / max(
        np.max(rec_tapping_clean[:start_include]),
        np.max(rec_tapping_clean[end_include:])
    )

    # Calculate SNR
    markers_rms = np.sqrt(np.mean(rec_markers ** 2))
    test_rms = np.sqrt(np.mean(rec_test ** 2))
    snr = markers_rms / test_rms

    return {
        "snr": snr,
        "max_ratio": np.max(ratio_clean_seed),
        "mean_ratio": np.mean(ratio_clean_seed),
        "mid2begratio_markers": mid2begratio_markers,
        "mid2begratio_tap": mid2begratio_tapping
    }

def prepare_onsets_extraction(
    markers_onsets: np.ndarray,
    stim_onsets: np.ndarray,
    onset_is_played: np.ndarray
) -> Dict[str, List[float]]:
    """
    Prepare stimulus information for onset extraction.

    Args:
        markers_onsets: List of marker onset times
        stim_onsets: List of stimulus onset times
        onset_is_played: Boolean array indicating which onsets were played

    Returns:
        Dictionary containing prepared onset information
    """
    markers_onsets, stim_onsets_all_info = convert_to_numpy(
        markers_onsets,
        stim_onsets,
        onset_is_played
    )

    return {
        'markers_onsets': markers_onsets,
        'stim_onsets_all_info': stim_onsets_all_info,
        'stim_onsets_played': [c[0] for c in stim_onsets_all_info if abs(c[1] - 1) < 1e-10],
        'stim_onsets': [c[0] for c in stim_onsets_all_info],
        'stim_onsets_unplayed': [c[0] for c in stim_onsets_all_info if abs(c[1]) < 1e-10]
    }

def convert_to_numpy(
    markers_onsets: List[float],
    stim_onsets: List[float],
    onset_is_played: List[bool]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert onset information to numpy arrays.

    Args:
        markers_onsets: List of marker onset times
        stim_onsets: List of stimulus onset times
        onset_is_played: List indicating which onsets were played

    Returns:
        Tuple containing:
        - Marker onsets array
        - Combined stimulus onsets and played info array
    """
    stim_onsets_all_info = np.ones((len(stim_onsets), 2))
    stim_onsets_all_info[:, 0] = np.array(stim_onsets)
    stim_onsets_all_info[:, 1] = np.array(onset_is_played)
    
    return np.array(markers_onsets), stim_onsets_all_info


########################################
# Functions supporting onset alignment
########################################
def align_onsets(
    initial_onsets: Dict[str, np.ndarray],
    raw_extracted_onsets: Dict[str, np.ndarray],
    markers_matching_window: float,
    onset_matching_window: float,
    onset_matching_window_phase: List[float]
) -> Dict[str, Union[np.ndarray, List[float]]]:
    """
    Align tapping and stimulus onsets with detected markers.

    Args:
        initial_onsets: Initial onsets including stimulus and markers
        raw_extracted_onsets: Extracted onsets including tapping and markers
        markers_matching_window: Window for matching markers onsets (ms)
        onset_matching_window: Window for matching tapping onsets (ms)
        onset_matching_window_phase: Phase window for matching tapping onsets

    Returns:
        Dictionary containing aligned onsets and timing information
    """
    # Extract onsets
    markers_detected = raw_extracted_onsets['markers_detected_onsets']
    tapping_detected = raw_extracted_onsets['tapping_detected_onsets']
    markers_onsets = initial_onsets['markers_onsets']
    
    # Calculate minimum marker interval
    min_marker_isi = np.min(np.diff(np.array(markers_onsets)))

    # Get and correct stimulus onsets
    stim_onsets = initial_onsets['stim_onsets']
    stim_onsets_played = initial_onsets['stim_onsets_played']
    
    # Align to first marker
    stim_onsets_corrected = align_to_first_marker(
        stim_onsets,
        stim_onsets_played,
        markers_onsets,
        markers_detected,
        tapping_detected
    )
    
    # Match onsets
    matched_onsets = compute_matched_onsets(
        stim_onsets_corrected['stim'],
        stim_onsets_corrected['tapping'],
        onset_matching_window,
        onset_matching_window_phase
    )

    # Verify markers
    markers_verification = verify_onsets_detection(
        markers_detected,
        markers_onsets,
        markers_matching_window,
        onset_matching_window_phase
    )

    return {
        'stim_onsets_input': stim_onsets,
        'stim_onsets_detected': stim_onsets_corrected['stim'],
        'resp_onsets_detected': tapping_detected,
        'stim_onsets_is_played': stim_onsets_corrected['is_played'],
        'stim_onsets_aligned': matched_onsets['stim_matched'],
        'resp_onsets_aligned': matched_onsets['resp_matched'],
        'stim_ioi': matched_onsets['stim_ioi'],
        'resp_ioi': matched_onsets['resp_ioi'],
        'asynchrony': matched_onsets['asynchrony'],
        'mean_async': matched_onsets['mean_async'],
        'first_stim': matched_onsets['first_stim'],
        'num_resp_raw_onsets': float(len(tapping_detected)),
        'num_stim_raw_onsets': float(len(stim_onsets)),
        'markers_onsets_detected': markers_detected,
        'markers_onsets_aligned': markers_onsets - markers_onsets[0] + markers_detected[0],
        'markers_onsets_input': markers_onsets,
        **markers_verification
    }

def align_to_first_marker(
    stim_onsets: np.ndarray,
    stim_onsets_played: np.ndarray,
    markers_onsets: np.ndarray,
    markers_detected: np.ndarray,
    tapping_detected: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Align all onsets to the first marker onset.

    Args:
        stim_onsets: Original stimulus onsets
        stim_onsets_played: Played stimulus onsets
        markers_onsets: Original marker onsets
        markers_detected: Detected marker onsets
        tapping_detected: Detected tapping onsets

    Returns:
        Dictionary containing aligned onsets
    """
    # Correct onsets relative to first marker
    stim_corrected = stim_onsets - markers_onsets[0]
    stim_played_corrected = stim_onsets_played - markers_onsets[0]
    
    # Check which onsets were played
    is_played = np.less(
        [min(np.abs(onset - stim_played_corrected)) for onset in stim_corrected],
        1.0
    )
    
    # Align tapping onsets
    tapping_corrected = tapping_detected - markers_detected[0]
    
    return {
        'stim': stim_corrected,
        'tapping': tapping_corrected,
        'is_played': is_played
    }

def verify_onsets_detection(
    onsets_detected: np.ndarray,
    onsets_ideal: np.ndarray,
    max_proximity: float,
    max_proximity_phase: List[float]
) -> Dict[str, Union[int, float, np.ndarray]]:
    """
    Verify detected onsets against ideal onsets.

    Args:
        onsets_detected: Detected onset times
        onsets_ideal: Expected onset times
        max_proximity: Maximum allowed timing error (ms)
        max_proximity_phase: Allowed phase error range

    Returns:
        Dictionary containing verification metrics
    """
    matched_onsets = compute_matched_onsets(
        onsets_ideal - onsets_ideal[0],
        onsets_detected - onsets_detected[0],
        max_proximity,
        max_proximity_phase
    )
    
    # Update: Use resp_matched instead of resp
    resp = matched_onsets['resp_matched']  # Changed from 'resp' to 'resp_matched'
    asynchrony = matched_onsets['asynchrony']
    stim_ioi = matched_onsets['stim_ioi']
    
    # Calculate verification metrics
    if np.sum(~np.isnan(asynchrony)) > 0:
        num_detected = np.sum(~np.isnan(asynchrony))
        num_missed = len(onsets_ideal) - num_detected
        max_difference = np.max(np.abs(asynchrony[~np.isnan(asynchrony)]))
    else:
        num_detected = 0
        num_missed = len(onsets_ideal)
        max_difference = -1

    # Align onsets for verification
    onsets_ideal_shifted = onsets_ideal - onsets_ideal[0] + onsets_detected[0]
    resp_shifted = resp + onsets_detected[0]

    return {
        'verify_num_detected': num_detected,
        'verify_num_missed': num_missed,
        'verify_max_difference': max_difference,
        'verify_stim_ioi': stim_ioi,
        'verify_asynchrony': asynchrony,
        'verify_stim_shifted': onsets_ideal_shifted,
        'verify_resp_shifted': resp_shifted
    }

def compute_matched_onsets(
    stim_raw: np.ndarray,
    resp_raw: np.ndarray,
    max_proximity: float,
    max_proximity_phase: List[float]
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Match stimulus and response onsets using proximity and phase criteria.

    Args:
        stim_raw: Stimulus onset times
        resp_raw: Response onset times
        max_proximity: Maximum allowed timing difference (ms)
        max_proximity_phase: Allowed phase difference range

    Returns:
        Dictionary containing matched onsets and timing information
    """
    mean_async = mean_asynchrony(stim_raw, resp_raw, max_proximity, max_proximity_phase)
    first_stim = stim_raw[0]

    # Align and match onsets
    resp, stim, is_matched, stim_ioi, resp_ioi, asynchrony = raw_onsets_to_matched_onsets(
        stim_raw - first_stim,
        resp_raw - first_stim - mean_async,
        max_proximity,
        max_proximity_phase
    )

    # Correct for mean asynchrony
    resp += mean_async
    asynchrony += mean_async

    return {
        'resp_matched': resp,
        'stim_matched': stim,
        'is_matched': is_matched,
        'stim_ioi': stim_ioi,
        'resp_ioi': resp_ioi,
        'asynchrony': asynchrony,
        'mean_async': mean_async,
        'first_stim': first_stim
    }

def mean_asynchrony(
    stim_raw: np.ndarray,
    resp_raw: np.ndarray,
    max_proximity: float,
    max_proximity_phase: List[float]
) -> float:
    """
    Calculate mean asynchrony between stimulus and response onsets.

    Args:
        stim_raw: Stimulus onset times
        resp_raw: Response onset times
        max_proximity: Maximum allowed timing difference (ms)
        max_proximity_phase: Allowed phase difference range

    Returns:
        Mean asynchrony in milliseconds
    """
    first_stim = stim_raw[0]
    
    # Align onsets relative to first stimulus
    _, _, _, _, _, asynchrony = raw_onsets_to_matched_onsets(
        stim_raw=stim_raw - first_stim,
        resp_raw=resp_raw - first_stim,
        max_proximity=max_proximity,
        max_proximity_phase=max_proximity_phase
    )

    # Calculate mean of valid asynchronies
    valid_asynchronies = asynchrony[~np.isnan(asynchrony)]
    return np.mean(valid_asynchronies) if len(valid_asynchronies) > 0 else 0

def raw_onsets_to_matched_onsets(
    stim_raw: np.ndarray,
    resp_raw: np.ndarray,
    max_proximity: float,
    max_proximity_phase: List[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Match stimulus and response onsets using greedy algorithm.
    
    Matches onsets based on both temporal proximity and phase relationship.
    Uses a greedy approach to pair the closest matching onsets first.

    Args:
        stim_raw: Stimulus onset times
        resp_raw: Response onset times
        max_proximity: Maximum allowed timing difference (ms)
        max_proximity_phase: Allowed phase difference range [-1 to 1]

    Returns:
        Tuple containing:
        - Response onsets aligned to stimulus
        - Stimulus onsets
        - Boolean mask of matched onsets
        - Stimulus inter-onset intervals
        - Response inter-onset intervals
        - Asynchronies between matched onsets
    """
    # Initialize output arrays
    N = len(stim_raw)
    stim = np.full(N, np.nan)
    resp = np.full(N, np.nan)
    is_matched = np.full(N, np.nan)
    stim_ioi = np.full(N, np.nan)
    resp_ioi = np.full(N, np.nan)
    asynchrony = np.full(N, np.nan)

    # Handle empty inputs
    if len(resp_raw) == 0 or len(stim_raw) == 0:
        return resp, stim, is_matched, stim_ioi, resp_ioi, asynchrony

    # Default phase window if not specified
    if not max_proximity_phase:
        max_proximity_phase = [-1, 1]

    # Track which onsets have been used
    stim_used = np.full(N, np.nan)
    resp_used = np.full(len(resp_raw), np.nan)

    # Find all valid onset pairs
    valid_pairs = find_valid_onset_pairs(
        stim_raw,
        resp_raw,
        max_proximity,
        max_proximity_phase
    )

    # Match onsets greedily
    step = 0
    while valid_pairs:
        # Find best remaining pair
        best_pair = get_best_onset_pair(
            valid_pairs,
            stim_used,
            resp_used
        )
        
        if not best_pair:
            break
            
        resp_idx, stim_idx = best_pair
        
        # Record match
        is_matched[stim_idx] = 0
        stim[stim_idx] = stim_raw[stim_idx]
        resp[stim_idx] = resp_raw[resp_idx]
        stim_used[stim_idx] = step
        resp_used[resp_idx] = step
        
        # Update valid pairs
        valid_pairs = [
            pair for pair in valid_pairs 
            if np.isnan(stim_used[pair[1]]) and np.isnan(resp_used[pair[0]])
        ]
        
        step += 1

    # Calculate intervals and asynchronies
    for j in range(1, N):
        if not np.isnan(stim[j]) and not np.isnan(stim[j-1]):
            stim_ioi[j] = stim[j] - stim[j-1]
        if not np.isnan(resp[j]) and not np.isnan(resp[j-1]):
            resp_ioi[j] = resp[j] - resp[j-1]
        if not np.isnan(resp[j]) and not np.isnan(stim[j]):
            asynchrony[j] = resp[j] - stim[j]
            
    # Calculate first asynchrony
    if not np.isnan(resp[0]) and not np.isnan(stim[0]):
        asynchrony[0] = resp[0] - stim[0]

    return resp, stim, is_matched, stim_ioi, resp_ioi, asynchrony

def find_valid_onset_pairs(
    stim_raw: np.ndarray,
    resp_raw: np.ndarray,
    max_proximity: float,
    max_proximity_phase: List[float]
) -> List[Tuple[int, int, float]]:
    """
    Find all valid pairs of stimulus and response onsets.

    Args:
        stim_raw: Stimulus onset times
        resp_raw: Response onset times
        max_proximity: Maximum allowed timing difference
        max_proximity_phase: Allowed phase difference range

    Returns:
        List of tuples containing (response_idx, stimulus_idx, phase)
    """
    valid_pairs = []
    
    for j, stim_time in enumerate(stim_raw):
        # Calculate intervals for phase calculation
        if j == 0:
            stim_next = stim_raw[j + 1] - stim_time
            stim_prev = stim_next
        elif j == len(stim_raw) - 1:
            stim_prev = stim_time - stim_raw[j - 1]
            stim_next = stim_prev
        else:
            stim_next = stim_raw[j + 1] - stim_time
            stim_prev = stim_time - stim_raw[j - 1]

        for k, resp_time in enumerate(resp_raw):
            # Calculate phase and temporal distance
            phase = calculate_phase(
                resp_time,
                stim_time,
                stim_next,
                stim_prev
            )
            
            distance = abs(stim_time - resp_time)
            
            # Check if pair is valid
            if (min(max_proximity_phase) < phase < max(max_proximity_phase) and 
                distance < max_proximity):
                valid_pairs.append((k, j, phase))
                
    return valid_pairs

def calculate_phase(
    resp_time: float,
    stim_time: float,
    stim_next: float,
    stim_prev: float
) -> float:
    """
    Calculate phase relationship between response and stimulus.

    Args:
        resp_time: Response onset time
        stim_time: Stimulus onset time
        stim_next: Next stimulus interval
        stim_prev: Previous stimulus interval

    Returns:
        Phase value between -1 and 1
    """
    if resp_time > stim_time:
        return (resp_time - stim_time) / stim_next
    else:
        return (stim_time - resp_time) / stim_prev

def get_best_onset_pair(
    valid_pairs: List[Tuple[int, int, float]],
    stim_used: np.ndarray,
    resp_used: np.ndarray
) -> Optional[Tuple[int, int]]:
    """
    Get the best unused pair of onsets.

    Args:
        valid_pairs: List of valid onset pairs
        stim_used: Array tracking used stimulus onsets
        resp_used: Array tracking used response onsets

    Returns:
        Tuple of (response_idx, stimulus_idx) or None if no valid pairs
    """
    # Get unused pairs
    unused_pairs = [
        (resp_idx, stim_idx, phase) 
        for resp_idx, stim_idx, phase in valid_pairs
        if np.isnan(stim_used[stim_idx]) and np.isnan(resp_used[resp_idx])
    ]
    
    if not unused_pairs:
        return None
        
    # Find pair with minimum phase
    best_pair = min(unused_pairs, key=lambda x: abs(x[2]))
    return best_pair[0], best_pair[1]

def extract_audio_signals_tapping_only(
    rec_filename: str,
    config: object
) -> Dict[str, Union[np.ndarray, int, float]]:
    """
    Extract and clean only the tapping channel from mono recording.

    Args:
        rec_filename: Path to the audio recording file
        config: Configuration parameters for the experiment

    Returns:
        Dictionary containing tapping channel and signal processing metadata

    Raises:
        FileNotFoundError: If recording file doesn't exist
        ValueError: If audio file is invalid
    """
    # Load and downsample audio
    audio_data = downsample_audio(rec_filename, config.FS0)
    samples, fs_rec, rec_downsampled, time_line = audio_data
    
    # Filter tapping channel
    tapping_channel = filter_recording(
        rec_downsampled,
        config.TAPPING_RANGE,
        config.FS0
    )

    # Normalize channel
    tapping_norm = 0.5 * tapping_channel / np.max(np.abs(tapping_channel))

    # Extract envelope
    tapping_env = extract_envelope(
        tapping_norm, 
        config.FS0, 
        config.EXTRACT_COMPRESS_FACTOR,
        config.EXTRACT_FADE_IN
    )
    
    rec_tapping = normalize_max(tapping_env)

    # Clean tapping channel
    rec_tapping_clean, _, _ = signal_cleaning_tapping(
        rec_tapping,
        config.CLEAN_LOCATION_RATIO,
        config.CLEAN_NORMALIZE_FACTOR
    )

    return {
        'fs': config.FS0,
        'rec_original': samples,
        'fs_recording': fs_rec,
        'time_line_for_sample': time_line,
        'rec_downsampled': rec_downsampled,
        'rec_tapping': rec_tapping,
        'rec_tapping_clean': rec_tapping_clean
    }
