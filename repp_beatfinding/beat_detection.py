####################################################################################################
# File:     beat_detection.py
# Purpose:  Analysis functions for beat detection tasks. Processes audio recordings to extract
#           and align tapping responses with marker onsets. Performs signal processing, statistical
#           analysis, and generates visualisations. Designed for stimuli with repeating patterns
#           where subjects tap to detect beats rather than tapping to every element.
#
# Author:   Vani Rajendran
#
####################################################################################################
import numpy as np
import gc
import matplotlib
# Set matplotlib backend to Agg for headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from repp import signal_processing as sp
from typing import Dict, Union, List, Tuple, Optional


def do_signal_processing_beat_detection(recording_filename, stim_info, config):
    """Process audio and extract/align tapping onsets for beat detection.
    
    This function handles the complete signal processing pipeline:
    1. Prepares marker onsets from stimulus information
    2. Extracts audio signals from the recording
    3. Detects onsets in the audio signals
    4. Aligns detected onsets with expected onsets

    Parameters
    ----------
    recording_filename : str
        Path to the audio recording file
    stim_info : dict
        Stimulus information dictionary containing 'markers_onsets' key
    config : class
        Configuration parameters for the experiment

    Returns
    -------
    markers_onsets : np.ndarray
        Array of marker onsets
    audio_signals : dict
        Dictionary containing extracted and cleaned audio signals
    raw_extracted_onsets : dict
        Dictionary with raw detected onsets before alignment
    aligned_onsets : dict
        Dictionary with final aligned onsets after processing
    """
    print("Preparing marker onsets...")
    markers_onsets = np.array(stim_info['markers_onsets'])
    
    print("Extracting audio signals from mono recording...")
    audio_signals = sp.extract_audio_signals(recording_filename, config)
    
    print("Extracting raw onsets from audio signals...")
    raw_extracted_onsets = sp.extract_onsets(audio_signals, config)
    
    print("Aligning onsets...")
    aligned_onsets = align_onsets_beat_detection(
        markers_onsets,
        raw_extracted_onsets,
        config.MARKERS_MATCHING_WINDOW,
        config.ONSET_MATCHING_WINDOW_PHASE
    )
    
    return markers_onsets, audio_signals, raw_extracted_onsets, aligned_onsets


def align_onsets_beat_detection(
    initial_onsets: np.ndarray,
    raw_extracted_onsets: Dict[str, np.ndarray],
    markers_matching_window: float,
    onset_matching_window_phase: list
) -> Dict[str, Union[np.ndarray, float]]:
    """Align tapping onsets with detected markers for beat detection tasks.

    Parameters
    ----------
    initial_onsets : np.ndarray
        Array of marker onsets from stimulus info
    raw_extracted_onsets : dict
        Dictionary containing extracted onsets including 'markers_detected_onsets' and
        'tapping_detected_onsets' keys
    markers_matching_window : float
        Window for matching marker onsets (ms)
    onset_matching_window_phase : list
        Phase window for matching onsets

    Returns
    -------
    dict
        Dictionary containing aligned onsets and timing information with keys:
        - 'resp_onsets_detected': Detected tapping onsets
        - 'resp_onsets_aligned': Aligned tapping onsets
        - 'num_resp_raw_onsets': Number of raw tapping onsets
        - 'markers_onsets_detected': Detected marker onsets
        - 'markers_onsets_aligned': Aligned marker onsets
        - 'markers_onsets_input': Original input marker onsets
        - Additional keys from markers_verification
    """
    markers_detected = raw_extracted_onsets['markers_detected_onsets']
    tapping_detected = raw_extracted_onsets['tapping_detected_onsets']
    markers_onsets = initial_onsets
    
    # Match raw tapping onsets against marker onsets to separate them
    # This identifies which tapping onsets are actually markers vs real taps
    if len(markers_detected) > 0 and len(tapping_detected) > 0:
        matched_onsets = compute_matched_onsets(
            markers_detected,
            tapping_detected,
            markers_matching_window,
            onset_matching_window_phase
        )
        
        # Extract matched marker onsets to identify which markers have matching taps
        # These markers indicate which tapping onsets should be filtered out
        matched_marker_onsets = matched_onsets['stim_matched']
        
        # Filter out tapping onsets that matched markers (these are marker onsets, not real taps)
        # Keep only tapping onsets that didn't match any marker
        # We identify matched taps by finding which original tapping onsets are close to matched markers
        tapping_mask = np.ones(len(tapping_detected), dtype=bool)
        
        # For each marker that was matched, find the corresponding tapping onset
        matched_marker_indices = np.where(~np.isnan(matched_marker_onsets))[0]
        for marker_idx in matched_marker_indices:
            marker_time = markers_detected[marker_idx]
            # Find tapping onsets within the matching window of this marker
            distances = np.abs(tapping_detected - marker_time)
            within_window = distances < markers_matching_window
            if np.any(within_window):
                # Mark the closest tap as matched (to be filtered out)
                closest_idx = np.argmin(distances)
                if distances[closest_idx] < markers_matching_window:
                    tapping_mask[closest_idx] = False
        
        # Keep only tapping onsets that didn't match markers
        tapping_detected_filtered = tapping_detected[tapping_mask]
    else:
        # No markers or no taps, keep all tapping onsets
        tapping_detected_filtered = tapping_detected.copy()
        matched_onsets = None
    
    # Align filtered tapping onsets to first marker
    tapping_aligned = align_to_first_marker_tapping_only(
        markers_onsets,
        markers_detected,
        tapping_detected_filtered
    )

    # Verify markers
    markers_verification = sp.verify_onsets_detection(
        markers_detected,
        markers_onsets,
        markers_matching_window,
        onset_matching_window_phase
    )

    return {
        'resp_onsets_detected': tapping_detected_filtered,
        'resp_onsets_aligned': tapping_aligned,
        'num_resp_raw_onsets': float(len(tapping_detected_filtered)),
        'markers_onsets_detected': markers_detected,
        'markers_onsets_aligned': markers_onsets - markers_onsets[0] + markers_detected[0],
        'markers_onsets_input': markers_onsets,
        **markers_verification
    }


def compute_matched_onsets(
    stim_raw: np.ndarray,
    resp_raw: np.ndarray,
    max_proximity: float,
    max_proximity_phase: List[float]
) -> Dict[str, Union[np.ndarray, float]]:
    """Match stimulus and response onsets using proximity and phase criteria.

    Parameters
    ----------
    stim_raw : np.ndarray
        Stimulus onset times
    resp_raw : np.ndarray
        Response onset times
    max_proximity : float
        Maximum allowed timing difference (ms)
    max_proximity_phase : List[float]
        Allowed phase difference range

    Returns
    -------
    dict
        Dictionary containing matched onsets and timing information with keys:
        - 'resp_matched': Matched response onsets
        - 'stim_matched': Matched stimulus onsets
        - 'is_matched': Boolean mask of matched onsets
        - 'stim_ioi': Stimulus inter-onset intervals
        - 'resp_ioi': Response inter-onset intervals
        - 'asynchrony': Asynchronies between matched onsets
        - 'mean_async': Mean asynchrony
        - 'first_stim': First stimulus time
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
    """Calculate mean asynchrony between stimulus and response onsets.

    Parameters
    ----------
    stim_raw : np.ndarray
        Stimulus onset times
    resp_raw : np.ndarray
        Response onset times
    max_proximity : float
        Maximum allowed timing difference (ms)
    max_proximity_phase : List[float]
        Allowed phase difference range

    Returns
    -------
    float
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
    """Match stimulus and response onsets using greedy algorithm.
    
    Matches onsets based on both temporal proximity and phase relationship.
    Uses a greedy approach to pair the closest matching onsets first.

    Parameters
    ----------
    stim_raw : np.ndarray
        Stimulus onset times
    resp_raw : np.ndarray
        Response onset times
    max_proximity : float
        Maximum allowed timing difference (ms)
    max_proximity_phase : List[float]
        Allowed phase difference range [-1 to 1]

    Returns
    -------
    Tuple
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
    """Find all valid pairs of stimulus and response onsets.

    Parameters
    ----------
    stim_raw : np.ndarray
        Stimulus onset times
    resp_raw : np.ndarray
        Response onset times
    max_proximity : float
        Maximum allowed timing difference
    max_proximity_phase : List[float]
        Allowed phase difference range

    Returns
    -------
    List[Tuple[int, int, float]]
        List of tuples containing (response_idx, stimulus_idx, phase)
    """
    valid_pairs = []
    
    for j, stim_time in enumerate(stim_raw):
        # Calculate intervals for phase calculation
        if len(stim_raw) == 1:
            # Single stimulus - use default interval
            stim_next = stim_time if stim_time > 0 else 1.0
            stim_prev = stim_next
        elif j == 0:
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
    """Calculate phase relationship between response and stimulus.

    Parameters
    ----------
    resp_time : float
        Response onset time
    stim_time : float
        Stimulus onset time
    stim_next : float
        Next stimulus interval
    stim_prev : float
        Previous stimulus interval

    Returns
    -------
    float
        Phase value between -1 and 1
    """
    if resp_time > stim_time:
        return (resp_time - stim_time) / stim_next if stim_next > 0 else 0
    else:
        return (stim_time - resp_time) / stim_prev if stim_prev > 0 else 0


def get_best_onset_pair(
    valid_pairs: List[Tuple[int, int, float]],
    stim_used: np.ndarray,
    resp_used: np.ndarray
) -> Optional[Tuple[int, int]]:
    """Get the best unused pair of onsets.

    Parameters
    ----------
    valid_pairs : List[Tuple[int, int, float]]
        List of valid onset pairs
    stim_used : np.ndarray
        Array tracking used stimulus onsets
    resp_used : np.ndarray
        Array tracking used response onsets

    Returns
    -------
    Optional[Tuple[int, int]]
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


def align_to_first_marker_tapping_only(
    markers_onsets: np.ndarray,
    markers_detected: np.ndarray,
    tapping_detected: np.ndarray
) -> np.ndarray:
    """Align tapping onsets to the first detected marker onset.

    Parameters
    ----------
    markers_onsets : np.ndarray
        Original marker onsets (not used in calculation, kept for consistency)
    markers_detected : np.ndarray
        Detected marker onsets from audio signal
    tapping_detected : np.ndarray
        Detected tapping onsets from audio signal

    Returns
    -------
    np.ndarray
        Array of tapping onsets aligned to first detected marker
    """
    return tapping_detected - markers_detected[0]


def do_beat_detection_analysis(recording_filename, title_plot, output_plot, stim_info=None, config=None, dpi=300):
    """Perform complete beat detection analysis including visualisation.
    
    This function provides a convenient interface for beat detection analysis.
    It runs the complete analysis pipeline:
    1. Processes the audio recording
    2. Extracts and aligns tapping responses
    3. Performs statistical analysis
    4. Generates visualisation plots
    
    Parameters
    ----------
    recording_filename : str
        Path to the audio recording file
    title_plot : str
        Title for the generated plots
    output_plot : str
        Path where plot should be saved
    stim_info : dict, optional
        Stimulus information dictionary containing 'markers_onsets' key
    config : class, optional
        Configuration parameters for the experiment. If not provided, will attempt to
        import from repp.config.sms_tapping
    dpi : int, optional
        Resolution for saved plot (default: 300)
    
    Returns
    -------
    output : dict
        Raw signal processing results
    analysis : dict
        Statistical analysis of tapping performance
    is_failed : dict
        Quality control checks and failure status
    
    Raises
    ------
    ValueError
        If config is not provided and cannot be imported
    """
    if config is None:
        try:
            from repp.config import sms_tapping as config
        except ImportError:
            raise ValueError(
                "config parameter is required. Please provide the experiment configuration "
                "or ensure repp.config.sms_tapping is available."
            )
        
    # Perform signal processing
    _, audio_signals, _, aligned_onsets = do_signal_processing_beat_detection(
        recording_filename, stim_info, config
    )
    
    # Perform statistical analysis
    print("Tapping analysis...")
    output, analysis, is_failed = do_stast_beat_detection(aligned_onsets, config)
    
    # Generate and save visualisation
    print("Analysing results...")
    fig = do_plot_beat_detection(title_plot, audio_signals, aligned_onsets, analysis, is_failed, config)
    save_local(fig, output_plot, dpi)
    print("Plot saved")
    del fig
    gc.collect()
    
    return output, analysis, is_failed


def reformat_output(onsets_aligned):
    """Reformat aligned onsets into standardised output structure.
    
    Parameters
    ----------
    onsets_aligned : dict
        Dictionary containing aligned onset data from signal processing
    
    Returns
    -------
    dict
        Standardised output dictionary
    """
    return {
        # Response timing information
        'resp_onsets_detected': np.round(onsets_aligned['resp_onsets_detected'], 2).tolist(),
        'resp_onsets_aligned': np.round(onsets_aligned['resp_onsets_aligned'], 2).tolist(),
        'num_resp_raw_onsets': onsets_aligned.get('num_resp_raw_onsets', 0),
        
        # Marker signal information
        'markers_onsets_input': np.round(onsets_aligned['markers_onsets_input'], 2).tolist(),
        'markers_onsets_detected': np.round(onsets_aligned['markers_onsets_detected'], 2).tolist(),
        'markers_onsets_aligned': np.round(onsets_aligned['markers_onsets_aligned'], 2).tolist(),
    }


def do_stast_beat_detection(onsets_aligned, config):
    """Calculate statistics for beat detection experiments.
    
    This method computes performance metrics for:
    - Marker detection quality
    - Tapping performance (inter-tap intervals and timing statistics)
    - No stimulus-related comparisons

    Parameters
    ----------
    onsets_aligned : dict
        Dictionary containing aligned onset data from signal processing
    config : class
        Configuration parameters for analysis thresholds and criteria

    Returns
    -------
    output : dict
        Raw signal processing results in standardised format
    analysis : dict
        Statistical analysis results including:
        - Marker detection performance
        - Tapping timing statistics (inter-tap intervals)
    is_failed : dict
        Quality control assessment including:
        - Overall pass/fail status
        - Specific reason for failure if applicable
    """
    # Reformat output into standardised structure
    output = reformat_output(onsets_aligned)
    
    # Get tapping onsets
    tapping_onsets_aligned = onsets_aligned.get('resp_onsets_aligned', np.array([]))
    tapping_onsets_detected = onsets_aligned.get('resp_onsets_detected', np.array([]))
    num_resp_raw = onsets_aligned.get('num_resp_raw_onsets', 0)
    
    # Calculate inter-tap intervals (ITIs/IOIs)
    if len(tapping_onsets_aligned) > 1:
        tapping_iois = np.diff(tapping_onsets_aligned)
        mean_ioi = np.mean(tapping_iois)
        sd_ioi = np.std(tapping_iois)
        median_ioi = np.median(tapping_iois)
        cv_ioi = (sd_ioi / mean_ioi * 100) if mean_ioi > 0 else 0
        min_ioi = np.min(tapping_iois)
        max_ioi = np.max(tapping_iois)
        ioi_range = max_ioi - min_ioi
    else:
        tapping_iois = np.array([])
        mean_ioi = None
        sd_ioi = None
        median_ioi = None
        cv_ioi = None
        min_ioi = None
        max_ioi = None
        ioi_range = None
    
    # Assess marker quality
    markers_status, markers_ok = "Good", True
    if (abs(onsets_aligned.get('verify_max_difference', 0)) >= config.MARKERS_MAX_ERROR or 
        onsets_aligned.get('verify_num_missed', 0) > 0):
        markers_status, markers_ok = "Bad", False
    

    # Compile analysis results
    analysis = {
        # Marker performance metrics
        'num_markers_onsets': len(onsets_aligned.get('markers_onsets_input', [])),
        'num_markers_detected': onsets_aligned.get('verify_num_detected', 0),
        'num_markers_missed': onsets_aligned.get('verify_num_missed', 0),
        'markers_max_difference': onsets_aligned.get('verify_max_difference', 0),
        'markers_status': markers_status,
        'markers_ok': markers_ok,
        
        # Tapping performance metrics
        'num_resp_raw_all': num_resp_raw,
        'num_resp_aligned_all': len(tapping_onsets_aligned),
        'num_taps_detected': len(tapping_onsets_detected),
        
        # Inter-tap interval statistics
        'tapping_onsets_aligned': tapping_onsets_aligned,
        'tapping_onsets_detected': tapping_onsets_detected,
        'tapping_iois': tapping_iois,
        'tapping_mean_ioi': mean_ioi,
        'tapping_sd_ioi': sd_ioi,
        'tapping_median_ioi': median_ioi,
        'tapping_cv_ioi': cv_ioi,
        'tapping_min_ioi': min_ioi,
        'tapping_max_ioi': max_ioi,
        'tapping_ioi_range': ioi_range,
        'num_iois': len(tapping_iois)
    }
    
    # Assess if experiment meets quality criteria
    is_failed = failing_criteria_beat_detection(analysis, config)
    
    return output, analysis, is_failed


def save_local(fig, output_filename, dpi):
    """Save the generated figure to file.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save
    output_filename : str
        Path where to save the figure
    dpi : int
        Resolution for the saved figure
    """
    if output_filename == '':
        fig.show()
    else:
        fig.savefig(
            output_filename, 
            format="png",
            dpi=dpi,
            facecolor='w', 
            edgecolor='w'
        )
        fig.clf()


def do_plot_beat_detection(title_plot, audio_signals, aligned_onsets, analysis, is_failed, config):
    """Create visualisation plots for beat detection analysis.
    
    This function generates plots showing:
    1. Raw audio signal with detected taps and markers
    2. Marker detection analysis
    3. Inter-tap interval (IOI) distribution
    4. Tap timing consistency
    5. Summary statistics
    
    Parameters
    ----------
    title_plot : str
        Title for the overall plot
    audio_signals : dict
        Dictionary containing processed audio signals
    aligned_onsets : dict
        Dictionary containing aligned onset data with keys:
        - 'resp_onsets_detected': Detected tapping onsets (ms)
        - 'resp_onsets_aligned': Aligned tapping onsets (ms)
        - 'markers_onsets_detected': Detected marker onsets (ms)
        - 'markers_onsets_aligned': Aligned marker onsets (ms)
    analysis : dict
        Dictionary containing analysis results from do_stast_beat_detection
    is_failed : dict
        Dictionary with 'failed' (bool) and 'reason' (str) keys
    config : class
        Configuration parameters
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    print(f"Creating plot: {title_plot}")
    
    # Extract data with error handling
    try:
        tt = audio_signals.get('time_line_for_sample', np.array([]))
        rec_downsampled = audio_signals.get('rec_downsampled', np.array([]))
        R_clean = audio_signals.get('rec_tapping_clean', np.array([]))
        
        # Get tapping onsets (use detected onsets for plotting timeline)
        tap_onsets = np.array(aligned_onsets.get('resp_onsets_detected', []))
        tapping_onsets_aligned = np.array(aligned_onsets.get('resp_onsets_aligned', []))
        
        # Get marker onsets
        markers_detected = np.array(aligned_onsets.get('markers_onsets_detected', []))
        markers_aligned = np.array(aligned_onsets.get('markers_onsets_aligned', []))
        
        # Get IOI data from analysis
        tapping_iois = np.array(analysis.get('tapping_iois', []))
        
        print(f"Data extracted - Time points: {len(tt)}, Taps: {len(tap_onsets)}, Markers: {len(markers_detected)}, IOIs: {len(tapping_iois)}")
        
    except Exception as e:
        print(f"Error extracting data for plotting: {e}")
        # Create empty arrays as fallback
        tt = np.array([])
        rec_downsampled = np.array([])
        R_clean = np.array([])
        tap_onsets = np.array([])
        tapping_onsets_aligned = np.array([])
        markers_detected = np.array([])
        markers_aligned = np.array([])
        tapping_iois = np.array([])
    
    # Check if we have any data to plot
    if len(tt) == 0:
        print("Warning: No time data available for plotting")
        tt = np.array([0, 1])  # Create minimal time array
        rec_downsampled = np.array([0, 0])
        R_clean = np.array([0, 0])
    
    # Clear any existing plots
    plt.clf()
    
    # Set style
    plt.style.use('default')
    colours = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']
    
    # Create a 2x3 grid
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Raw audio signal with detected taps and markers (top row, spans 2 columns)
    plt.subplot(2, 3, (1, 2))
    if len(rec_downsampled) > 0:
        plt.plot(tt, rec_downsampled, color=colours[0], alpha=0.7, linewidth=0.8, label='Raw audio')
    if len(R_clean) > 0:
        plt.plot(tt, R_clean, color=colours[1], linewidth=1.2, label='Filtered tapping')
    
    # Add detected tap markers (convert from ms to seconds)
    if len(tap_onsets) > 0 and len(R_clean) > 0:
        mmx = np.max(R_clean) * 0.8
        plt.scatter(tap_onsets / 1000.0, [mmx] * len(tap_onsets), 
                   color=colours[2], s=100, marker='o', zorder=5, label='Detected taps')
    
    # Add detected marker indicators (if available)
    if len(markers_detected) > 0 and len(R_clean) > 0:
        mmx_markers = np.max(R_clean) * 0.6
        plt.scatter(markers_detected / 1000.0, [mmx_markers] * len(markers_detected), 
                   color=colours[4], s=80, marker='s', zorder=5, label='Detected markers')
    
    plt.title(f'{title_plot}: Beat Detection Analysis', fontsize=16, fontweight='bold')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Marker detection analysis (top row, column 3)
    plt.subplot(2, 3, 3)
    markers_input = np.array(aligned_onsets.get('markers_onsets_input', []))
    if len(markers_detected) > 0 and len(markers_input) > 0:
        # Calculate errors between detected and expected (input) markers
        # First, align input markers to detected markers by matching the first marker
        if len(markers_input) > 0 and len(markers_detected) > 0:
            # Calculate offset from first marker
            offset = markers_detected[0] - markers_input[0]
            markers_input_aligned = markers_input + offset
            
            # Match detected markers to input markers and calculate errors
            min_len = min(len(markers_detected), len(markers_input_aligned))
            marker_errors = markers_detected[:min_len] - markers_input_aligned[:min_len]
            marker_indices = np.arange(len(marker_errors))
            
            if len(marker_errors) > 0:
                plt.plot(marker_indices, marker_errors, 'o-', color=colours[0], linewidth=1.5, markersize=6)
                plt.axhline(0, color='black', linestyle='--', alpha=0.5)
                plt.axhline(config.MARKERS_MAX_ERROR, color=colours[3], linestyle=':', alpha=0.7, 
                           label=f'Max error: {config.MARKERS_MAX_ERROR}ms')
                plt.axhline(-config.MARKERS_MAX_ERROR, color=colours[3], linestyle=':', alpha=0.7)
                plt.title('Marker Detection Error', fontsize=14, fontweight='bold')
                plt.xlabel('Marker number')
                plt.ylabel('Error (ms)')
                plt.legend()
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No marker error data', ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Marker Detection Error', fontsize=14, fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'Insufficient marker data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Marker Detection Error', fontsize=14, fontweight='bold')
    elif len(markers_detected) > 0:
        # Just show detected markers count if we don't have input markers
        plt.text(0.5, 0.5, f'Detected: {len(markers_detected)} markers\n(No input markers for comparison)', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
        plt.title('Marker Detection', fontsize=14, fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'No marker data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Marker Detection Error', fontsize=14, fontweight='bold')
    
    # 3. Inter-tap interval distribution (second row, left)
    plt.subplot(2, 3, 4)
    if len(tapping_iois) > 0:
        # IOI histogram
        bins = min(20, max(5, len(tapping_iois) // 2))
        plt.hist(tapping_iois, bins=bins, alpha=0.7, color=colours[0], edgecolor='black')
        if analysis.get('tapping_median_ioi') is not None:
            plt.axvline(analysis['tapping_median_ioi'], color=colours[2], linestyle='--', linewidth=2, 
                       label=f'Median: {analysis["tapping_median_ioi"]:.1f}ms')
        if analysis.get('tapping_mean_ioi') is not None:
            plt.axvline(analysis['tapping_mean_ioi'], color=colours[3], linestyle=':', linewidth=2, 
                       label=f'Mean: {analysis["tapping_mean_ioi"]:.1f}ms')
    plt.title('Inter-tap Interval Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Interval (ms)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Tap timing consistency (second row, middle)
    plt.subplot(2, 3, 5)
    if len(tapping_iois) > 1:
        # Plot IOI over time
        tap_indices = np.arange(1, len(tapping_iois) + 1)
        plt.plot(tap_indices, tapping_iois, 'o-', color=colours[0], linewidth=1.5, markersize=6)
        if analysis.get('tapping_mean_ioi') is not None:
            plt.axhline(analysis['tapping_mean_ioi'], color=colours[2], linestyle='--', alpha=0.8,
                       label=f'Mean: {analysis["tapping_mean_ioi"]:.1f}ms')
            if analysis.get('tapping_sd_ioi') is not None:
                plt.axhline(analysis['tapping_mean_ioi'] + analysis['tapping_sd_ioi'], 
                           color=colours[3], linestyle=':', alpha=0.7)
                plt.axhline(analysis['tapping_mean_ioi'] - analysis['tapping_sd_ioi'], 
                           color=colours[3], linestyle=':', alpha=0.7)
    plt.title('Inter-tap Intervals Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Tap number')
    plt.ylabel('Interval (ms)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Summary statistics (second row, right)
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Create summary text
    summary_lines = [
        "Summary Statistics",
        "=" * 20,
        "",
        f"Status: {'FAILED' if is_failed.get('failed', False) else 'PASSED'}",
        f"Reason: {is_failed.get('reason', 'N/A')}",
        "",
        "Markers:",
        f"  Detected: {analysis.get('num_markers_detected', 0)}/{analysis.get('num_markers_onsets', 0)}",
        f"  Status: {analysis.get('markers_status', 'N/A')}",
        f"  Max error: {analysis.get('markers_max_difference', 0):.1f}ms",
        "",
        "Tapping:",
        f"  Detected taps: {analysis.get('num_taps_detected', 0)}",
        f"  Aligned taps: {analysis.get('num_resp_aligned_all', 0)}",
    ]
    
    if analysis.get('tapping_mean_ioi') is not None:
        summary_lines.extend([
            "",
            "IOI Statistics:",
            f"  Mean: {analysis['tapping_mean_ioi']:.1f}ms",
            f"  Median: {analysis.get('tapping_median_ioi', 0):.1f}ms",
            f"  SD: {analysis.get('tapping_sd_ioi', 0):.1f}ms",
            f"  CV: {analysis.get('tapping_cv_ioi', 0):.1f}%",
            f"  Range: {analysis.get('tapping_min_ioi', 0):.1f}-{analysis.get('tapping_max_ioi', 0):.1f}ms",
        ])
    
    summary_text = "\n".join(summary_lines)
    plt.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    
    return fig


def failing_criteria_beat_detection(analysis, config):
    """Evaluate if the experiment meets quality criteria.
    
    Checks multiple criteria including:
    1. Marker detection completeness
    2. Marker timing accuracy
    3. Minimum and maximum number of taps
    
    Parameters
    ----------
    analysis : dict
        Dictionary containing analysis results
    config : class
        Configuration parameters defining quality thresholds

    Returns
    -------
    is_failed : dict
        Dictionary containing:
        - failed: bool, True if any criteria failed
        - reason: str, description of the first failed criterion
    """
    # Check all quality criteria
    all_markers_are_detected = analysis['num_markers_onsets'] == analysis['num_markers_detected']
    markers_error_is_low = analysis['markers_max_difference'] < config.MARKERS_MAX_ERROR
    
    # For beat detection, check raw tap count against thresholds
    num_taps = analysis.get('num_resp_raw_all', 0)
    min_num_taps_is_ok = num_taps >= config.MIN_RAW_TAPS
    max_num_taps_is_ok = num_taps <= config.MAX_RAW_TAPS
    
    # Combine all criteria
    failed = not (all_markers_are_detected and 
                 markers_error_is_low and 
                 min_num_taps_is_ok and 
                 max_num_taps_is_ok)
    
    # Define possible failure reasons
    options = [
        all_markers_are_detected,
        markers_error_is_low,
        min_num_taps_is_ok,
        max_num_taps_is_ok
    ]
    reasons = [
        "Not all markers detected",
        "Markers error too large",
        "Too few detected taps",
        "Too many detected taps"
    ]
    
    # Find first failed criterion
    if False in options:
        index = options.index(False)
        reason = reasons[index]
    else:
        reason = "All good"
        
    is_failed = {'failed': failed, 'reason': reason}
    return is_failed
