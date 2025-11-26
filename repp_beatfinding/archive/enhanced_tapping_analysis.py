# THIS CODE HAS NOT BEEN TESTED PROPERLY AND SHOULD NOT BE USED

"""
Enhanced tapping analysis module.
This module provides comprehensive tapping analysis with detailed visualisations.

NEW FEATURE: Precise Marker Filtering and Alignment
==================================================

This module now supports precise filtering of audio markers using the actual marker
timing information from the stimulus. 

Usage:
------
The enhanced analysis requires marker timing information in the `stim_info` parameter:

```python
# When calling enhanced_tapping_analysis
audio_signals, extracted_onsets, analysis = enhanced_tapping_analysis(
    recording_filename, 
    title_plot, 
    output_plot, 
    stim_info=stim_info  # Must contain markers_onsets data
)
```

The `stim_info` must contain:
- `markers_onsets`: Array of marker onset times in milliseconds (REQUIRED)
- `stim_duration`: Total stimulus duration in seconds

The analysis results now include:
- `alignment_info`: Details about tap alignment relative to first marker
- `markers_removed`: Number of marker-related taps that were filtered out
- Enhanced summary statistics in plots

"""

import numpy as np
import matplotlib
# Set matplotlib backend to Agg for headless environments (Docker containers)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os
import gc
from typing import Dict, List, Tuple, Optional, Union
try:
    from scipy import signal
except ImportError:
    signal = None


def save_local(fig, output_filename, dpi):
    """Save or display the generated figure.
    
    This function handles both saving to file and displaying the figure,
    following the same pattern as the REPPAnalysis.save_local method.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save/display
    output_filename : str
        Path where to save the figure. If empty, displays figure instead
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


def create_enhanced_tapping_plots(audio_signals, extracted_onsets, analysis, title_plot, output_plot, dpi=300, stim_info=None):
    """
    Create comprehensive visualisations for tapping analysis.
    
    This function generates multiple plots to help understand tapping patterns:
    1. Raw audio signal with detected taps and markers (spans 2 columns)
    2. Marker detection analysis
    3. Inter-onset interval (IOI) analysis
    4. Tap timing consistency
    5. Summary statistics
    
    Parameters
    ----------
    audio_signals : dict
        Dictionary containing processed audio signals
    extracted_onsets : dict
        Dictionary containing detected tap onsets
    analysis : dict
        Dictionary containing analysis results
    title_plot : str
        Title for the overall plot
    output_plot : str
        Path where plot should be saved
    dpi : int, optional
        Resolution for saved plot (default: 300)
    stim_info : dict, optional
        Stimulus information containing markers_onsets and other metadata
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    print(f"Starting plot creation for: {title_plot}")
    print(f"Output path: {output_plot}")
    
    # Extract data with error handling
    try:
        tt = audio_signals.get('time_line_for_sample', np.array([]))
        rec_downsampled = audio_signals.get('rec_downsampled', np.array([]))
        R_clean = audio_signals.get('rec_tapping_clean', np.array([]))
        tap_onsets = np.array(extracted_onsets.get('tapping_detected_onsets', []))
        iois = np.array(analysis.get('resp_ioi_detected', []))
        
        # Get alignment data if available
        alignment = analysis.get('alignment', {})
        tapping_onsets_aligned = alignment.get('tapping_onsets_aligned', np.array([]))
        markers_onsets_aligned = alignment.get('markers_onsets_aligned', np.array([]))
        tapping_ioi_aligned = alignment.get('tapping_ioi', np.array([]))
        markers_ioi = alignment.get('markers_ioi', np.array([]))
        verify = alignment.get('verify', {})
        
        # Get detected markers from signal processing (if available)
        detected_markers = np.array([])
        if 'markers_detected_onsets' in audio_signals:
            detected_markers = audio_signals['markers_detected_onsets']
        elif 'rec_markers_clean' in audio_signals:
            # Try to detect markers from the markers channel
            from repp import signal_processing as sp
            try:
                markers_extracted = sp.detect_onsets(
                    audio_signals['rec_markers_clean'],
                    0.1,  # threshold
                    50,   # first window
                    100,  # second window
                    audio_signals.get('fs', 22050)
                )
                detected_markers = markers_extracted
            except:
                pass
        
        # For plotting, we need the original tap onsets (not aligned) to match the audio timeline
        # The aligned onsets are used for analysis, but plotting should use original timing
        original_tap_onsets = tap_onsets
        if len(tapping_onsets_aligned) > 0 and 'alignment_offset' in alignment:
            # Convert aligned taps back to original timing for plotting
            alignment_offset = alignment.get('alignment_offset', 0)
            original_tap_onsets = tapping_onsets_aligned + alignment_offset
        
        print(f"Data extracted - Time points: {len(tt)}, Taps: {len(tap_onsets)}, IOIs: {len(iois)}")
        print(f"Alignment data - Aligned taps: {len(tapping_onsets_aligned)}, Markers: {len(markers_onsets_aligned)}")
        print(f"Detected markers: {len(detected_markers)}")
        print(f"Using original tap onsets for plotting: {len(original_tap_onsets)}")
        
    except Exception as e:
        print(f"Error extracting data for plotting: {e}")
        # Create empty arrays as fallback
        tt = np.array([])
        rec_downsampled = np.array([])
        R_clean = np.array([])
        tap_onsets = np.array([])
        original_tap_onsets = np.array([])
        iois = np.array([])
        tapping_onsets_aligned = np.array([])
        markers_onsets_aligned = np.array([])
        tapping_ioi_aligned = np.array([])
        markers_ioi = np.array([])
        verify = {}
        detected_markers = np.array([])
    
    # Check if we have any data to plot
    if len(tt) == 0:
        print("Warning: No time data available for plotting")
        tt = np.array([0, 1])  # Create minimal time array
        rec_downsampled = np.array([0, 0])
        R_clean = np.array([0, 0])
    
    # Determine stimulus type and target tempo
    stimulus_type = "unknown"
    target_ioi = None
    target_bpm = None
    is_music_stimulus = False
    
    # Get IOI data for target calculation
    iois = np.array(analysis.get('resp_ioi_detected', []))
    
    # Use title-based detection for stimulus type
    if 'iso_800ms' in title_plot:
        stimulus_type = "isochronous"
        target_ioi = 800
        target_bpm = 75
    elif 'iso_600ms' in title_plot:
        stimulus_type = "isochronous"
        target_ioi = 600
        target_bpm = 100
    elif 'track' in title_plot or 'music' in title_plot:
        stimulus_type = "music"
        is_music_stimulus = True
        if len(iois) > 0:
            target_ioi = np.median(iois)
            target_bpm = 60000 / target_ioi if target_ioi > 0 else None
    
    print(f"Stimulus type: {stimulus_type}, Target IOI: {target_ioi}ms, Target BPM: {target_bpm}")
    
    print("Creating matplotlib figure...")
    
    # Clear any existing plots
    plt.clf()
    
    # Set style
    plt.style.use('default')
    colours = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']
    
    print("Creating subplots...")
    
    # Create a 2x3 grid to accommodate the wider first plot and 5 total plots
    # 1. Raw audio signal with detected taps and markers (top row, spans 2 columns)
    plt.subplot(2, 3, (1, 2))  # Span columns 1-2
    if len(rec_downsampled) > 0:
        plt.plot(tt, rec_downsampled, color=colours[0], alpha=0.7, linewidth=0.8, label='Raw audio')
    if len(R_clean) > 0:
        plt.plot(tt, R_clean, color=colours[1], linewidth=1.2, label='Filtered tapping')
    
    # Add detected tap markers
    if len(original_tap_onsets) > 0 and len(R_clean) > 0:
        mmx = np.max(R_clean) * 0.8
        plt.scatter(original_tap_onsets / 1000.0, [mmx] * len(original_tap_onsets), 
                   color=colours[2], s=100, marker='o', zorder=5, label='Detected taps')
    
    # Add detected marker indicators (if available)
    if len(detected_markers) > 0 and len(R_clean) > 0:
        mmx_markers = np.max(R_clean) * 0.6
        plt.scatter(detected_markers / 1000.0, [mmx_markers] * len(detected_markers), 
                   color=colours[4], s=80, marker='s', zorder=5, label='Detected markers')
    
    # Add expected marker indicators (if available)
    if stim_info and 'markers_onsets' in stim_info and len(stim_info['markers_onsets']) > 0:
        expected_markers = np.array(stim_info['markers_onsets'])
        
        # Align expected markers with the actual audio timeline
        # Use the first detected marker as a reference point
        if len(detected_markers) > 0:
            # Calculate the offset between expected first marker (0ms) and actual first detected marker
            first_detected_marker = detected_markers[0]
            expected_first_marker = expected_markers[0]  # Should be 0ms
            timeline_offset = first_detected_marker - expected_first_marker
            
            # Apply the timeline offset to position expected markers correctly
            expected_markers_aligned = expected_markers + timeline_offset
            print(f"Aligning expected markers: timeline offset = {timeline_offset:.1f}ms")
            print(f"Expected markers aligned: {expected_markers_aligned}")
        else:
            # If no detected markers, use markers as-is but convert to seconds
            expected_markers_aligned = expected_markers
        
        mmx_expected = np.max(R_clean) * 0.4 if len(R_clean) > 0 else 0.5
        plt.scatter(expected_markers_aligned / 1000.0, [mmx_expected] * len(expected_markers_aligned), 
                   color='green', s=60, marker='^', zorder=5, label='Expected markers')
    
    # Add stimulus type to title
    title_with_type = f'{title_plot}: {stimulus_type.title()} Tapping Analysis'
    plt.title(title_with_type, fontsize=16, fontweight='bold')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Marker detection analysis (top row, column 3)
    plt.subplot(2, 3, 3)
    plot_markers_detection_enhanced(tt, audio_signals, detected_markers, analysis, colours)
    
    # 3. Inter-onset interval analysis (second row, left)
    plt.subplot(2, 3, 4)
    # Use aligned IOI data if available, otherwise fall back to original
    plot_iois = tapping_ioi_aligned if len(tapping_ioi_aligned) > 0 else iois
    
    if len(plot_iois) > 0:
        # IOI histogram
        plt.hist(plot_iois, bins=min(20, len(plot_iois)//2), alpha=0.7, color=colours[0], edgecolor='black')
        plt.axvline(np.median(plot_iois), color=colours[2], linestyle='--', linewidth=2, 
                   label=f'Median: {np.median(plot_iois):.1f}ms')
        plt.axvline(np.mean(plot_iois), color=colours[3], linestyle=':', linewidth=2, 
                   label=f'Mean: {np.mean(plot_iois):.1f}ms')
        
        # Add target tempo reference if available
        if target_ioi is not None:
            plt.axvline(target_ioi, color='green', linestyle='-', linewidth=2, alpha=0.7, 
                       label=f'Target: {target_ioi}ms')
        
        # Remove marker IOI reference - no longer needed
    
    plt.title('Inter-onset Interval Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Interval (ms)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Tap timing consistency (second row, middle)
    plt.subplot(2, 3, 5)
    if len(plot_iois) > 1:
        # Calculate local variability (rolling standard deviation)
        window_size = min(5, len(plot_iois))
        local_std = []
        for i in range(len(plot_iois)):
            start = max(0, i - window_size // 2)
            end = min(len(plot_iois), i + window_size // 2 + 1)
            local_std.append(np.std(plot_iois[start:end]))
        
        tap_indices = np.arange(1, len(plot_iois) + 1)
        plt.plot(tap_indices, local_std, 'o-', color=colours[0], linewidth=1.5, markersize=6)
        plt.axhline(np.std(plot_iois), color=colours[2], linestyle='--', alpha=0.8,
                   label=f'Overall SD: {np.std(plot_iois):.1f}ms')
    
    plt.title('Local Timing Variability', fontsize=14, fontweight='bold')
    plt.xlabel('Tap number')
    plt.ylabel('Local SD (ms)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Summary statistics (second row, right)
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Create summary text
    duration = max(tt) if len(tt) > 0 else 0.0
    summary_text = f"""
    TAPPING SUMMARY
    
    Stimulus: {stimulus_type.title()}
    Total taps: {len(original_tap_onsets)}
    Duration: {duration:.1f}s
    
    Timing Statistics:
    • Median IOI: {analysis.get('median_ioi', 0):.1f}ms
    • Mean IOI: {np.mean(iois) if len(iois) > 0 else 0:.1f}ms
    • IOI SD: {np.std(iois) if len(iois) > 0 else 0:.1f}ms
    • CV: {(np.std(iois)/np.mean(iois)*100) if len(iois) > 0 and np.mean(iois) > 0 else 0:.1f}%
    • Overall BPM: {analysis.get('bpm', 0):.1f}
    """
    
    # MARKER DETECTION SECTION (enhanced with verification)
    summary_text += f"""
    Marker Detection:
    """
    
    # Get detected markers from audio_signals
    detected_markers = audio_signals.get('markers_detected_onsets', [])
    expected_markers = len(stim_info.get('markers_onsets', [])) if stim_info else 0
    
    # Get enhanced marker verification data if available
    marker_verification = analysis.get('marker_verification', {})
    
    if marker_verification and 'verification_status' in marker_verification:
        # Use enhanced verification data
        status = marker_verification['verification_status']
        detection_rate = marker_verification.get('detection_rate', 0)
        quality_score = marker_verification.get('quality_score', 0)
        mean_error = marker_verification.get('mean_timing_error', -1)
        timing_std = marker_verification.get('timing_std', -1)
        
        summary_text += f"• Status: {status}\n"
        summary_text += f"• Detection: {detection_rate:.0f}% ({marker_verification.get('num_detected', 0)}/{marker_verification.get('num_expected', 0)})\n"
        summary_text += f"• Quality Score: {quality_score:.0f}/100\n"
        
        if mean_error >= 0:
            summary_text += f"• Timing Error: {mean_error:.1f}±{timing_std:.1f}ms\n"
        
        # Add verification reason if not passed
        if not marker_verification.get('verification_passed', True):
            summary_text += f"• Issue: {marker_verification.get('verification_reason', 'Unknown')}\n"
    else:
        # Fallback to basic detection statistics
        if expected_markers > 0:
            percent_markers_detected = (len(detected_markers) / expected_markers) * 100
            summary_text += f"• {percent_markers_detected:.0f}% markers detected ({len(detected_markers)} out of {expected_markers})\n"
        else:
            summary_text += f"• {len(detected_markers)} markers detected\n"
    
    # QUALITY ASSESSMENT SECTION
    summary_text += f"""
    Quality Assessment:
    • Failed: {analysis.get('failed', False)}
    • Reason: {analysis.get('reason', 'N/A')}
    """
    
    # Add marker warning if present
    if 'marker_warning' in analysis:
        summary_text += f"• Marker Warning: {analysis['marker_warning']}\n"
    
    # Add marker criteria information if available
    marker_criteria = analysis.get('marker_criteria', {})
    if marker_criteria:
        summary_text += f"""
    Marker Criteria:
    • All markers detected: {marker_criteria.get('all_markers_detected', 'N/A')} ({marker_criteria.get('num_detected_markers', 0)}/{marker_criteria.get('num_expected_markers', 0)})
    • Timing error acceptable: {marker_criteria.get('markers_error_low', 'N/A')} (max: {marker_criteria.get('max_timing_error', -1):.1f}ms, threshold: {marker_criteria.get('markers_max_error_threshold', 0)}ms)
    """
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Adjust layout
    print("Adjusting layout...")
    
    # Get the current figure
    fig = plt.gcf()
    fig.set_size_inches(24, 12)  # Adjusted for 2x3 layout
    
    # Use subplots_adjust instead of tight_layout for better control
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, 
                       hspace=0.3, wspace=0.3)
    
    # Save figure 
    print(f"Saving figure to: {output_plot}")
    save_local(fig, output_plot, dpi)  # save local
    print("Plot saved")
    
    print(f"Figure created and saved successfully")
    
    return fig


def plot_markers_detection_enhanced(tt, audio_signals, detected_markers, analysis, colours):
    """Enhanced version of plot_markers_detection for the enhanced tapping analysis.
    
    Creates a subplot showing:
    - Final marker signal (blue)
    - Test signal (red) if available
    - Detection statistics
    
    Parameters
    ----------
    tt : array
        Time points array
    audio_signals : dict
        Dictionary containing processed audio signals
    detected_markers : array
        Array of detected marker onset times
    analysis : dict
        Analysis results including detection statistics
    colours : list
        Color palette for plotting
    """
    # Plot marker and test signals if available
    if 'rec_markers_final' in audio_signals:
        rec_markers_final = audio_signals['rec_markers_final']
        plt.plot(tt, rec_markers_final, 'b-', linewidth=1.2, label='Marker signal')
        
        if 'rec_test_final' in audio_signals:
            rec_test_final = audio_signals['rec_test_final']
            plt.plot(tt, rec_test_final, 'r-', linewidth=1.0, alpha=0.7, label='Test signal')
    elif 'rec_markers_clean' in audio_signals:
        rec_markers_clean = audio_signals['rec_markers_clean']
        plt.plot(tt, rec_markers_clean, 'b-', linewidth=1.2, label='Marker signal')
    
    # Calculate and display detection statistics
    expected_markers = len(detected_markers)
    detected_count = len(detected_markers)
    
    if expected_markers > 0:
        percent_markers_detected = (detected_count / expected_markers) * 100
        message = f'Markers detection: \n {percent_markers_detected:.0f}% markers detected ({detected_count} out of {expected_markers})'
    else:
        message = f'Markers detection: \n {detected_count} markers detected'
    
    plt.title(message, fontsize=14, fontweight='bold')
    
    # Add marker indicators if we have detected markers
    if len(detected_markers) > 0 and len(tt) > 0:
        y_max = np.max(audio_signals.get('rec_markers_final', audio_signals.get('rec_markers_clean', [0.1])))
        plt.scatter(detected_markers / 1000.0, [y_max * 0.8] * len(detected_markers), 
                   color=colours[4], s=60, marker='s', zorder=5, label='Detected markers')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)


def plot_markers_error_enhanced(tt, detected_markers, markers_onsets_aligned, analysis, verify, colours):
    """Enhanced version of plot_markers_error for the enhanced tapping analysis.
    
    Creates a subplot showing:
    - Marker timing accuracy
    - Expected vs detected marker timing
    - Error statistics from enhanced verification
    
    Parameters
    ----------
    tt : array
        Time points array
    detected_markers : array
        Array of detected marker onset times
    markers_onsets_aligned : array
        Array of aligned marker onset times
    analysis : dict
        Analysis results including enhanced marker verification
    verify : dict
        Verification data from alignment
    colours : list
        Color palette for plotting
    """
    # Get enhanced marker verification data if available
    marker_verification = analysis.get('marker_verification', {})
    
    if len(detected_markers) > 0 and len(markers_onsets_aligned) > 0:
        # Calculate marker IOIs
        detected_ioi = np.diff(detected_markers) if len(detected_markers) > 1 else np.array([])
        aligned_ioi = np.diff(markers_onsets_aligned) if len(markers_onsets_aligned) > 1 else np.array([])
        
        if len(detected_ioi) > 0:
            # Plot detected marker IOIs
            plt.plot(detected_markers[1:] / 1000.0, detected_ioi, 'rx-', markersize=6, linewidth=1.5, label='Detected IOI')
            
            # Plot aligned marker IOIs if different
            if len(aligned_ioi) > 0 and not np.array_equal(detected_ioi, aligned_ioi):
                plt.plot(markers_onsets_aligned[1:] / 1000.0, aligned_ioi, 'go-', markersize=6, linewidth=1.5, label='Aligned IOI')
            
            # Add mean line
            mean_ioi = np.mean(detected_ioi)
            plt.axhline(mean_ioi, color=colours[2], linestyle='--', alpha=0.8, 
                       label=f'Mean: {mean_ioi:.1f}ms')
            
            # Add error bounds if verification data available
            if verify and 'markers_ioi_std' in verify:
                std_ioi = verify['markers_ioi_std']
                plt.axhline(mean_ioi + std_ioi, color=colours[3], linestyle=':', alpha=0.7, 
                           label=f'+1SD: {std_ioi:.1f}ms')
                plt.axhline(mean_ioi - std_ioi, color=colours[3], linestyle=':', alpha=0.7, 
                           label=f'-1SD: {std_ioi:.1f}ms')
    
    # Use enhanced verification data for title if available
    if marker_verification and 'verification_status' in marker_verification:
        status = marker_verification['verification_status']
        detection_rate = marker_verification.get('detection_rate', 0)
        quality_score = marker_verification.get('quality_score', 0)
        
        if status == 'PASSED':
            status_color = 'green'
        elif status == 'WARNING':
            status_color = 'orange'
        else:
            status_color = 'red'
        
        message = f'Marker Verification: {status}\nDetection: {detection_rate:.0f}% | Quality: {quality_score:.0f}/100'
        
        # Add timing error info if available
        if 'mean_timing_error' in marker_verification and marker_verification['mean_timing_error'] >= 0:
            mean_error = marker_verification['mean_timing_error']
            timing_std = marker_verification.get('timing_std', 0)
            message += f'\nTiming: {mean_error:.1f}±{timing_std:.1f}ms'
    else:
        # Fallback to original verification data
        if verify and 'markers_ioi_cv' in verify:
            markers_cv = verify['markers_ioi_cv']
            markers_status = "Good" if markers_cv < 5.0 else "Poor"
            message = f'Markers timing accuracy: {markers_status}\n CV: {markers_cv:.1f}%'
        else:
            message = 'Markers timing accuracy'
    
    plt.title(message, fontsize=14, fontweight='bold')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Marker IOI (ms)')
    plt.legend()
    plt.grid(True, alpha=0.3)


def enhanced_tapping_analysis(recording_filename, title_plot, output_plot, dpi=300, stim_info=None):
    """
    Enhanced analysis function that provides comprehensive tapping analysis with detailed plots.
    
    Parameters
    ----------
    recording_filename : str
        Path to the audio recording file
    title_plot : str
        Title for the generated plots
    output_plot : str
        Path where plot should be saved
    dpi : int, optional
        Resolution for saved plot (default: 300)
    stim_info : dict, required
        Stimulus information containing markers_onsets and other metadata.
        MUST contain 'markers_onsets' array with marker onset times in milliseconds.
        If markers_onsets is not provided, a ValueError will be raised.
    
    Returns
    -------
    audio_signals : dict
        Processed audio signals from the recording
    extracted_onsets : dict
        Detected tap onsets and timing information
    analysis : dict
        Enhanced statistical analysis of tapping performance
        
    Raises
    ------
    ValueError
        If stim_info is None or does not contain markers_onsets
    """
    # Import REPP modules
    from repp.analysis import REPPAnalysis
    from repp.config import sms_tapping
    import gc
    
    # Validate that stim_info and markers_onsets are provided
    if stim_info is None:
        raise ValueError(
            "stim_info is required for enhanced tapping analysis. "
            "Please provide stim_info containing markers_onsets."
        )
    
    markers_onsets = stim_info.get('markers_onsets', [])
    if len(markers_onsets) == 0:
        raise ValueError(
            "markers_onsets is required in stim_info for enhanced tapping analysis. "
            "The experiment should always include markers_onsets. "
            f"Received stim_info: {stim_info}"
        )
    
    analysis = REPPAnalysis(config=sms_tapping)
    
    try:
        # Use custom tapping-only analysis that skips plotting
        print("Using tapping-only analysis to detect participant taps...")
        audio_signals, extracted_onsets, analysis_results = do_analysis_tapping_only_no_plot(
            analysis, recording_filename, title_plot, output_plot
        )
        
        # Extract the tapping onsets from the analysis results
        tap_onsets = analysis_results.get('resp_onsets_detected', [])
        
        # Determine stimulus type and target tempo
        stimulus_type = "unknown"
        target_ioi = None
        target_bpm = None
        is_music_stimulus = False
        
        # Get IOI data for target calculation
        iois = np.array(analysis_results.get('resp_ioi_detected', []))
        
        # Use title-based detection for stimulus type
        if 'iso_800ms' in title_plot:
            stimulus_type = "isochronous"
            target_ioi = 800
            target_bpm = 75
        elif 'iso_600ms' in title_plot:
            stimulus_type = "isochronous"
            target_ioi = 600
            target_bpm = 100
        elif 'track' in title_plot or 'music' in title_plot:
            stimulus_type = "music"
            is_music_stimulus = True
            if len(iois) > 0:
                target_ioi = np.median(iois)
                target_bpm = 60000 / target_ioi if target_ioi > 0 else None
        
        print(f"Stimulus type: {stimulus_type}, Target IOI: {target_ioi}ms, Target BPM: {target_bpm}")
        
        # Filter out audio markers for all stimulus types
        if len(tap_onsets) > 0:
            print("Filtering audio markers from all stimulus types...")
            duration_seconds = stim_info.get('stim_duration', 0) if stim_info else 0
            if duration_seconds == 0 and len(audio_signals.get('time_line_for_sample', [])) > 0:
                duration_seconds = max(audio_signals['time_line_for_sample'])
            
            # Use precise filtering with actual marker timing (markers_onsets already validated)
            # This provides a second layer of filtering using the expected marker times
            print("Using precise marker filtering with markers_onsets data...")
            filtered_onsets, filtering_info = filter_audio_markers_precise(
                np.array(tap_onsets), 
                markers_onsets,
                duration_seconds,
                tolerance_ms=300  # Reduced tolerance since we already filtered with detected markers
            )
            tap_onsets = filtered_onsets.tolist()
            
            # Store filtering information in analysis results
            analysis_results['filtering_info'] = filtering_info
            analysis_results['markers_removed'] = filtering_info.get('markers_removed', 0)
            
            print(f"Precise filtering completed - {filtering_info.get('markers_removed', 0)} additional markers removed")
            print(f"  - Original taps: {filtering_info.get('original_taps', 0)}")
            print(f"  - Filtered taps: {filtering_info.get('filtered_taps', 0)}")

        # Align taps to markers (beat-finding style) - this is the ONLY alignment step
        print("Aligning taps to first marker...")
        alignment = align_taps_to_markers(
            np.array(tap_onsets),
            np.array(markers_onsets)
        )
        
        print(f"Alignment completed - offset: {alignment['alignment_offset']:.1f}ms")
        print(f"  - Taps aligned: {len(alignment['tapping_onsets_aligned'])}")
        
        # Validate the alignment
        validation = validate_alignment(
            np.array(tap_onsets),  # Original taps before alignment
            alignment['tapping_onsets_aligned'],  # Aligned taps
            np.array(markers_onsets),  # Original markers
            alignment['alignment_offset']  # Applied offset
        )
        
        if not validation["valid"]:
            print("WARNING: Alignment validation failed!")
            for error in validation["errors"]:
                print(f"  ERROR: {error}")
        else:
            print("Alignment validation passed ✓")
            
        for warning in validation["warnings"]:
            print(f"  WARNING: {warning}")

        # Convert to the format expected by the enhanced analysis
        # Use the aligned taps from the alignment result, not the original filtered taps
        final_tap_onsets = alignment['tapping_onsets_aligned'] if len(alignment['tapping_onsets_aligned']) > 0 else np.array(tap_onsets)
        extracted_onsets = {
            'tapping_detected_onsets': final_tap_onsets,
            'num_tapping_detected_onsets': len(final_tap_onsets)
        }

        # Update analysis_results to use the aligned taps
        analysis_results['resp_onsets_detected'] = final_tap_onsets.tolist()
        analysis_results['num_resp_onsets_detected'] = len(final_tap_onsets)
        
        # Final validation of the results
        print(f"\nFINAL VALIDATION:")
        print(f"  - Final tap onsets: {len(final_tap_onsets)}")
        if len(final_tap_onsets) > 0:
            print(f"  - First tap time: {final_tap_onsets[0]:.1f}ms")
            print(f"  - Last tap time: {final_tap_onsets[-1]:.1f}ms")
            print(f"  - Tap range: {final_tap_onsets[-1] - final_tap_onsets[0]:.1f}ms")
            
            # Check that first marker is at 0
            markers_aligned = np.array(markers_onsets) - alignment['alignment_offset']
            print(f"  - First marker aligned: {markers_aligned[0]:.1f}ms (should be 0)")
            
            # Check for any taps before the first marker (should not happen)
            taps_before_first_marker = np.sum(final_tap_onsets < 0)
            if taps_before_first_marker > 0:
                print(f"  WARNING: {taps_before_first_marker} taps occur before the first marker")
            else:
                print(f"  ✓ All taps occur after first marker")
        
        print(f"  - Alignment offset applied: {alignment['alignment_offset']:.1f}ms")
        print(f"  - Total markers removed: {analysis_results.get('markers_removed', 0)}")

        # Recalculate IOIs if we filtered markers
        if len(final_tap_onsets) > 1:
            new_iois = np.diff(final_tap_onsets)
            analysis_results['resp_ioi_detected'] = new_iois.tolist()
            print(f"Recalculated IOIs after alignment: {len(new_iois)} intervals")
            
            # Recalculate target IOI for music after filtering
            if is_music_stimulus and len(new_iois) > 0:
                target_ioi = np.median(new_iois)
                target_bpm = 60000 / target_ioi if target_ioi > 0 else None
                print(f"Updated target IOI after alignment: {target_ioi:.1f}ms ({target_bpm:.1f} BPM)")

        # Add alignment info to analysis_results for downstream use
        analysis_results['alignment'] = alignment

        # ENHANCED MARKER VERIFICATION
        # Get detected markers from audio_signals for verification
        detected_markers = audio_signals.get('markers_detected_onsets', [])
        
        # Use the enhanced marker verification function to assess marker detection quality
        if len(detected_markers) > 0 and len(markers_onsets) > 0:
            print("\nENHANCED MARKER VERIFICATION:")
            marker_verification = verify_markers_detection_enhanced(
                np.array(detected_markers),
                np.array(markers_onsets),
                max_proximity=300.0,  # 300ms tolerance for marker detection
                max_proximity_phase=[-0.5, 0.5]  # Phase tolerance
            )
            
            # Add verification results to analysis_results
            analysis_results['marker_verification'] = marker_verification
            
            # Print verification results
            print(f"  - Verification status: {marker_verification['verification_status']}")
            print(f"  - Detection rate: {marker_verification['detection_rate']:.1f}% ({marker_verification['num_detected']}/{marker_verification['num_expected']})")
            print(f"  - Max timing error: {marker_verification['max_timing_error']:.1f}ms")
            print(f"  - Mean timing error: {marker_verification['mean_timing_error']:.1f}ms")
            print(f"  - Timing std: {marker_verification['timing_std']:.1f}ms")
            print(f"  - Quality score: {marker_verification['quality_score']:.1f}/100")
            print(f"  - Reason: {marker_verification['verification_reason']}")
            
            # Update overall analysis quality based on marker verification
            if not marker_verification['verification_passed']:
                if analysis_results.get('failed', False):
                    # If already failed, append marker issue to reason
                    analysis_results['reason'] += f" | Marker verification: {marker_verification['verification_reason']}"
                else:
                    # If not already failed, consider marker issues as warnings
                    analysis_results['marker_warning'] = marker_verification['verification_reason']
        else:
            print("\nENHANCED MARKER VERIFICATION: Skipped (no detected markers or expected markers)")
            analysis_results['marker_verification'] = {
                'verification_passed': False,
                'verification_status': 'SKIPPED',
                'verification_reason': 'No markers available for verification'
            }

        # Apply quality assessment using the same function as the experiment
        # Use the same thresholds as defined in the experiment (MIN_RAW_TAPS=10, MAX_RAW_TAPS=500)
        quality_assessment = check_tapping_quality(final_tap_onsets, min_taps=10, max_taps=500, cv_threshold=35.0, stimulus_type=stimulus_type)
        analysis_results['failed'] = quality_assessment['failed']
        analysis_results['reason'] = quality_assessment['reason']
        
        # ADD MARKER DETECTION CRITERIA 
        # Check if all markers are detected and timing error is acceptable
        marker_verification = analysis_results.get('marker_verification', {})
        
        # Criterion 1: All markers must be detected
        all_markers_detected = True
        if len(markers_onsets) > 0:
            num_expected_markers = len(markers_onsets)
            num_detected_markers = marker_verification.get('num_detected', 0)
            all_markers_detected = num_detected_markers == num_expected_markers
            
            if not all_markers_detected:
                if analysis_results.get('failed', False):
                    # If already failed, append marker issue to reason
                    analysis_results['reason'] += f" | Not all markers detected ({num_detected_markers}/{num_expected_markers})"
                else:
                    # If not already failed, fail due to marker detection
                    analysis_results['failed'] = True
                    analysis_results['reason'] = f"Not all markers detected ({num_detected_markers}/{num_expected_markers})"
        
        # Criterion 2: Marker timing error must be low (using MARKERS_MAX_ERROR threshold)
        markers_error_low = True
        MARKERS_MAX_ERROR = 100.0  # Default value from REPP config (can be overridden)
        max_timing_error = marker_verification.get('max_timing_error', -1)
        
        if max_timing_error >= 0:  # Only check if we have timing error data
            markers_error_low = max_timing_error < MARKERS_MAX_ERROR
            
            if not markers_error_low:
                if analysis_results.get('failed', False):
                    # If already failed, append marker timing issue to reason
                    analysis_results['reason'] += f" | Marker timing error too large ({max_timing_error:.1f}ms > {MARKERS_MAX_ERROR}ms)"
                else:
                    # If not already failed, fail due to marker timing
                    analysis_results['failed'] = True
                    analysis_results['reason'] = f"Marker timing error too large ({max_timing_error:.1f}ms > {MARKERS_MAX_ERROR}ms)"
        
        # Store marker criteria results for reference
        analysis_results['marker_criteria'] = {
            'all_markers_detected': all_markers_detected,
            'markers_error_low': markers_error_low,
            'num_expected_markers': len(markers_onsets),
            'num_detected_markers': marker_verification.get('num_detected', 0),
            'max_timing_error': max_timing_error,
            'markers_max_error_threshold': MARKERS_MAX_ERROR
        }
        
        print(f"\nMARKER DETECTION CRITERIA:")
        print(f"  - All markers detected: {all_markers_detected} ({marker_verification.get('num_detected', 0)}/{len(markers_onsets)})")
        print(f"  - Markers error low: {markers_error_low} (max error: {max_timing_error:.1f}ms, threshold: {MARKERS_MAX_ERROR}ms)")
        print(f"  - Overall analysis failed: {analysis_results.get('failed', False)}")
        print(f"  - Failure reason: {analysis_results.get('reason', 'N/A')}")
        
        # Add enhanced metrics if we have IOI data
        if len(final_tap_onsets) > 1:
            aligned_iois = np.diff(final_tap_onsets)
            analysis_results['mean_ioi'] = np.mean(aligned_iois)
            analysis_results['ioi_std'] = np.std(aligned_iois)
            analysis_results['ioi_cv'] = (np.std(aligned_iois) / np.mean(aligned_iois)) * 100 if np.mean(aligned_iois) > 0 else 0
            analysis_results['ioi_range'] = np.max(aligned_iois) - np.min(aligned_iois)
            analysis_results['ioi_min'] = np.min(aligned_iois)
            analysis_results['ioi_max'] = np.max(aligned_iois)
            
            # Calculate tempo drift
            if len(aligned_iois) > 2:
                tap_indices = np.arange(1, len(aligned_iois) + 1)
                z = np.polyfit(tap_indices, aligned_iois, 1)
                analysis_results['tempo_drift_ms_per_tap'] = z[0]
                analysis_results['tempo_drift_bpm_per_tap'] = z[0] * 60000 / (np.mean(aligned_iois) ** 2)

        # Create enhanced plots with stimulus information
        try:
            fig = create_enhanced_tapping_plots(
                audio_signals, extracted_onsets, analysis_results, 
                title_plot, output_plot, dpi, stim_info
            )
            print("Plot saved")
            del fig
            gc.collect()
        except Exception as plot_error:
            print(f"Warning: Could not create enhanced plots: {plot_error}")
            # Continue without enhanced plots
        
        return audio_signals, extracted_onsets, analysis_results
        
    except Exception as e:
        print(f"Error in enhanced analysis: {e}")
        # Create fallback results
        audio_signals = {
            'time_line_for_sample': np.array([]),
            'rec_downsampled': np.array([]),
            'rec_tapping_clean': np.array([])
        }
        extracted_onsets = {
            'tapping_detected_onsets': np.array([]),
            'num_tapping_detected_onsets': 0
        }
        analysis_results = {
            'resp_ioi_detected': [],
            'resp_onsets_detected': [],
            'num_resp_onsets_detected': 0,
            'median_ioi': 9999,
            'bpm': 0,
            'q1_ioi': 0,
            'q3_ioi': 0,
            'failed': True,
            'reason': f"Analysis failed: {str(e)}"
        }
        
        return audio_signals, extracted_onsets, analysis_results


def do_analysis_tapping_only_no_plot(analysis_instance, recording_filename, title_plot, output_plot, dpi=300):
    """
    Perform analysis for unconstrained tapping experiments without plotting.
    
    This is a modified version of do_analysis_tapping_only that skips the plotting
    step, similar to how do_only_stats works for the main analysis.
    
    Parameters
    ----------
    analysis_instance : REPPAnalysis
        The REPPAnalysis instance to use
    recording_filename : str
        Path to the audio recording file
    title_plot : str
        Title for the generated plots (not used in this version)
    output_plot : str
        Path where plot would be saved (not used in this version)
    dpi : int, optional
        Resolution for saved plot (not used in this version)

    Returns
    -------
    audio_signals : dict
        Processed audio signals from the recording
    extracted_onsets : dict
        Detected tap onsets and timing information
    analysis : dict
        Statistical analysis of tapping performance
    """
    # Import signal processing module
    from repp import signal_processing as sp
    
    # Extract and process audio signals using FULL signal processing (not just tapping)
    # This ensures we get both markers and tapping channels for better separation
    audio_signals = sp.extract_audio_signals(recording_filename, analysis_instance.config)
    print("Tapping analysis with full signal processing...")
    
    # Detect BOTH markers and tapping onsets for better filtering
    extracted_onsets = sp.extract_onsets(audio_signals, analysis_instance.config)
    
    # Get detected markers and taps
    detected_markers = extracted_onsets.get('markers_detected_onsets', [])
    detected_taps = extracted_onsets.get('tapping_detected_onsets', [])
    
    print(f"Detected {len(detected_markers)} markers and {len(detected_taps)} taps")
    
    # Filter out taps that are too close to detected markers
    if len(detected_markers) > 0 and len(detected_taps) > 0:
        print("Filtering taps that are too close to detected markers...")
        print(f"  - Detected markers: {detected_markers}")
        print(f"  - Initial taps: {detected_taps}")
        
        filtered_taps = []
        markers_removed = 0
        
        for tap_time in detected_taps:
            # Check if this tap is close to any detected marker
            is_marker = False
            for marker_time in detected_markers:
                if abs(tap_time - marker_time) <= 500:  # 500ms tolerance
                    is_marker = True
                    markers_removed += 1
                    print(f"  - Removed tap at {tap_time:.1f}ms (close to detected marker at {marker_time:.1f}ms)")
                    break
            
            if not is_marker:
                filtered_taps.append(tap_time)
        
        print(f"Removed {markers_removed} marker-related taps using detected markers")
        print(f"  - Remaining taps: {filtered_taps}")
        detected_taps = filtered_taps
    
    # Create the format expected by the analysis
    extracted_onsets_tapping_only = {
        'tapping_detected_onsets': np.array(detected_taps),
        'num_tapping_detected_onsets': len(detected_taps)
    }
    
    # Add detected markers information to audio_signals for plotting
    audio_signals['markers_detected_onsets'] = detected_markers
    
    # Analyze tapping patterns
    analysis = analysis_instance.do_stats_only_tapping(extracted_onsets_tapping_only)
    print("Analysing results...")
    
    # Skip plotting and return results directly
    return audio_signals, extracted_onsets_tapping_only, analysis


def filter_audio_markers_precise(tap_onsets, markers_onsets, stim_duration_seconds, tolerance_ms=500):
    """
    Filter out audio markers using precise timing information from markers_onsets.
    
    This function uses the actual marker onset times to precisely identify and remove
    audio markers. It does NOT perform alignment - that is handled separately by align_taps_to_markers.
    
    Parameters
    ----------
    tap_onsets : np.array
        Array of tap onset times in milliseconds
    markers_onsets : np.array
        Array of marker onset times in milliseconds (from stim_info['markers_onsets'])
    stim_duration_seconds : float
        Total duration of the stimulus in seconds
    tolerance_ms : float
        Tolerance window in milliseconds to identify marker-related taps (default: 500ms)
        
    Returns
    -------
    tuple
        (filtered_onsets, alignment_info)
        - filtered_onsets: np.array of tap onsets with markers removed (NOT aligned)
        - alignment_info: dict with filtering details (no alignment performed)
    """
    if len(tap_onsets) == 0:
        return tap_onsets, {"filtered": False, "markers_removed": 0}
    
    if len(markers_onsets) == 0:
        raise ValueError(
            "markers_onsets is required for precise audio marker filtering. "
            "The experiment should always include markers_onsets. "
            f"Received markers_onsets: {markers_onsets}"
        )
    
    print(f"Precise marker filtering with {len(markers_onsets)} markers")
    print(f"Marker times: {markers_onsets}")
    
    # Convert to numpy arrays
    tap_onsets = np.array(tap_onsets)
    markers_onsets = np.array(markers_onsets)
    
    # Create a mask for all taps initially
    valid_mask = np.ones(len(tap_onsets), dtype=bool)
    
    # Identify and remove marker-related taps
    markers_removed = 0
    for marker_time in markers_onsets:
        # Find taps that are close to this marker time
        time_diff = np.abs(tap_onsets - marker_time)
        marker_indices = np.where(time_diff <= tolerance_ms)[0]
        
        for idx in marker_indices:
            if valid_mask[idx]:  # Only remove if not already marked for removal
                valid_mask[idx] = False
                markers_removed += 1
                print(f"  - Removed marker-related tap at {tap_onsets[idx]:.1f}ms (close to marker at {marker_time:.1f}ms)")
    
    # Apply filter
    filtered_onsets = tap_onsets[valid_mask]
    
    print(f"Precise filtering removed {markers_removed} marker-related taps")
    print(f"  - Valid taps remaining: {len(filtered_onsets)}")
    
    # Return filtering info (no alignment performed here)
    alignment_info = {
        "filtered": True, 
        "markers_removed": markers_removed,
        "original_taps": len(tap_onsets),
        "filtered_taps": len(filtered_onsets)
    }
    
    return filtered_onsets, alignment_info


def check_tapping_quality(tapping_onsets, min_taps=10, max_taps=500, cv_threshold=35.0, stimulus_type="unknown"):
    """
    Evaluates the quality of tapping in a tapping task.

    Parameters
    ----------
    tapping_onsets : list or np.array
        Array of tap onset times in milliseconds
    min_taps : int
        Minimum number of taps required (default: 10)
    max_taps : int
        Maximum number of taps allowed (default: 500)
    cv_threshold : float
        Maximum allowed coefficient of variation (CV) in percentage (default: 35.0%)
        Based on literature: 
        - CV < 15%: Excellent consistency
        - CV 15-25%: Good consistency  
        - CV 25-35%: Acceptable consistency
        - CV > 35%: Poor consistency
    stimulus_type : str
        Type of stimulus ("isochronous", "music", or "unknown") for context-specific assessment

    Returns
    -------
    dict
        Dictionary containing quality assessment results
    """
    failed = False
    reason = None
    interval_std = None
    cv = None
    
    # Handle None or empty input
    if tapping_onsets is None:
        failed = True
        reason = "No tapping data received"
        return {
            "failed": failed,
            "num_taps": 0,
            "interval_SD_ms": interval_std,
            "cv_percent": cv,
            "reason": reason,
            "stimulus_type": stimulus_type
        }
    
    # Convert to numpy array if it's not already
    tapping_onsets = np.array(tapping_onsets, dtype=float)

    # Check minimum number of taps
    if len(tapping_onsets) < min_taps:
        failed = True
        reason = f"Too few taps ({len(tapping_onsets)} < {min_taps})"
    
    # Check maximum number of taps
    if len(tapping_onsets) > max_taps:
        failed = True
        reason = f"Too many taps ({len(tapping_onsets)} > {max_taps})"

    # Check consistency (isochrony) of inter-tap intervals using CV
    # Only assess consistency if the trial hasn't already failed for other reasons
    if len(tapping_onsets) > 1 and not failed:
        intervals = np.diff(tapping_onsets)
        mean_iti = np.mean(intervals)
        interval_std = np.std(intervals)
        
        # Calculate coefficient of variation (CV)
        cv = (interval_std / mean_iti) * 100 if mean_iti > 0 else 0
        
        # Adjust thresholds based on stimulus type
        adjusted_cv_threshold = cv_threshold
        if stimulus_type == "isochronous":
            # For isochronous stimuli, we expect more consistent tapping
            adjusted_cv_threshold = min(cv_threshold, 30.0)
        elif stimulus_type == "music":
            # For music stimuli, we allow more variability due to musical complexity
            adjusted_cv_threshold = max(cv_threshold, 40.0)
        
        # Check against threshold
        if cv > adjusted_cv_threshold:
            failed = True
            reason = f"Inconsistent tapping (CV = {cv:.1f}% > {adjusted_cv_threshold}%)"
        
        # Additional quality indicators with stimulus-specific context
        if cv > 50:
            reason = f"Very poor consistency (CV = {cv:.1f}%)"
        elif cv > adjusted_cv_threshold:
            reason = f"Poor consistency (CV = {cv:.1f}%)"
        elif cv > 25:
            reason = f"Acceptable consistency (CV = {cv:.1f}%)"
        elif cv > 15:
            reason = f"Good consistency (CV = {cv:.1f}%)"
        else:
            reason = f"Excellent consistency (CV = {cv:.1f}%)"
        
        # Add stimulus-specific context to reason
        if stimulus_type == "isochronous":
            reason += " (isochronous)"
        elif stimulus_type == "music":
            reason += " (music)"
    
    # If we have taps but not enough for consistency analysis, provide basic info
    elif len(tapping_onsets) == 1:
        reason = "Single tap - cannot assess consistency"
    elif len(tapping_onsets) == 0:
        reason = "No taps detected"

    return {
        "failed": failed,
        "num_taps": len(tapping_onsets),
        "interval_SD_ms": interval_std if len(tapping_onsets) > 1 else None,
        "cv_percent": cv if len(tapping_onsets) > 1 else None,
        "reason": reason,
        "stimulus_type": stimulus_type
    }

def validate_alignment(tap_onsets_original, tap_onsets_aligned, markers_onsets, alignment_offset):
    """
    Validate that the alignment is working correctly and there are no redundancies.
    
    Parameters
    ----------
    tap_onsets_original : np.array
        Original tap onsets before alignment
    tap_onsets_aligned : np.array
        Tap onsets after alignment
    markers_onsets : np.array
        Marker onsets
    alignment_offset : float
        The alignment offset applied
        
    Returns
    -------
    dict
        Validation results
    """
    validation = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check that alignment offset matches first marker
    expected_offset = markers_onsets[0] if len(markers_onsets) > 0 else 0
    if abs(alignment_offset - expected_offset) > 0.1:  # Allow small floating point differences
        validation["valid"] = False
        validation["errors"].append(f"Alignment offset mismatch: {alignment_offset} vs expected {expected_offset}")
    
    # Check that aligned taps are correctly shifted
    if len(tap_onsets_original) > 0 and len(tap_onsets_aligned) > 0:
        # Verify that aligned = original - offset
        expected_aligned = tap_onsets_original - alignment_offset
        if not np.allclose(tap_onsets_aligned, expected_aligned, atol=0.1):
            validation["valid"] = False
            validation["errors"].append("Aligned taps do not match expected alignment calculation")
    
    # Check for duplicates in aligned taps
    if len(tap_onsets_aligned) > 1:
        unique_taps = np.unique(tap_onsets_aligned)
        if len(unique_taps) != len(tap_onsets_aligned):
            validation["valid"] = False
            validation["errors"].append(f"Duplicate taps found: {len(tap_onsets_aligned)} taps, {len(unique_taps)} unique")
    
    # Check that first marker is at 0 after alignment
    if len(markers_onsets) > 0:
        markers_aligned = markers_onsets - alignment_offset
        if abs(markers_aligned[0]) > 0.1:
            validation["valid"] = False
            validation["errors"].append(f"First marker not at 0 after alignment: {markers_aligned[0]}")
    
    # Check for negative tap times (should not happen with proper alignment)
    if len(tap_onsets_aligned) > 0:
        negative_taps = np.sum(tap_onsets_aligned < 0)
        if negative_taps > 0:
            validation["warnings"].append(f"{negative_taps} taps have negative times after alignment")
    
    return validation


def align_taps_to_markers(
    tap_onsets: np.ndarray,
    markers_onsets: np.ndarray,
    tolerance_ms: float = 500.0
) -> dict:
    """
    Align tapping onsets to the first marker onset for beat-finding tasks (no stimulus onsets).
    Returns a dictionary similar in style to align_onsets, but without stimulus fields.
    Includes marker verification and quality metrics.
    """
    if len(markers_onsets) == 0:
        raise ValueError("No marker onsets provided for alignment.")
    
    # Align everything to first marker
    alignment_offset = markers_onsets[0]
    tapping_onsets_aligned = tap_onsets - alignment_offset if len(tap_onsets) > 0 else np.array([])
    markers_onsets_aligned = markers_onsets - alignment_offset
    
    # Calculate IOIs
    tapping_ioi = np.diff(tapping_onsets_aligned) if len(tapping_onsets_aligned) > 1 else np.array([])
    markers_ioi = np.diff(markers_onsets_aligned) if len(markers_onsets_aligned) > 1 else np.array([])
    
    # Marker verification (similar to verify_onsets_detection but for markers only)
    verify = {}
    if len(markers_onsets) > 1:
        # Check marker consistency
        markers_ioi_mean = np.mean(markers_ioi)
        markers_ioi_std = np.std(markers_ioi)
        markers_ioi_cv = (markers_ioi_std / markers_ioi_mean * 100) if markers_ioi_mean > 0 else 0
        
        # Check for marker detection issues
        verify = {
            'num_markers': len(markers_onsets),
            'markers_ioi_mean': markers_ioi_mean,
            'markers_ioi_std': markers_ioi_std,
            'markers_ioi_cv': markers_ioi_cv,
            'markers_consistent': markers_ioi_cv < 5.0,  # Markers should be very consistent
            'markers_ioi_range': np.max(markers_ioi) - np.min(markers_ioi) if len(markers_ioi) > 0 else 0,
            'markers_ioi_min': np.min(markers_ioi) if len(markers_ioi) > 0 else 0,
            'markers_ioi_max': np.max(markers_ioi) if len(markers_ioi) > 0 else 0
        }
    else:
        verify = {
            'num_markers': len(markers_onsets),
            'markers_ioi_mean': None,
            'markers_ioi_std': None,
            'markers_ioi_cv': None,
            'markers_consistent': False,
            'markers_ioi_range': 0,
            'markers_ioi_min': 0,
            'markers_ioi_max': 0
        }
    
    # Add tapping statistics
    if len(tapping_onsets_aligned) > 1:
        tapping_ioi_mean = np.mean(tapping_ioi)
        tapping_ioi_std = np.std(tapping_ioi)
        tapping_ioi_cv = (tapping_ioi_std / tapping_ioi_mean * 100) if tapping_ioi_mean > 0 else 0
        verify.update({
            'num_taps': len(tapping_onsets_aligned),
            'tapping_ioi_mean': tapping_ioi_mean,
            'tapping_ioi_std': tapping_ioi_std,
            'tapping_ioi_cv': tapping_ioi_cv,
            'tapping_ioi_range': np.max(tapping_ioi) - np.min(tapping_ioi),
            'tapping_ioi_min': np.min(tapping_ioi),
            'tapping_ioi_max': np.max(tapping_ioi)
        })
    else:
        verify.update({
            'num_taps': len(tapping_onsets_aligned),
            'tapping_ioi_mean': None,
            'tapping_ioi_std': None,
            'tapping_ioi_cv': None,
            'tapping_ioi_range': 0,
            'tapping_ioi_min': 0,
            'tapping_ioi_max': 0
        })
    
    return {
        'tapping_onsets_aligned': tapping_onsets_aligned,
        'markers_onsets_aligned': markers_onsets_aligned,
        'tapping_ioi': tapping_ioi,
        'markers_ioi': markers_ioi,
        'alignment_offset': alignment_offset,
        'verify': verify
    } 


########################################
# Marker verification functions
########################################

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


def verify_markers_detection_enhanced(
    detected_markers: np.ndarray,
    expected_markers: np.ndarray,
    max_proximity: float = 300.0,
    max_proximity_phase: List[float] = [-0.5, 0.5]
) -> Dict[str, Union[int, float, np.ndarray, str]]:
    """
    Enhanced marker verification using the verify_onsets_detection function.
    
    This function provides comprehensive verification of detected markers against
    expected markers, including detection accuracy, timing precision, and quality metrics.
    
    Parameters
    ----------
    detected_markers : np.ndarray
        Array of detected marker onset times in milliseconds
    expected_markers : np.ndarray
        Array of expected marker onset times in milliseconds
    max_proximity : float
        Maximum allowed timing error in milliseconds (default: 300ms)
    max_proximity_phase : List[float]
        Allowed phase error range (default: [-0.5, 0.5])
        
    Returns
    -------
    dict
        Dictionary containing comprehensive verification metrics
    """
    if len(detected_markers) == 0:
        return {
            'verification_passed': False,
            'verification_status': 'FAILED',
            'verification_reason': 'No markers detected',
            'num_detected': 0,
            'num_expected': len(expected_markers),
            'detection_rate': 0.0,
            'max_timing_error': -1,
            'mean_timing_error': -1,
            'timing_std': -1,
            'quality_score': 0.0
        }
    
    if len(expected_markers) == 0:
        return {
            'verification_passed': False,
            'verification_status': 'FAILED',
            'verification_reason': 'No expected markers provided',
            'num_detected': len(detected_markers),
            'num_expected': 0,
            'detection_rate': 0.0,
            'max_timing_error': -1,
            'mean_timing_error': -1,
            'timing_std': -1,
            'quality_score': 0.0
        }
    
    # Use the verify_onsets_detection function
    verification = verify_onsets_detection(
        detected_markers,
        expected_markers,
        max_proximity,
        max_proximity_phase
    )
    
    # Calculate additional metrics
    num_detected = verification['verify_num_detected']
    num_expected = len(expected_markers)
    detection_rate = (num_detected / num_expected) * 100 if num_expected > 0 else 0
    max_timing_error = verification['verify_max_difference']
    
    # Calculate mean timing error and standard deviation
    asynchrony = verification['verify_asynchrony']
    valid_asynchronies = asynchrony[~np.isnan(asynchrony)]
    mean_timing_error = np.mean(np.abs(valid_asynchronies)) if len(valid_asynchronies) > 0 else -1
    timing_std = np.std(valid_asynchronies) if len(valid_asynchronies) > 0 else -1
    
    # Determine verification status
    verification_passed = True
    verification_status = 'PASSED'
    verification_reason = 'All markers detected within acceptable timing'
    
    # Check detection rate
    if detection_rate < 80:  # Less than 80% detection rate
        verification_passed = False
        verification_status = 'FAILED'
        verification_reason = f'Low detection rate ({detection_rate:.1f}%)'
    
    # Check timing accuracy
    elif max_timing_error > max_proximity:
        verification_passed = False
        verification_status = 'FAILED'
        verification_reason = f'Timing error too large ({max_timing_error:.1f}ms > {max_proximity}ms)'
    
    # Check timing consistency
    elif timing_std > max_proximity / 2:  # Standard deviation should be less than half the max proximity
        verification_passed = False
        verification_status = 'WARNING'
        verification_reason = f'Inconsistent timing (std: {timing_std:.1f}ms)'
    
    # Calculate quality score (0-100)
    quality_score = 0.0
    if num_expected > 0:
        # Detection component (40% weight)
        detection_component = (num_detected / num_expected) * 40
        
        # Timing accuracy component (40% weight)
        timing_component = max(0, 40 - (mean_timing_error / max_proximity) * 40) if mean_timing_error >= 0 else 0
        
        # Consistency component (20% weight)
        consistency_component = max(0, 20 - (timing_std / (max_proximity / 2)) * 20) if timing_std >= 0 else 0
        
        quality_score = detection_component + timing_component + consistency_component
    
    return {
        'verification_passed': verification_passed,
        'verification_status': verification_status,
        'verification_reason': verification_reason,
        'num_detected': num_detected,
        'num_expected': num_expected,
        'detection_rate': detection_rate,
        'max_timing_error': max_timing_error,
        'mean_timing_error': mean_timing_error,
        'timing_std': timing_std,
        'quality_score': quality_score,
        # Include all original verification data
        **verification
    } 