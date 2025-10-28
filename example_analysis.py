####################################################################################################
# File: analysis.py
# Purpose: Main functions for analyzing sensorimotor synchronization (SMS) experiments
# 
# This module provides functionality for:
# 1. Signal processing of audio recordings
# 2. Analysis of tapping responses
# 3. Visualization of results
#
# Key classes:
# - REPPAnalysis: Main class handling all analysis steps
#
# Authors: Manuel Anglada-Tort, Peter Harrison, Nori Jacoby
####################################################################################################

import numpy as np
from matplotlib import pyplot as plt
import gc
from repp import signal_processing as sp

class REPPAnalysis:
    """Main class for analyzing sensorimotor synchronization (SMS) experiments.
    
    This class handles the complete analysis pipeline including:
    - Signal processing of audio recordings
    - Extraction and alignment of tapping responses
    - Statistical analysis of tapping performance
    - Visualization of results
    
    Typical usage:
    ```python
    from repp.analysis import REPPAnalysis
    
    # Initialize analyzer with configuration
    analyzer = REPPAnalysis(config)
    
    # Run complete analysis with plots
    output, analysis, is_failed = analyzer.do_analysis(
        stim_info=stim_dict,
        recording_filename='recording.wav',
        title_plot='My Experiment',
        output_plot='results.png'
    )
    ```

    Attributes
    ----------
    config : class
        Configuration parameters for the experiment (see ``config.py``)
        Contains critical parameters like thresholds, windows sizes, etc.
    """

    def __init__(self, config):
        self.config = config

    def do_only_stats(self, stim_info, recording_filename):
        """Perform statistical analysis without generating plots.
        
        This is a lightweight version of do_analysis() that skips visualization.
        Useful for batch processing or when plots aren't needed.

        Parameters
        ----------
        stim_info : dict
            Dictionary containing stimulus information including:
            - markers_onsets: timing of marker signals
            - stim_shifted_onsets: timing of stimulus onsets
            - onset_is_played: boolean array indicating which onsets were played
        recording_filename : str
            Path to the audio recording file

        Returns
        -------
        output : dict
            Raw signal processing results
        analysis : dict
            Statistical analysis of tapping performance
        is_failed : dict
            Quality control checks and failure status
        """
        _, audio_signals, _, aligned_onsets = self.do_signal_processing(recording_filename, stim_info)
        print("Tapping analysis...")
        output, analysis, is_failed = self.do_stats(aligned_onsets, self.config)
        print("Analysing results...")
        return output, analysis, is_failed

    def do_analysis(self, stim_info, recording_filename, title_plot, output_plot, dpi=300):
        """Perform complete analysis including visualization.
        
        This is the main analysis function that:
        1. Processes the audio recording
        2. Extracts and aligns tapping responses
        3. Performs statistical analysis
        4. Generates visualization plots
        
        Parameters
        ----------
        stim_info : dict
            Stimulus information (see do_only_stats())
        recording_filename : str
            Path to the audio recording
        title_plot : str
            Title for the generated plots
        output_plot : str
            Path where plot should be saved
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
        """
        _, audio_signals, _, aligned_onsets = self.do_signal_processing(recording_filename, stim_info)
        print("Tapping analysis...")
        output, analysis, is_failed = self.do_stats(aligned_onsets, self.config)
        print("Analysing results...")
        fig = self.do_plot(title_plot, audio_signals, aligned_onsets, analysis, is_failed, self.config)
        self.save_local(fig, output_plot, dpi)  # save local
        print("Plot saved")
        del fig
        gc.collect()
        return output, analysis, is_failed

    def do_signal_processing(self, recording_filename, stim_info):
        """Process audio signals and extract/align onsets.
        
        This method handles the complete signal processing pipeline:
        1. Prepares initial onsets from stimulus information
        2. Extracts audio signals from the recording
        3. Detects onsets in the audio signals
        4. Aligns detected onsets with expected onsets

        Parameters
        ----------
        stim_info : dict
            A dictionary containing key stimulus information:
            - markers_onsets: timing of marker signals
            - stim_shifted_onsets: timing of stimulus onsets
            - onset_is_played: boolean array indicating which onsets were played
        recording_filename : str
            Path to the audio recording file

        Returns
        -------
        initial_onsets : dict
            Dictionary with initial onsets prepared for signal processing
        audio_signals : dict
            Dictionary containing extracted and cleaned audio signals
        raw_extracted_onsets : dict
            Dictionary with raw detected onsets before alignment
        aligned_onsets : dict
            Dictionary with final aligned onsets after processing
        """
        print("Preparing initial onsets...")
        initial_onsets = sp.prepare_onsets_extraction(
            stim_info['markers_onsets'],
            stim_info['stim_shifted_onsets'],
            stim_info['onset_is_played'])
        
        print("Extracting audio signals from mono recording...")
        audio_signals = sp.extract_audio_signals(recording_filename, self.config)
        
        print("Extracting raw onsets from audio signals...")
        raw_extracted_onsets = sp.extract_onsets(audio_signals, self.config)
        
        print("Aligning onsets...")
        aligned_onsets = sp.align_onsets(
            initial_onsets,
            raw_extracted_onsets,
            self.config.MARKERS_MATCHING_WINDOW,
            self.config.ONSET_MATCHING_WINDOW_MS,
            self.config.ONSET_MATCHING_WINDOW_PHASE)
        
        return initial_onsets, audio_signals, raw_extracted_onsets, aligned_onsets

    def do_stats(self, onsets_aligned, config):
        """Calculate main statistics for SMS experiments.
        
        This method computes key performance metrics including:
        - Mean and SD of asynchronies
        - Response rates and accuracy
        - Marker detection quality
        - Failure criteria assessment

        Parameters
        ----------
        onsets_aligned : dict
            Dictionary containing aligned onset data from signal processing
        config : class
            Configuration parameters for analysis thresholds and criteria

        Returns
        -------
        output : dict
            Raw signal processing results in standardized format
        analysis : dict
            Comprehensive statistical analysis results including:
            - Marker detection performance
            - Response rates and timing accuracy
            - Separate stats for played/unplayed stimuli
        is_failed : dict
            Quality control assessment including:
            - Overall pass/fail status
            - Specific reason for failure if applicable
        """
        # Reformat output into standardized structure
        output = self.reformat_output(onsets_aligned)
        
        # Calculate asynchronies for different conditions
        asynchronies = onsets_aligned['asynchrony']
        
        # Compute statistics for all onsets
        mean_all, sd_all, number_of_resp_all, number_of_stim_all = self.compute_mean_std_async(
            asynchronies,
            config.MIN_NUM_ASYNC)  # Requires minimum 2 asynchronies for valid response
        
        # Compute statistics for played stimuli only
        mean_played, sd_played, number_of_resp_played, number_of_stim_played = self.compute_mean_std_async(
            asynchronies[onsets_aligned['stim_onsets_is_played']],
            config.MIN_NUM_ASYNC)
        
        # Compute statistics for unplayed stimuli
        mean_notplayed, sd_notplayed, number_of_resp_notplayed, number_of_stim_notplayed = self.compute_mean_std_async(
            asynchronies[~onsets_aligned['stim_onsets_is_played']],
            config.MIN_NUM_ASYNC)
        
        # Calculate number of invalid/bad taps
        num_of_bad_taps = onsets_aligned['num_resp_raw_onsets'] - number_of_resp_all
        
        # Assess marker quality
        markers_status, markers_ok = "Good", True
        if (abs(onsets_aligned['verify_max_difference']) >= config.MARKERS_MAX_ERROR or 
            onsets_aligned['verify_num_missed'] > 0):
            markers_status, markers_ok = "Bad", False

        # Compile analysis results
        analysis = {
            # Marker performance metrics
            'num_markers_onsets': len(onsets_aligned['markers_onsets_input']),
            'num_markers_detected': onsets_aligned['verify_num_detected'],
            'num_markers_missed': onsets_aligned['verify_num_missed'],
            'markers_max_difference': onsets_aligned['verify_max_difference'],
            'markers_status': markers_status,
            'markers_ok': markers_ok,
            
            # Overall performance metrics
            'num_stim_raw_all': onsets_aligned['num_stim_raw_onsets'],
            'num_stim_aligned_all': number_of_stim_all,
            'num_resp_raw_all': onsets_aligned['num_resp_raw_onsets'],
            'num_resp_aligned_all': number_of_resp_all,
            'mean_async_all': mean_all,
            'sd_async_all': sd_all,
            
            # Response rate metrics
            'ratio_resp_to_stim': 100.0 * float(np.size(onsets_aligned['resp_onsets_detected'])) / 
                                float(np.size(onsets_aligned['stim_onsets_input'], 0)),
            'percent_resp_aligned_all': 100.0 * number_of_resp_all / (0.0001 + 1.0 * number_of_stim_all),
            'num_of_bad_taps': num_of_bad_taps,
            'percent_of_bad_taps_all': round(
                100.0 * (0.0001 + num_of_bad_taps) / 
                (0.0001 + 1.0 * onsets_aligned['num_resp_raw_onsets'])),
            
            # Statistics for played stimuli
            'num_resp_aligned_played': number_of_resp_played,
            'num_stim_aligned_played': number_of_stim_played,
            'mean_async_played': mean_played,
            'sd_async_played': sd_played,
            'percent_response_aligned_played': 100.0 * number_of_resp_played / 
                                            (0.0001 + 1.0 * number_of_stim_played),
            
            # Statistics for unplayed stimuli
            'num_resp_aligned_notplayed': number_of_resp_notplayed,
            'num_stim_aligned_notplayed': number_of_stim_notplayed,
            'mean_async_notplayed': mean_notplayed,
            'sd_async_notplayed': sd_notplayed,
            'percent_response_aligned_notplayed': 100.0 * number_of_resp_notplayed / 
                                                (0.0001 + 1.0 * number_of_stim_notplayed)
        }
        
        # Assess if experiment meets quality criteria
        is_failed = self.failing_criteria(analysis, config)
        
        return output, analysis, is_failed

    def do_analysis_tapping_only(self, recording_filename, title_plot, output_plot, dpi=300):
        """Perform analysis for unconstrained tapping experiments.
        
        This method analyzes free tapping recordings without a stimulus, useful for:
        - Measuring spontaneous tapping rates
        - Analyzing tapping consistency
        - Visualizing tapping patterns
        
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

        Returns
        -------
        audio_signals : dict
            Processed audio signals from the recording
        extracted_onsets : dict
            Detected tap onsets and timing information
        analysis : dict
            Statistical analysis of tapping performance
        """
        # Extract and process audio signals
        audio_signals = sp.extract_audio_signals_tapping_only(recording_filename, self.config)
        print("Tapping analysis...")
        
        # Detect tap onsets
        extracted_onsets = sp.extract_onsets_only_tapping(audio_signals, self.config)
        
        # Analyze tapping patterns
        analysis = self.do_stats_only_tapping(extracted_onsets)
        print("Analysing results...")
        
        # Generate and save visualization
        fig = self.do_plot_tapping_only(title_plot, audio_signals, extracted_onsets, analysis)
        self.save_local(fig, output_plot, dpi)  # save local
        print("Plot saved")
        
        # Clean up matplotlib figure
        del fig
        gc.collect()
        
        return audio_signals, extracted_onsets, analysis

    def save_local(self, fig, output_filename, dpi):
        """Save or display the generated figure.
        
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

    def do_stats_only_tapping(self, extracted_onsets):
        """Calculate statistics for unconstrained tapping experiments.
        
        Computes key metrics including:
        - Inter-onset intervals (IOIs)
        - Tapping rate (BPM)
        - Tapping variability measures
        
        Parameters
        ----------
        extracted_onsets : dict
            Dictionary containing detected tap onsets including:
            - tapping_detected_onsets: array of tap onset times
            - num_tapping_detected_onsets: total number of taps

        Returns
        -------
        analysis : dict
            Dictionary containing:
            - resp_onsets_detected: list of tap onset times
            - resp_ioi_detected: list of inter-onset intervals
            - num_resp_onsets_detected: total number of taps
            - median_ioi: median inter-onset interval
            - bpm: tempo in beats per minute
            - q1_ioi: 25th percentile of IOIs
            - q3_ioi: 75th percentile of IOIs
        """
        # Extract tap onsets and convert to list
        resp_onsets_detected = extracted_onsets['tapping_detected_onsets'].tolist()
        
        # Calculate inter-onset intervals
        resp_ioi_detected = np.diff(extracted_onsets['tapping_detected_onsets']).tolist()
        
        # Get total number of taps
        num_resp_onsets_detected = extracted_onsets['num_tapping_detected_onsets']
        
        # Compile analysis results
        analysis = {
            'resp_onsets_detected': resp_onsets_detected,
            'resp_ioi_detected': resp_ioi_detected,
            'num_resp_onsets_detected': num_resp_onsets_detected,
            'median_ioi': np.nanmedian(resp_ioi_detected),
            'bpm': 60000 / np.nanmedian(resp_ioi_detected),  # Convert IOI to BPM
            'q1_ioi': np.nanpercentile(resp_ioi_detected, 25),
            'q3_ioi': np.nanpercentile(resp_ioi_detected, 75)
        }
        return analysis

    def reformat_output(self, onsets_aligned):
        """Reformat aligned onsets into a standardized output structure.
        
        Parameters
        ----------
        onsets_aligned : dict
            Raw aligned onset data from signal processing

        Returns
        -------
        output : dict
            Standardized dictionary containing:
            - Stimulus onset information
            - Response onset information
            - Asynchrony measurements
            - Marker signal information
            All timing values are rounded to 2 decimal places
        """
        output = {
            # Stimulus timing information
            'stim_onsets_input': np.round(onsets_aligned['stim_onsets_input'], 2).tolist(),
            'stim_onsets_detected': np.round(onsets_aligned['stim_onsets_detected'], 2).tolist(),
            'stim_onsets_aligned': np.round(onsets_aligned['stim_onsets_aligned'], 2).tolist(),
            'stim_ioi': np.round(onsets_aligned['stim_ioi'], 2).tolist(),
            
            # Response timing information
            'resp_onsets_detected': np.round(onsets_aligned['resp_onsets_detected'], 2).tolist(),
            'resp_onsets_aligned': np.round(onsets_aligned['resp_onsets_aligned'], 2).tolist(),
            'resp_ioi': np.round(onsets_aligned['resp_ioi'], 2).tolist(),
            
            # Asynchrony measurements
            'resp_stim_asynch': np.round(onsets_aligned['asynchrony'], 2).tolist(),
            
            # Marker signal information
            'markers_onsets_input': np.round(onsets_aligned['markers_onsets_input'], 2).tolist(),
            'markers_onsets_detected': np.round(onsets_aligned['markers_onsets_detected'], 2).tolist(),
            'markers_onsets_aligned': np.round(onsets_aligned['markers_onsets_aligned'], 2).tolist(),
            'first_marker_detected': np.round(onsets_aligned['markers_onsets_detected'], 2)[0]
        }
        return output

    def compute_mean_std_async(self, asynchronies, min_num_async):
        """Calculate mean and standard deviation of asynchronies.
        
        Computes basic statistical measures for tapping asynchronies, with checks
        for minimum number of valid responses.
        
        Parameters
        ----------
        asynchronies : list
            List of response-stimulus asynchronies (in milliseconds)
        min_num_async : float
            Minimum number of asynchronies required for valid calculation

        Returns
        -------
        mean_all : float
            Mean asynchrony (999 if insufficient valid responses)
        sd_all : float
            Standard deviation of asynchrony (999 if insufficient valid responses)
        number_of_resp : int
            Number of valid response onsets after alignment
        number_of_stim : int
            Total number of stimulus onsets
        """
        # Check if we have enough valid asynchronies
        if np.sum(~np.isnan(asynchronies)) >= min_num_async:
            # Calculate mean and SD for valid (non-NaN) asynchronies
            mean_all = np.mean(asynchronies[~np.isnan(asynchronies)])
            sd_all = np.std(asynchronies[~np.isnan(asynchronies)])
        else:
            # Set to 999 if insufficient valid responses
            mean_all = 999
            sd_all = 999
            
        # Count valid responses and total stimuli
        number_of_asynchronies = np.sum(~np.isnan(asynchronies))
        mean_all = float(mean_all)
        sd_all = float(sd_all)
        number_of_resp = int(number_of_asynchronies)
        number_of_stim = int(len(asynchronies))
        
        return mean_all, sd_all, number_of_resp, number_of_stim

    def failing_criteria(self, analysis, config):
        """Evaluate if the experiment meets quality criteria.
        
        Checks multiple criteria including:
        1. Marker detection completeness
        2. Marker timing accuracy
        3. Minimum and maximum number of taps
        4. Valid asynchrony measurements
        
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
        min_num_taps_is_ok = analysis['ratio_resp_to_stim'] >= config.MIN_RAW_TAPS
        max_num_taps_is_ok = analysis['ratio_resp_to_stim'] <= config.MAX_RAW_TAPS
        tapping_async_is_ok = (analysis['mean_async_all'] != 999 and 
                             analysis['sd_async_all'] > config.MIN_SD_ASYNC)
        
        # Combine all criteria
        failed = not (all_markers_are_detected and 
                     markers_error_is_low and 
                     min_num_taps_is_ok and 
                     max_num_taps_is_ok and 
                     tapping_async_is_ok)
        
        # Define possible failure reasons
        options = [
            all_markers_are_detected,
            markers_error_is_low,
            min_num_taps_is_ok,
            max_num_taps_is_ok,
            tapping_async_is_ok
        ]
        reasons = [
            "Not all markers detected",
            "Markers error too large",
            "Too few detected taps",
            "Too many detected taps",
            "Error in asynchrony"
        ]
        
        # Find first failed criterion
        if False in options:
            index = options.index(False)
            reason = reasons[index]
        else:
            reason = "All good"
            
        is_failed = {'failed': failed, 'reason': reason}
        return is_failed

    def do_plot(self, title_plot, audio_signals, aligned_onsets, analysis, is_failed, config):
        """Generate comprehensive visualization of analysis results.
        
        Creates a multi-panel figure showing:
        1. Original recording with detected events
        2. Tapping response analysis
        3. Marker signal analysis
        4. Asynchrony and timing analysis
        
        Parameters
        ----------
        title_plot : str
            Title for the overall plot
        audio_signals : dict
            Dictionary containing processed audio signals
        aligned_onsets : dict
            Dictionary containing aligned onset information
        analysis : dict
            Dictionary containing analysis results
        is_failed : dict
            Dictionary containing quality check results
        config : class
            Configuration parameters for plotting

        Returns
        -------
        fig : matplotlib.figure.Figure
            Complete figure with all subplots
        """
        # Setup basic plot parameters
        tt = audio_signals['time_line_for_sample']
        mxlim = [min(tt), max(tt)]
        plt.clf()

        # First row: Original recording and tapping analysis
        plot_original_recording(title_plot, mxlim, tt, audio_signals, aligned_onsets, is_failed, 1, config)
        plot_tapping_rec(tt, mxlim, audio_signals, aligned_onsets, analysis, 2, config)
        plot_tapping_zoomed(tt, audio_signals, aligned_onsets, analysis, 3, config)

        # Second row: Marker detection analysis
        plot_markers_detection(mxlim, tt, audio_signals, aligned_onsets, analysis, 5, config)
        plot_markers_cleaned(mxlim, tt, audio_signals, aligned_onsets, 6, config)
        plot_markers_zoomed(tt, audio_signals, aligned_onsets, 7, config)
        plot_markers_error(mxlim, aligned_onsets, analysis, 8, config)

        # Third row: Timing analysis
        plot_asynchrony(mxlim, aligned_onsets, analysis, 9, config)
        plot_alignment(mxlim, aligned_onsets, 10, False, config)
        plot_alignment(mxlim, aligned_onsets, 11, True, config)
        plot_asynch_and_ioi(mxlim, aligned_onsets, analysis, 12, config)

        # Set figure size and return
        fig = plt.gcf()
        fig.set_size_inches(25, 10.5)
        return fig

    def do_plot_tapping_only(self, title_plot, audio_signals, raw_onsets_extracted, analysis):
        """Generate visualization for unconstrained tapping analysis.
        
        Creates a single plot showing:
        - Raw audio signal
        - Cleaned tapping signal
        - Detected tap onsets
        - Summary statistics
        
        Parameters
        ----------
        title_plot : str
            Title for the plot
        audio_signals : dict
            Dictionary containing processed audio signals
        raw_onsets_extracted : dict
            Dictionary containing detected onset information
        analysis : dict
            Dictionary containing analysis results

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the visualization
        """
        print("plot tapping data (tapping only)...")
        
        # Extract required data
        tt = audio_signals['time_line_for_sample']
        rec_downsampled = audio_signals['rec_downsampled']
        R = raw_onsets_extracted['tapping_detected_onsets']
        R_clean = audio_signals['rec_tapping_clean']
        
        # Create plot
        plt.clf()
        plt.plot(tt, rec_downsampled)
        plt.plot(tt, R_clean)
        
        # Add detected onsets
        mmx = self.config.EXTRACT_THRESH[1]
        R = np.array(R)
        plt.plot(R / 1000.0, R * 0 + mmx, 'xr')
        
        # Add title with summary statistics
        plt.title('{}: {} detected taps; median(Q1-Q3) = {:2.2f}({:2.2f}-{:2.2f})'.format(
            title_plot,
            raw_onsets_extracted['num_tapping_detected_onsets'],
            analysis['median_ioi'],
            analysis['q1_ioi'],
            analysis['q3_ioi']))
            
        # Add labels and set size
        plt.xlabel('Time (sec)')
        plt.ylabel('Amplitude')
        fig = plt.gcf()
        fig.set_size_inches(25, 10.5)
        return fig


##################################
# supporting functions for plots
##################################


def plot_original_recording(title_plot, mxlim, tt, audio_signals, aligned_onsets, is_failed, position_subplot, config):
    """Plot the original recording with detected events overlaid.
    
    Creates a subplot showing:
    - Raw downsampled audio signal
    - Detected marker onsets
    - Detected response (tap) onsets 
    - Detected stimulus onsets
    
    Parameters
    ----------
    title_plot : str
        Title for the plot
    mxlim : list
        [min_x, max_x] time limits for x-axis
    tt : array
        Time points array
    audio_signals : dict
        Dictionary containing processed audio signals
    aligned_onsets : dict
        Dictionary containing detected and aligned onset times
    is_failed : dict
        Quality check results including failure status and reason
    position_subplot : int
        Position index for this subplot
    config : class
        Configuration parameters
    """
    # Create subplot at specified position
    plt.subplot(config.PLOTS_TO_DISPLAY[0], config.PLOTS_TO_DISPLAY[1], position_subplot)
    
    # Get raw onset times
    Rraw = aligned_onsets['resp_onsets_detected']  # Response onsets
    Sraw = aligned_onsets['stim_onsets_detected']  # Stimulus onsets
    
    # Plot downsampled audio signal
    plt.plot(tt, audio_signals['rec_downsampled'], 'k-')
    
    # Add title with failure status
    message = "Recording: {} \n Failed = {}; Reason = {}".format(
        title_plot,
        is_failed['failed'],
        is_failed['reason'])
    plt.title(message)
    
    # Get maximum amplitude for scaling markers
    y = np.max(audio_signals['rec_downsampled'])
    
    # Plot detected markers
    plt.plot(aligned_onsets['markers_onsets_detected'] / 1000,
             y * config.EXTRACT_THRESH[1] * np.ones(np.size(aligned_onsets['markers_onsets_detected'])), 'mx')
    plt.plot(aligned_onsets['markers_onsets_aligned'] / 1000,
             config.EXTRACT_THRESH[0] * np.ones(np.size(aligned_onsets['markers_onsets_aligned'])), 'go')
    
    # Plot detected responses if any exist
    if len(Rraw) > 0:
        plt.plot(Rraw / 1000.0,
                 np.max(audio_signals['rec_tapping_clean']) * config.EXTRACT_THRESH[1] * np.ones(np.size(Rraw)), 'xr')
    
    # Plot detected stimuli
    plt.plot(Sraw / 1000.0,
             np.max(audio_signals['rec_tapping_clean']) * config.EXTRACT_THRESH[0] * np.ones(np.size(Sraw)), 'og')
    
    plt.xlim(mxlim)


def plot_alignment(mxlim, aligned_onsets, position_subplot, is_zoomed, config):
    """Plot alignment between stimulus and response onsets.
    
    Creates a subplot showing:
    - Detected marker positions
    - Raw stimulus and response onsets
    - Lines connecting aligned stimulus-response pairs
    - Phase and timing proximity indicators
    
    Parameters
    ----------
    mxlim : list
        [min_x, max_x] time limits for x-axis
    aligned_onsets : dict
        Dictionary containing detected and aligned onset times
    position_subplot : int
        Position index for this subplot
    is_zoomed : bool
        Whether to show zoomed view of a segment
    config : class
        Configuration parameters
    """
    # Create subplot
    plt.subplot(config.PLOTS_TO_DISPLAY[0], config.PLOTS_TO_DISPLAY[1], position_subplot)
    
    # Extract onset times
    Rraw = aligned_onsets['resp_onsets_detected']
    Sraw = aligned_onsets['stim_onsets_detected']
    stim_onsets_aligned = aligned_onsets['stim_onsets_aligned']
    resp_onsets_aligned = aligned_onsets['resp_onsets_aligned']
    num_aligned = np.sum(~np.isnan(resp_onsets_aligned))
    
    # Plot marker positions
    markers_onsets_detected = aligned_onsets['markers_onsets_detected']
    markers_onsets_aligned = aligned_onsets['markers_onsets_aligned']
    if len(markers_onsets_aligned) > 0:
        plt.plot(np.array(markers_onsets_aligned) / 1000.0, 0.25 * np.ones(np.size(markers_onsets_aligned)), 'dm')
    if len(markers_onsets_detected) > 0:
        plt.plot(np.array(markers_onsets_detected) / 1000.0, 0.75 * np.ones(np.size(markers_onsets_detected)), 'bs')

    # Get matching windows from config
    max_proximity_phase = config.ONSET_MATCHING_WINDOW_PHASE
    max_proximity = config.ONSET_MATCHING_WINDOW_MS

    # For each stimulus onset, find matching responses
    for j, _ in enumerate(Sraw):
        # Calculate inter-stimulus intervals
        if j == 0:
            stim_next = Sraw[j + 1] - Sraw[j]
            stim_last = stim_next
        elif (j + 1) == len(Sraw):
            stim_last = Sraw[j] - Sraw[j - 1]
            stim_next = stim_last
        else:
            stim_next = Sraw[j + 1] - Sraw[j]
            stim_last = Sraw[j] - Sraw[j - 1]

        # Check each response for potential matches
        for k, _ in enumerate(Rraw):
            stim_proposal = Sraw[j]
            resp_proposal = Rraw[k]

            # Calculate phase and timing differences
            if resp_proposal > stim_proposal:
                phase = (resp_proposal - stim_proposal) / stim_next
            else:
                phase = (stim_proposal - resp_proposal) / stim_last
            distance_ms = abs(stim_proposal - resp_proposal)
            
            # Plot connections for matches within windows
            if (distance_ms < max_proximity) and (phase < max(max_proximity_phase)) and (
                    phase > min(max_proximity_phase)):
                plt.plot([resp_proposal / 1000.0, stim_proposal / 1000.0], [0, 1], 'y--')
            if distance_ms < max_proximity:
                plt.plot(resp_proposal / 1000.0, 0, 'cx')
            if (phase < max(max_proximity_phase)) and (phase > min(max_proximity_phase)):
                plt.plot(resp_proposal / 1000.0, 0, 'bx')

    # Set plot limits and title based on zoom state
    if is_zoomed:
        min_x, max_x = find_min_max(Rraw, aligned_onsets, config)
        plt.xlim([min_x, max_x])
        plt.title("Zoomed view: Aligned onsets")
    else:
        plt.xlim(mxlim)
        plt.title("Aligned onsets Rraw={} Sraw={} aligned={}".format(len(Rraw), len(Sraw), num_aligned))

    # Plot final aligned pairs
    if len(Rraw) > 0:
        plt.plot(np.array(Rraw) / 1000.0, 0 * np.ones(np.size(Rraw)), '+k')
    plt.plot(np.array(Sraw) / 1000.0, 1 * np.ones(np.size(Sraw)), 'og')
    for ll in range(len(resp_onsets_aligned)):
        plt.plot([resp_onsets_aligned[ll] / 1000.0, stim_onsets_aligned[ll] / 1000.0], [0, 1], 'k-')
        plt.plot(resp_onsets_aligned[ll] / 1000.0, 0, 'rx')


def plot_markers_detection(mxlim, tt, audio_signals, aligned_onsets, analysis, position_subplot, config):
    """Plot marker signal detection results.
    
    Creates a subplot showing:
    - Final marker signal (blue)
    - Test signal (red)
    - Detection statistics
    
    Parameters
    ----------
    mxlim : list
        [min_x, max_x] time limits for x-axis
    tt : array
        Time points array
    audio_signals : dict
        Dictionary containing processed audio signals
    aligned_onsets : dict
        Dictionary containing detected and aligned onset times
    analysis : dict
        Analysis results including detection statistics
    position_subplot : int
        Position index for this subplot
    config : class
        Configuration parameters
    """
    plt.subplot(config.PLOTS_TO_DISPLAY[0], config.PLOTS_TO_DISPLAY[1], position_subplot)
    
    # Plot marker and test signals
    rec_markers_final = audio_signals['rec_markers_final']
    rec_test_final = audio_signals['rec_test_final']
    plt.plot(tt, rec_markers_final, 'b-')
    plt.plot(tt, rec_test_final, 'r-')
    
    # Calculate and display detection statistics
    percent_markers_detected = ((analysis["num_markers_detected"] / analysis["num_markers_onsets"]) * 100)
    message = 'Markers detection: \n {:2.0f}% markers detected ({} out of {})'.format(
        percent_markers_detected,
        analysis["num_markers_detected"],
        analysis["num_markers_onsets"])
    plt.title(message)
    
    # Add marker indicators
    y = np.max(rec_markers_final)
    plot_markers(y, aligned_onsets, config)
    plt.xlim(mxlim)


def plot_markers_cleaned(mxlim, tt, audio_signals, aligned_onsets, position_subplot, config):
    """Plot cleaned marker signal with SNR information.
    
    Creates a subplot showing:
    - Cleaned marker signal
    - Signal-to-noise ratio statistics
    - Marker positions
    
    Parameters
    ----------
    mxlim : list
        [min_x, max_x] time limits for x-axis
    tt : array
        Time points array
    audio_signals : dict
        Dictionary containing processed audio signals and SNR info
    aligned_onsets : dict
        Dictionary containing detected and aligned onset times
    position_subplot : int
        Position index for this subplot
    config : class
        Configuration parameters
    """
    plt.subplot(config.PLOTS_TO_DISPLAY[0], config.PLOTS_TO_DISPLAY[1], position_subplot)
    
    # Get SNR information and plot cleaned signal
    snr_info = audio_signals['snr_info']
    plt.plot(tt, audio_signals['rec_markers_clean'], 'b-')
    
    # Display SNR statistics
    message = 'Markers cleaning signal: \n SNR: {:2.2f}; max_ratio: {:2.2f}; mean_ratio: {:2.2f}'.format(
        snr_info['snr'],
        snr_info['max_ratio'],
        snr_info['mean_ratio'])
    plt.title(message)
    
    # Add marker indicators
    y = max(audio_signals['rec_markers_clean'])
    plot_markers(y, aligned_onsets, config)
    plt.xlim(mxlim)


def plot_markers_zoomed(tt, audio_signals, aligned_onsets, position_subplot, config):
    # plot markers detection plot: zoomed
    plt.subplot(config.PLOTS_TO_DISPLAY[0], config.PLOTS_TO_DISPLAY[1], position_subplot)
    plt.plot(tt, audio_signals['rec_markers_final'], 'b-')
    plt.plot(tt, audio_signals['rec_test_final'], 'r-')
    message = 'Zoomed view on markers: (test signal in red)'
    plt.title(message)
    y = np.max(audio_signals['rec_markers_final'])
    plot_markers(y, aligned_onsets, config)
    mn1 = min([min(aligned_onsets['markers_onsets_detected']), min(aligned_onsets['markers_onsets_detected'])]) / 1000.0
    mx1 = mn1 + 0.8
    plt.xlim([mn1 - 0.1, mx1 + 0.1])


def plot_markers_zoomed_cleaned(tt, audio_signals, aligned_onsets, position_subplot, config):
    # plot markers detection plot: zoomed (cleaned signal)
    plt.subplot(config.PLOTS_TO_DISPLAY[0], config.PLOTS_TO_DISPLAY[1], position_subplot)
    plt.plot(tt, audio_signals['rec_markers_clean'], 'b-')
    message = 'Zoomed view on cleaned markers'
    plt.title(message)
    y = max(audio_signals['rec_markers_clean'])
    plot_markers(y, aligned_onsets, config)
    mn1 = min([min(aligned_onsets['markers_onsets_detected']), min(aligned_onsets['markers_onsets_detected'])]) / 1000.0
    mx1 = mn1 + 0.8
    plt.xlim([mn1 - 0.1, mx1 + 0.1])


def plot_markers_error(mxlim, aligned_onsets, analysis, position_subplot, config):
    # plot markers error
    plt.subplot(config.PLOTS_TO_DISPLAY[0], config.PLOTS_TO_DISPLAY[1], position_subplot)
    e_marker_ideal = aligned_onsets['verify_asynchrony']
    s_marker_ideal = aligned_onsets['verify_stim_ioi']
    S_marker_ideal = aligned_onsets['verify_stim_shifted']
    R_marker_ideal = aligned_onsets['verify_resp_shifted']
    plt.plot(S_marker_ideal / 1000.0, s_marker_ideal, 'sg')
    plt.plot(R_marker_ideal / 1000.0, s_marker_ideal + e_marker_ideal, 'rx')
    t = s_marker_ideal[~np.isnan(s_marker_ideal)]
    if len(aligned_onsets['markers_onsets_detected']) > 0 and len(t) > 0:
        plt.plot(aligned_onsets['markers_onsets_detected'] / 1000.0,
                 0.95 * np.min(t[t < 1000]) * np.ones(np.size(aligned_onsets['markers_onsets_detected'])), 'sm')
        plt.ylim([0.9 * min(t[t < 1000]), 1.05 * max(t[t < 1000])])
    if len(t) == 0:
        analysis["markers_max_difference"] = 9999
    message = 'Markers timing accuracy: ' + analysis['markers_status'] + '\n Error: {:2.2f} ms'.format(
        analysis["markers_max_difference"])
    plt.title('Legend inside')
    plt.title(message)
    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    plt.xlim(mxlim)


def plot_tapping_rec(tt, mxlim, audio_signals, aligned_onsets, analysis, position_subplot, config):
    # plot tapping recording
    plt.subplot(config.PLOTS_TO_DISPLAY[0], config.PLOTS_TO_DISPLAY[1], position_subplot)
    rec_tap_final = audio_signals['rec_tapping_clean']
    Rraw = aligned_onsets['resp_onsets_detected']
    Sraw = aligned_onsets['stim_onsets_detected']
    plt.plot(tt, rec_tap_final, 'k-')
    message = 'Tapping detection: \n {:2.0f} of {:2.0f} stim onsets ({:2.2f}% ratio)\n {:2.2f}% bad taps'.format(
        analysis["num_resp_raw_all"],
        analysis["num_stim_raw_all"],
        analysis["ratio_resp_to_stim"],
        analysis["percent_of_bad_taps_all"])
    plt.title(message)
    if len(Rraw) > 0:
        plt.plot(Rraw / 1000.0, np.max(rec_tap_final) * config.EXTRACT_THRESH[1] * np.ones(np.size(Rraw)), 'xr')
    plt.plot(Sraw / 1000.0, np.max(rec_tap_final) * config.EXTRACT_THRESH[0] * np.ones(np.size(Sraw)), 'og')
    plt.plot(aligned_onsets['markers_onsets_aligned'] / 1000,
             config.EXTRACT_THRESH[0] * np.ones(np.size(aligned_onsets['markers_onsets_aligned'])), 'gs')
    plt.xlim(mxlim)


def plot_tapping_zoomed(tt, audio_signals, aligned_onsets, analysis, position_subplot, config):
    # plot tapping response: zoomed
    plt.subplot(config.PLOTS_TO_DISPLAY[0], config.PLOTS_TO_DISPLAY[1], position_subplot)
    rec_tap_final = audio_signals['rec_tapping_clean']
    Rraw = aligned_onsets['resp_onsets_detected']
    Sraw = aligned_onsets['stim_onsets_detected']
    plt.plot(tt, rec_tap_final, 'k-')
    message = 'Zoomed view on tapping: \n {} aligned resp onsets out of {} stim onsets ({:2.2f}%)'.format(
        analysis["num_resp_aligned_all"],
        analysis["num_stim_aligned_all"],
        analysis["percent_resp_aligned_all"])
    plt.title(message)
    if len(Rraw) > 0:
        plt.plot(Rraw / 1000.0, np.max(rec_tap_final) * config.EXTRACT_THRESH[1] * np.ones(np.size(Rraw)), 'xr')
    plt.plot(Sraw / 1000.0, np.max(rec_tap_final) * config.EXTRACT_THRESH[0] * np.ones(np.size(Sraw)), 'og')
    plt.plot(aligned_onsets['markers_onsets_aligned'] / 1000,
             config.EXTRACT_THRESH[0] * np.ones(np.size(aligned_onsets['markers_onsets_aligned'])), 'gs')
    min_x, max_x = find_min_max(Rraw, aligned_onsets, config)
    plt.xlim([min_x, max_x])


def plot_asynchrony(mxlim, aligned_onsets, analysis, position_subplot, config):
    """Plot tapping asynchrony analysis.
    
    Creates a subplot showing:
    - Asynchrony measurements over time
    - Mean asynchrony line
    - Standard deviation bounds
    - Matching windows
    
    Parameters
    ----------
    mxlim : list
        [min_x, max_x] time limits for x-axis
    aligned_onsets : dict
        Dictionary containing aligned onset times and asynchrony data
    analysis : dict
        Analysis results including mean and SD of asynchronies
    position_subplot : int
        Position index for this subplot
    config : class
        Configuration parameters
    """
    plt.subplot(config.PLOTS_TO_DISPLAY[0], config.PLOTS_TO_DISPLAY[1], position_subplot)
    
    # Extract timing data
    resp_onsets_aligned = aligned_onsets['resp_onsets_aligned']
    asynchrony = aligned_onsets['asynchrony']
    verify_asynchrony = aligned_onsets['verify_asynchrony']
    verify_markers_shifted = aligned_onsets['verify_stim_shifted']
    markers_onsets_aligned = aligned_onsets['markers_onsets_aligned']
    mean_all = analysis["mean_async_all"]
    std_all = analysis["sd_async_all"]

    # Plot verification points
    plt.plot(verify_markers_shifted / 1000.0, verify_asynchrony, 'sg')
    
    # Plot matching window boundaries
    plt.plot([min(markers_onsets_aligned) / 1000.0, max(markers_onsets_aligned) / 1000.0],
             [config.MARKERS_MATCHING_WINDOW, config.MARKERS_MATCHING_WINDOW], 'g--')
    plt.plot([min(markers_onsets_aligned) / 1000.0, max(markers_onsets_aligned) / 1000.0],
             [-config.MARKERS_MATCHING_WINDOW, -config.MARKERS_MATCHING_WINDOW], 'g--')
    
    # Plot mean and SD if valid responses exist
    if sum(~np.isnan(resp_onsets_aligned)) > 0:
        if mean_all < 999:  # Valid mean exists
            # Plot mean line
            plt.plot([min(markers_onsets_aligned) / 1000.0, max(markers_onsets_aligned) / 1000.0], 
                    [mean_all, mean_all], 'm-')
            
            # Plot standard deviation bounds
            if std_all > 0:
                plt.plot([min(markers_onsets_aligned) / 1000.0, max(markers_onsets_aligned) / 1000.0],
                         [mean_all + std_all, mean_all + std_all], 'y-')
                plt.plot([min(markers_onsets_aligned) / 1000.0, max(markers_onsets_aligned) / 1000.0],
                         [mean_all - std_all, mean_all - std_all], 'y-')
        
        # Plot onset matching windows if within reasonable range
        if config.ONSET_MATCHING_WINDOW_MS < 500:
            valid_resp = resp_onsets_aligned[~np.isnan(resp_onsets_aligned)]
            plt.plot([min(valid_resp) / 1000.0, max(valid_resp) / 1000.0],
                     [config.ONSET_MATCHING_WINDOW_MS, config.ONSET_MATCHING_WINDOW_MS], 'r--')
            plt.plot([min(valid_resp) / 1000.0, max(valid_resp) / 1000.0],
                     [-config.ONSET_MATCHING_WINDOW_MS, -config.ONSET_MATCHING_WINDOW_MS], 'r--')
    
    # Plot asynchrony measurements
    plt.plot(resp_onsets_aligned / 1000.0, asynchrony, 'xr-')
    
    # Add title with statistics
    message = 'Tapping asynchrony: \n M = {:2.2f}ms; SD = {:2.2f}ms'.format(mean_all, std_all)
    plt.title(message)
    plt.xlim(mxlim)


def plot_asynch_and_ioi(mxlim, aligned_onsets, analysis, position_subplot, config):
    """Plot asynchrony and inter-onset intervals.
    
    Creates a subplot showing:
    - Stimulus IOIs for played and unplayed stimuli
    - Response asynchronies
    - Statistics for played vs unplayed conditions
    
    Parameters
    ----------
    mxlim : list
        [min_x, max_x] time limits for x-axis
    aligned_onsets : dict
        Dictionary containing aligned onset times and IOI data
    analysis : dict
        Analysis results including condition-specific statistics
    position_subplot : int
        Position index for this subplot
    config : class
        Configuration parameters
    """
    plt.subplot(config.PLOTS_TO_DISPLAY[0], config.PLOTS_TO_DISPLAY[1], position_subplot)
    
    # Extract timing data
    stim_onsets_aligned = aligned_onsets['stim_onsets_aligned']
    stim_ioi = aligned_onsets['stim_ioi']
    is_played = aligned_onsets['stim_onsets_is_played']
    resp_onsets_aligned = aligned_onsets['resp_onsets_aligned']
    asynchrony = aligned_onsets['asynchrony']
    
    # Get statistics for played/unplayed conditions
    mean_async_played = analysis["mean_async_played"]
    sd_async_played = analysis["sd_async_played"]
    mean_async_notplayed = np.round(analysis["mean_async_notplayed"], 2)
    sd_async_notplayed = np.round(analysis["sd_async_notplayed"], 2)
    
    # Plot IOIs and asynchronies
    plt.plot(stim_onsets_aligned / 1000.0, stim_ioi, 'bs-')  # All stimuli
    plt.plot(stim_onsets_aligned[~is_played] / 1000.0, stim_ioi[~is_played], 'cs-')  # Unplayed stimuli
    plt.plot(resp_onsets_aligned / 1000.0, stim_ioi + asynchrony, 'xr-')  # Responses
    
    # Handle invalid statistics for unplayed condition
    if mean_async_notplayed == 999:
        mean_async_notplayed = "na"
        sd_async_notplayed = "na"
    
    # Add title with condition statistics
    message = ('Onset is played: {:2.0f}% tapping (M = {:2.2f}; SD = {:2.2f}) \n'
              'Onset is not played: {:2.0f}% tapping (M = {}; SD = {})').format(
        analysis["percent_response_aligned_played"],
        mean_async_played,
        sd_async_played,
        analysis["percent_response_aligned_notplayed"],
        mean_async_notplayed,
        sd_async_notplayed)
    plt.title(message)
    plt.xlim(mxlim)


def plot_markers(y, aligned_onsets, config):
    """Plot marker indicators at specified vertical positions.
    
    Parameters
    ----------
    y : float
        Base vertical position for markers
    aligned_onsets : dict
        Dictionary containing marker onset times
    config : class
        Configuration parameters for marker thresholds
    """
    # Plot detected markers at upper threshold
    plt.plot(aligned_onsets['markers_onsets_detected'] / 1000,
             y * config.EXTRACT_THRESH[1] * np.ones(np.size(aligned_onsets['markers_onsets_detected'])), 'mx')
    
    # Plot detected markers at lower threshold
    plt.plot(aligned_onsets['markers_onsets_detected'] / 1000,
             y * config.EXTRACT_THRESH[0] * np.ones(np.size(aligned_onsets['markers_onsets_detected'])), 'ms')
    
    # Plot aligned markers
    plt.plot(aligned_onsets['markers_onsets_aligned'] / 1000,
             y * config.EXTRACT_THRESH[0] * np.ones(np.size(aligned_onsets['markers_onsets_aligned'])), 'go')


def find_min_max(Rraw, aligned_onsets, config):
    """Calculate appropriate x-axis limits for zoomed plots.
    
    Parameters
    ----------
    Rraw : array
        Raw response onset times
    aligned_onsets : dict
        Dictionary containing marker onset times
    config : class
        Configuration parameters
    
    Returns
    -------
    min_x : float
        Minimum x-axis value
    max_x : float
        Maximum x-axis value
    """
    # Set default window
    min_x = config.STIM_BEGINNING / 1000
    max_x = min_x + 5.0
    
    # Adjust window based on number of responses
    if len(Rraw) > 2:
        min_x = min(Rraw / 1000) - 0.5
        max_x = max(Rraw / 1000) + 0.5
    if len(Rraw) > 4:
        min_x = (Rraw[2] / 1000) - 0.5
        max_x = (Rraw[4] / 1000) + 0.5
    
    # Ensure window doesn't exceed marker bounds
    max_x = min(max_x, max(aligned_onsets['markers_onsets_detected'] / 1000))
    max_x = min(max_x, min_x + 10)  # Maximum 10-second window
    
    return min_x, max_x