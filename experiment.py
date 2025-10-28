# imports
import json
import tempfile
from functools import cache
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import os

from markupsafe import Markup
from repp.analysis import REPPAnalysis
from repp.config import sms_tapping
from repp.stimulus import REPPStimulus
from repp.utils import save_json_to_file, save_samples_to_file

import psynet.experiment
from psynet.asset import CachedFunctionAsset, LocalStorage 
from psynet.consent import NoConsent
from psynet.modular_page import AudioPrompt, AudioRecordControl, ModularPage
from psynet.page import InfoPage, SuccessfulEndPage
from psynet.timeline import ProgressDisplay, ProgressStage, Timeline, join
from psynet.trial.audio import AudioRecordTrial
from psynet.trial.static import StaticNode, StaticTrial, StaticTrialMaker

# Import enhanced analysis functions from separate module
from .repp_beatfinding import (
    enhanced_tapping_analysis,
)

# repp
from .repp_prescreens import (
    NumpySerializer,
    REPPMarkersTest,
    REPPTappingCalibration,
    REPPVolumeCalibrationMusic,
)

########################################################################################################################
# SETUP
########################################################################################################################
DEBUG = True

# recruitment
RECRUITER = "prolific" # prolific vs hotair
INITIAL_RECRUITMENT_SIZE = 10
AUTO_RECRUIT = False 
NUM_PARTICIPANTS = 20
NUM_TRIALS_PER_PARTICIPANT = 2

# time estimates
DURATION_ESTIMATED_TRIAL = 40

# failing criteria
MIN_RAW_TAPS = 5 # TO BE DECIDED
MAX_RAW_TAPS = 500 # TO BE DECIDED


def get_prolific_settings():
    with open("qualification_prolific_en.json", "r") as f:
        qualification = json.dumps(json.load(f))
    return {
        "recruiter": RECRUITER,
        # "id": "singing-nets",
        "prolific_estimated_completion_minutes": 14,
        "prolific_recruitment_config": qualification,
        "base_payment": 2.1,
        "auto_recruit": False,
        "currency": "£",
        "wage_per_hour": 0.01
    }


########################################################################################################################
# TODOS
########################################################################################################################
# TODO: Output: make sure I get the right output, including tapping onsets raw and aligned ones
# TODO: Double check failing criteria
# TODO: add good instructions
# TODO: add practice
# TODO: add pre-screens.
# TODO: add harin's music.


########################################################################################################################
# TAPPING ANALYSIS
########################################################################################################################
# Note: Enhanced analysis functions are now imported from enhanced_tapping_analysis module
# This eliminates code duplication and improves maintainability


########################################################################################################################
# Stimuli
########################################################################################################################
# Isochronus stimuli
tempo_800_ms = [800] * 15 # ISO 800ms
tempo_600_ms = [600] * 12 # ISO 600ms

iso_stimulus_onsets = [tempo_800_ms, tempo_600_ms]
iso_stimulus_names = ["iso_800ms", "iso_600ms"]


@cache
def create_iso_stim_with_repp(stim_name, stim_ioi):
    stimulus = REPPStimulus(stim_name, config=sms_tapping)
    stim_onsets = stimulus.make_onsets_from_ioi(stim_ioi)
    stim_prepared, stim_info, _ = stimulus.prepare_stim_from_onsets(stim_onsets)
    info = json.dumps(stim_info, cls=NumpySerializer)
    return stim_prepared, info

def generate_iso_stimulus_audio(path, stim_name, list_iois):
    stim_prepared, info = create_iso_stim_with_repp(stim_name, tuple(list_iois))
    save_samples_to_file(stim_prepared, path, sms_tapping.FS)
    
def generate_iso_stimulus_info(path, stim_name, list_iois):
    stim_prepared, info = create_iso_stim_with_repp(stim_name, tuple(list_iois))
    save_json_to_file(info, path)


nodes_iso = [
    StaticNode(
        definition={
            "stim_name": name,
            "list_iois": iois,
        },
        assets={
            "stimulus_audio": CachedFunctionAsset(generate_iso_stimulus_audio),
            "stimulus_info": CachedFunctionAsset(generate_iso_stimulus_info),
        },
    )
    for name, iois in zip(iso_stimulus_names, iso_stimulus_onsets)
]

nodes_silent = [
    StaticNode(
        definition={
            "stim_name": name,
            "list_iois": iois,
        },
        assets={
            "stimulus_audio": CachedFunctionAsset(generate_iso_stimulus_audio),
            "stimulus_info": CachedFunctionAsset(generate_iso_stimulus_info),
        },
    )
    for name, iois in zip(iso_stimulus_names, iso_stimulus_onsets)
]


# Music stimuli for beat-finding task (no onsets required)
music_stimulus_name = ["track1", "track2"]
music_stimulus_audio = ["music/0R8IbpKXavM.wav", "music/ehHKH5PbGYc.wav"]


def load_audio_only_from_file(fs, audio_filename):
    """
    Load audio file without requiring onsets file.
    
    Parameters
    ----------
    fs : int
        Target sampling frequency in Hz
    audio_filename : str
        Path to audio file
        
    Returns
    -------
    np.ndarray
        Loaded and resampled audio data
    """
    stimulus = REPPStimulus("temp", config=sms_tapping)
    return stimulus.load_resample_file(fs, audio_filename)


def filter_and_add_markers_no_onsets(stim, config):
    """
    Apply filtering and add markers without requiring onset information.
    
    Parameters
    ----------
    stim : np.ndarray
        Raw audio stimulus data
    config : Config
        Configuration parameters
        
    Returns
    -------
    tuple[np.ndarray, dict]
        - Prepared stimulus array
        - Dictionary containing stimulus information
    """
    stimulus = REPPStimulus("temp", config=config)
    
    # Apply spectral filtering
    filtered_stim = stimulus.filter_stim(
        config.FS, stim, config.STIM_RANGE, config.STIM_AMPLITUDE
    )
    
    # Create marker sounds
    markers_sound = stimulus.make_markers_sound(
        config.FS,
        config.MARKERS_DURATION,
        config.MARKERS_ATTACK,
        config.MARKERS_RANGE,
        config.MARKERS_AMPLITUDE
    )
    
    # Add markers at beginning and end
    markers_onsets, markers_channel = stimulus.add_markers_sound(
        config.FS,
        stim,
        config.MARKERS_IOI,
        config.MARKERS_BEGINNING,
        config.MARKERS_END,
        config.STIM_BEGINNING,
        config.MARKERS_END_SLACK
    )
    
    # Combine markers with filtered stimulus
    stim_prepared = stimulus.put_clicks_in_audio(markers_channel, config.FS, markers_sound, markers_onsets)
    stim_start_samples = int(round(config.STIM_BEGINNING * config.FS / 1000.0))
    stim_prepared[stim_start_samples:(stim_start_samples + len(filtered_stim))] += filtered_stim
    
    stim_duration = len(stim_prepared) / config.FS
    
    # Create minimal stim_info for beat-finding task
    stim_info = {
        'stim_duration': stim_duration,
        'stim_onsets': [],  # Empty for beat-finding
        'stim_shifted_onsets': [],  # Empty for beat-finding
        'onset_is_played': np.array([]),  # Empty for beat-finding
        'markers_onsets': markers_onsets,
        'stim_name': 'beat_finding_music'
    }
    
    return stim_prepared, stim_info


@cache
def create_music_stim_with_repp_beat_finding(stim_name, audio_filename, fs=44100):
    """
    Create music stimulus for beat-finding task without requiring onsets file.
    """
    # Load audio file
    stim = load_audio_only_from_file(fs, audio_filename)
    
    # Convert stereo to mono if needed
    if len(stim.shape) == 2:
        stim = stim[:, 0]
    
    # Apply filtering and add markers
    stim_prepared, stim_info = filter_and_add_markers_no_onsets(stim, sms_tapping)
    stim_info["stim_name"] = stim_name
    
    info = json.dumps(stim_info, cls=NumpySerializer)
    return stim_prepared, info


def generate_music_stimulus_audio(path, stim_name, audio_filename):
    stim_prepared, _ = create_music_stim_with_repp_beat_finding(stim_name, audio_filename)
    save_samples_to_file(stim_prepared, path, sms_tapping.FS)
    
def generate_music_stimulus_info(path, stim_name, audio_filename):
    stim_prepared, info = create_music_stim_with_repp_beat_finding(stim_name, audio_filename)
    save_json_to_file(info, path)


nodes_music = [
    StaticNode(
        definition={
            "stim_name": name,
            "audio_filename": audio,
        },
        assets={
            "stimulus_audio": CachedFunctionAsset(generate_music_stimulus_audio),
            "stimulus_info": CachedFunctionAsset(generate_music_stimulus_info),
        },
    )
     for name, audio in zip(music_stimulus_name, music_stimulus_audio)
]


########################################################################################################################
# Experiment parts
########################################################################################################################
class TapTrialAnalysis(AudioRecordTrial, StaticTrial):
    def get_info(self):
        with tempfile.NamedTemporaryFile() as f:
            self.assets["stimulus_info"].export(f.name)
            with open(f.name, "r") as reader:
                return json.loads(
                    json.load(reader)
                )  # For some reason REPP double-JSON-encodes its output

    def analyze_recording(self, audio_file: str, output_plot: str):
        info = self.get_info()
        stim_name = info["stim_name"]
        title_in_graph = "Participant {}".format(self.participant_id)
        
        # Use enhanced analysis instead of basic analysis
        # Pass the stimulus info to the analysis function
        _, extracted_onsets, stats = enhanced_tapping_analysis(
            audio_file, title_in_graph, output_plot, stim_info=info)
        
        # Extract the quality results from stats
        is_failed = stats.get("failed", True)
        reason = stats.get("reason", "Analysis failed")

        extracted_onsets_json = json.dumps(extracted_onsets, cls=NumpySerializer)
        stats = json.dumps(stats, cls=NumpySerializer)
        
        return {
            "failed": is_failed,
            "reason": reason,
            "extracted_onsets": extracted_onsets_json,
            "stats": stats,
            "stim_name": stim_name,
        }

class TapTrial(TapTrialAnalysis):
    def show_trial(self, experiment, participant):
        info = self.get_info()
        duration_rec = info["stim_duration"]
        trial_number = self.position + 1
        return ModularPage(
            "trial_main_page",
            AudioPrompt(
                self.assets["stimulus_audio"].url,
                Markup(
                    f"""
                    <br><h3>Tap in time to the musical beat.</h3>
                    Trial number {trial_number} out of {NUM_TRIALS_PER_PARTICIPANT}  trials.
                    """
                ),
            ),
            AudioRecordControl(
                duration=duration_rec,
                show_meter=False,
                controls=False,
                auto_advance=False,
                bot_response_media=self.get_bot_response_media(),
            ),
            time_estimate=duration_rec + 5,
            progress_display=ProgressDisplay(
                show_bar=True,  # set to False to hide progress bar in movement
                stages=[
                    ProgressStage(
                        3.5,
                        "Wait in silence...",
                        "red",
                    ),
                    ProgressStage(
                        [3.5, (duration_rec - 6)],
                        "START TAPPING!",
                        "green",
                    ),
                    ProgressStage(
                        3.5,
                        "Stop tapping and wait in silence...",
                        "red",
                        persistent=False,
                    ),
                    ProgressStage(
                        0.5,
                        "Press Next when you are ready to continue...",
                        "orange",
                        persistent=True,
                    ),
                ],
            ),
        )

    def get_bot_response_media(self):
        raise NotImplementedError


class TapTrialISO(TapTrial):
    time_estimate = DURATION_ESTIMATED_TRIAL

    def get_bot_response_media(self):
        return {
            "iso_800ms": "boot_responses/example_iso_slow_tap.wav",
            "iso_600ms": "boot_responses/example_iso_fast_tap.wav",
        }[self.definition["stim_name"]]


class TapTrialExplore(TapTrialAnalysis):
    time_estimate = DURATION_ESTIMATED_TRIAL
    
    def show_trial(self, experiment, participant):
        info = self.get_info()
        duration_rec = info["stim_duration"]
        trial_number = self.position + 1
        return ModularPage(
            "trial_main_page",
            AudioPrompt(
                "", # TODO: add silent audio file
                Markup(
                    f"""
                    <br><h3>Explore the hidden rhythm by tapping.</h3>
                    Trial number {trial_number} out of {NUM_TRIALS_PER_PARTICIPANT}  trials.
                    """
                ),
            ),
            AudioRecordControl(
                duration=duration_rec,
                show_meter=False,
                controls=False,
                auto_advance=False,
                bot_response_media=self.get_bot_response_media(),
            ),
            time_estimate=duration_rec + 5,
            progress_display=ProgressDisplay(
                show_bar=True,  # set to False to hide progress bar in movement
                stages=[
                    ProgressStage(
                        3.5,
                        "Wait in silence...",
                        "red",
                    ),
                    ProgressStage(
                        [3.5, (duration_rec - 6)],
                        "START TAPPING!",
                        "green",
                    ),
                    ProgressStage(
                        3.5,
                        "Stop tapping and wait in silence...",
                        "red",
                        persistent=False,
                    ),
                    ProgressStage(
                        0.5,
                        "Press Next when you are ready to continue...",
                        "orange",
                        persistent=True,
                    ),
                ],
            ),
        )

    def get_bot_response_media(self): # TODO: add silent example for bot response
        return {
            "iso_800ms": "boot_responses/example_iso_slow_tap.wav",
            "iso_600ms": "boot_responses/example_iso_fast_tap.wav",
        }[self.definition["stim_name"]]



class TapTrialMusic(TapTrial):
    time_estimate = DURATION_ESTIMATED_TRIAL

    def get_bot_response_media(self):
        return {
            "track1": "boot_responses/example_music_tapping_track_1.wav",
            "track2": "boot_responses/example_music_tapping_track_7.wav",
        }[self.definition["stim_name"]]


def welcome():
    return InfoPage(
        Markup(
            """
            <h3>Welcome</h3>
            <hr>
            In this experiment, you will hear music and be asked to tap in time to the beat of the music by tapping with your finger.
            <br><br>
            We will monitor your responses throughout the experiment.
            <br><br>
            Press <b><b>next</b></b> when you are ready to start.
            <hr>
            """
        ),
        time_estimate=3
    )


# Tapping tasks
ISO_tapping = join(
    InfoPage(
        Markup(
            """
            <h3>Tapping to Rhythms</h3>
            <hr>
            In each trial, you will hear a rhythm playing at a constant pace.
            <br><br>
            <b><b>Your goal is to tap in time with the beat of the rhythm.</b></b> <br><br>
            Note:
            <ul>
                <li>Start tapping as soon as the rhythm starts and continue tapping until the rhythm ends.</li>
                <li>At the beginning and end of each rhythm, you will hear three consecutive beeps.</li>
                <li>Do not tap during these beeps, as they signal the beginning and end of each rhythm.</li>
            </ul>
            <br>
            <hr>
            """
        ),
        time_estimate=10,
    ),
    StaticTrialMaker(
        id_="ISO_tapping",
        trial_class=TapTrialISO,
        nodes=nodes_iso,
        expected_trials_per_participant=len(nodes_iso),
        target_n_participants=NUM_PARTICIPANTS,
        recruit_mode="n_participants",
        check_performance_at_end=False,
    ),
)


silent_tapping = join(
    InfoPage(
        Markup(
            """
            <h3>NEW TASK</h3>
            <hr>
            EXPLAIN HERE INSTRUCTION OF NEW TASK: FIND THE HIDDEN RHYTHM BY TAPPING!
            <hr>
            """
        ),
        time_estimate=10,
    ),
    StaticTrialMaker(
        id_="silent_tapping",
        trial_class=TapTrialExplore,
        nodes=nodes_silent,
        expected_trials_per_participant=len(nodes_silent),
        target_n_participants=NUM_PARTICIPANTS,
        recruit_mode="n_participants",
        check_performance_at_end=False,
    ),
)


music_tapping = join(
    InfoPage(
        Markup(
            """
        <h3>Tapping to Music</h3>
        <hr>
        You will now listen to music.
        <br><br>
        <b><b>Your goal is to tap in time with the beat of the music until the music ends.</b></b>
        <hr>
        """
        ),
        time_estimate=5,
    ),
    StaticTrialMaker(
        id_="music_tapping",
        trial_class=TapTrialMusic,
        nodes=nodes_music,
        expected_trials_per_participant=len(nodes_music),
        target_n_participants=NUM_PARTICIPANTS,
        recruit_mode="n_participants",
        check_performance_at_end=False,
    ),
)


########################################################################################################################
# Timeline
########################################################################################################################
class Exp(psynet.experiment.Experiment):
    label = "Tapping Experiment"
    asset_storage = LocalStorage()

    config = {
        **get_prolific_settings(),
        "initial_recruitment_size": INITIAL_RECRUITMENT_SIZE,
        "auto_recruit": AUTO_RECRUIT, 
        "title": "Tapping experiment (Chrome browser, ~14 mins)",
        "description": "This is a tapping experiment. You will be asked to listen to rhythms and synchronize to the beat by tapping with your finger.",
        "contact_email_on_error": "m.angladatort@gold.ac.uk",
        "organization_name": "Max Planck Institute for Empirical Aesthetics",
        "show_reward": False
    }

    if DEBUG:
        timeline = Timeline(
            NoConsent(),
            welcome(),
            # REPPVolumeCalibrationMusic(),
            ISO_tapping,
            silent_tapping,
            # music_tapping,
            SuccessfulEndPage(),
        )
    else:
        timeline = Timeline(
            NoConsent(),
            welcome(),
            REPPVolumeCalibrationMusic(),  # calibrate volume with music
            REPPMarkersTest(),  # pre-screening filtering participants based on recording test (markers)
            REPPTappingCalibration(),  # calibrate tapping
            ISO_tapping,
            music_tapping,
            SuccessfulEndPage(),
        )

    # def __init__(self, session=None):
    #     super().__init__(session)
    #     self.initial_recruitment_size = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)