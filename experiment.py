# imports
import json
import tempfile
from functools import cache
import numpy as np
import math
from numpy.linalg import norm
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
from psynet.asset import CachedFunctionAsset, ExperimentAsset, LocalStorage
from psynet.consent import NoConsent
from psynet.modular_page import AudioPrompt, AudioRecordControl, ModularPage, SurveyJSControl
from psynet.page import InfoPage, SuccessfulEndPage
from psynet.timeline import Module, ProgressDisplay, ProgressStage, Timeline, join, randomize, while_loop
from psynet.trial.audio import AudioRecordTrial
from psynet.trial.static import StaticNode, StaticTrial, StaticTrialMaker


# Import beat detection analysis functions from separate module
from .repp_beatfinding.beat_detection import do_beat_detection_analysis

# repp

from .repp_prescreens import (
    NumpySerializer,
    REPPMarkersTest,
    REPPTappingCalibration,
    REPPVolumeCalibrationMusic,
)

class REPPMarkersTestRepeat(REPPMarkersTest):
    label = "repp_markers_test_repeat"

########################################################################################################################
# TODOS
########################################################################################################################
# TODO: Remove next button after tapping trial

########################################################################################################################
# SETUP
########################################################################################################################
DEBUG = True
MAIN_TASK_ORDER = "punishment_first"  # change: "reward_first" or "punishment_first"
TARGET_NODE_1 = "323" # first target
TARGET_NODE_2 = "332" # second target

# recruitment
RECRUITER = "prolific" # prolific vs hotair vs generic
INITIAL_RECRUITMENT_SIZE = 10 # N  people to recruit initially
NUM_PARTICIPANTS = 20 # N people to run the experiment
AUTO_RECRUIT = False # Keep recruiting until we have NUM_PARTICIPANTS participants

# N trials
NUM_TRIALS_PER_PARTICIPANT_ISO = 2 # practice trials

# Per trial maker: 1 target × N trials (participant repeats the same trial N times)
NUM_TRIALS_PER_TARGET = 25  # 25 repetitions of the same trial within each trial maker
    
DURATION_ESTIMATED_TRIAL = 10 # estimated duration of each trial in seconds

# failing criteria (only allow 4 taps)
TARGET_NUM_TAPS = 4
MIN_RAW_TAPS = 4
MAX_RAW_TAPS = 4

# Config wrapper to include MIN_RAW_TAPS and MAX_RAW_TAPS for beat detection analysis
class ConfigWithThresholds:
    """Wrapper around sms_tapping config that adds MIN_RAW_TAPS and MAX_RAW_TAPS attributes."""
    def __init__(self, base_config):
        # Copy all attributes from base_config
        for attr in dir(base_config):
            if not attr.startswith('_'):
                try:
                    setattr(self, attr, getattr(base_config, attr))
                except AttributeError:
                    pass  # Skip attributes that can't be read
        # Add custom thresholds from experiment settings
        self.MIN_RAW_TAPS = MIN_RAW_TAPS
        self.MAX_RAW_TAPS = MAX_RAW_TAPS


def get_prolific_settings():
    with open("qualification_prolific_en.json", "r") as f:
        qualification = json.dumps(json.load(f))
    return {
        "recruiter": RECRUITER,
        # "id": "singing-nets",
        "prolific_estimated_completion_minutes": 14,
        "prolific_recruitment_config": qualification,
        "base_payment": 2.1,
        "currency": "£",
        "wage_per_hour": 0.01,
        # "prolific_workspace": "Goldsmiths",
        # "prolific_project": "Pilot",
    }


########################################################################################################################
########################################################################################################################
# Scoring function
########################################################################################################################
def to_simplex(vector):
    total = sum(vector)
    if total == 0:
        return [0 for _ in vector]
    return [v / total for v in vector]


def reward_scoring_function(target, tapping_iois):
    perc = 0.3
    D = 1.55
    a = 15
    b = 45

    # Need at least 3 IOIs to compare against a 3-element target rhythm
    if len(tapping_iois) < 3 or len(target) < 3:
        return 0

    # Use the first 3 IOIs if more were recorded
    tapping_iois = tapping_iois[:3]
    target = target[:3]

    target_simplex = to_simplex(target)
    tapping_simplex = to_simplex(tapping_iois)
    error = norm([tapping_simplex[i] - target_simplex[i] for i in range(3)])
    
    score = math.exp(-error * perc) * 100
    final_score = a + (score - b) * D
    final_score = max(0, min(100, final_score))

    return round(final_score)



def punishment_scoring_function(target, tapping_iois):
    reward_score = reward_scoring_function(target, tapping_iois)
    punishment_score = reward_score - 100
    punishment_score = max(-100, min(0, punishment_score))
    return round(punishment_score)


def scoring_function(target, tapping_iois, mode):
    if mode == "reward":
        return reward_scoring_function(target, tapping_iois)
    elif mode == "punishment":
        return punishment_scoring_function(target, tapping_iois)
    else:
        raise ValueError(f"Unknown SCORE_MODE: {mode}")



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

# Stilent stimuli for explore tapping task (no onsets required)
music_stimulus_name = ["121", "112"] # TODO: Replace with actual targets (seed interval categories)
music_stimulus_audio = ["music/silence_3sec.wav", "music/silence_3sec.wav"]

def load_audio_only_from_file(fs, audio_filename):
    """
    Load audio file without requiring onsets file.
    """
    stimulus = REPPStimulus("temp", config=sms_tapping)
    return stimulus.load_resample_file(fs, audio_filename)

def filter_and_add_markers_no_onsets(stim, config):
    """
    Apply filtering and add markers without requiring onset information.
    """
    stimulus = REPPStimulus("temp", config=config)
    
    # --- NEW: handle silent input ---------------------------------
    import numpy as np
    # If the signal is (almost) silent, skip filter_stim entirely
    if np.max(np.abs(stim)) < 1e-8:
        # keep it silent; we only want the markers
        filtered_stim = np.zeros_like(stim, dtype=float)
    else:
        # Apply spectral filtering as usual
        filtered_stim = stimulus.filter_stim(
            config.FS, stim, config.STIM_RANGE, config.STIM_AMPLITUDE
        )

        # Guard against NaNs/Infs just in case
        filtered_stim = np.nan_to_num(filtered_stim, nan=0.0, posinf=0.0, neginf=0.0)
    # ----------------------------------------------------------------
    
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
    Create music stimulus without requiring onsets file.
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

reward_nodes_target1a = [
    StaticNode(
        definition={
            "stim_name": TARGET_NODE_1,
            "audio_filename": "music/silence_3sec.wav",
        },
        assets={
            "stimulus_audio": CachedFunctionAsset(generate_music_stimulus_audio),
            "stimulus_info": CachedFunctionAsset(generate_music_stimulus_info),
        },
    )
]

reward_nodes_target1b = [
    StaticNode(
        definition={
            "stim_name": TARGET_NODE_1,
            "audio_filename": "music/silence_3sec.wav",
        },
        assets={
            "stimulus_audio": CachedFunctionAsset(generate_music_stimulus_audio),
            "stimulus_info": CachedFunctionAsset(generate_music_stimulus_info),
        },
    )
]

reward_nodes_target2a = [
    StaticNode(
        definition={
            "stim_name": TARGET_NODE_2,
            "audio_filename": "music/silence_3sec.wav",
        },
        assets={
            "stimulus_audio": CachedFunctionAsset(generate_music_stimulus_audio),
            "stimulus_info": CachedFunctionAsset(generate_music_stimulus_info),
        },
    )
]

reward_nodes_target2b = [
    StaticNode(
        definition={
            "stim_name": TARGET_NODE_2,
            "audio_filename": "music/silence_3sec.wav",
        },
        assets={
            "stimulus_audio": CachedFunctionAsset(generate_music_stimulus_audio),
            "stimulus_info": CachedFunctionAsset(generate_music_stimulus_info),
        },
    )
]

punishment_nodes_target1a = [
    StaticNode(
        definition={
            "stim_name": TARGET_NODE_1,
            "audio_filename": "music/silence_3sec.wav",
        },
        assets={
            "stimulus_audio": CachedFunctionAsset(generate_music_stimulus_audio),
            "stimulus_info": CachedFunctionAsset(generate_music_stimulus_info),
        },
    )
]

punishment_nodes_target1b = [
    StaticNode(
        definition={
            "stim_name": TARGET_NODE_1,
            "audio_filename": "music/silence_3sec.wav",
        },
        assets={
            "stimulus_audio": CachedFunctionAsset(generate_music_stimulus_audio),
            "stimulus_info": CachedFunctionAsset(generate_music_stimulus_info),
        },
    )
]

punishment_nodes_target2a = [
    StaticNode(
        definition={
            "stim_name": TARGET_NODE_2,
            "audio_filename": "music/silence_3sec.wav",
        },
        assets={
            "stimulus_audio": CachedFunctionAsset(generate_music_stimulus_audio),
            "stimulus_info": CachedFunctionAsset(generate_music_stimulus_info),
        },
    )
]

punishment_nodes_target2b = [
    StaticNode(
        definition={
            "stim_name": TARGET_NODE_2,
            "audio_filename": "music/silence_3sec.wav",
        },
        assets={
            "stimulus_audio": CachedFunctionAsset(generate_music_stimulus_audio),
            "stimulus_info": CachedFunctionAsset(generate_music_stimulus_info),
        },
    )
]

nodes_familiarisation = [
    StaticNode(
        definition={
            "stim_name": "121",
            "audio_filename": "music/silence_3sec.wav"
        },
        assets={
            "stimulus_audio": CachedFunctionAsset(generate_music_stimulus_audio),
            "stimulus_info": CachedFunctionAsset(generate_music_stimulus_info),
        },
    )
]

nodes_familiarisation_second = [
    StaticNode(
        definition={
            "stim_name": "121",
            "audio_filename": "music/silence_3sec.wav"
        },
        assets={
            "stimulus_audio": CachedFunctionAsset(generate_music_stimulus_audio),
            "stimulus_info": CachedFunctionAsset(generate_music_stimulus_info),
        },
    )
]

nodes_training = [
    StaticNode(
        definition={
            "stim_name": "121",
            "audio_filename": "music/silence_3sec.wav"
        },
        assets={
            "stimulus_audio": CachedFunctionAsset(generate_music_stimulus_audio),
            "stimulus_info": CachedFunctionAsset(generate_music_stimulus_info),
        },
    )
]

nodes_regular_tapping = [
    StaticNode(
        definition={
            "stim_name": "111",
            "audio_filename": "music/silence_3sec.wav"
        },
        assets={
            "stimulus_audio": CachedFunctionAsset(generate_music_stimulus_audio),
            "stimulus_info": CachedFunctionAsset(generate_music_stimulus_info),
        },
    )
]

########################################################################################################################
# Experiment parts
########################################################################################################################
# class for iso tapping trials
class TapTrialAnalysisISO(AudioRecordTrial, StaticTrial):
    def format_answer(self, raw_answer, **kwargs):
        """Add the Recording asset to trial.assets so async_post_trial can find it."""
        answer = raw_answer
        if isinstance(answer, dict) and answer.get("supports_record_trial") and "asset_id" in answer:
            asset = ExperimentAsset.query.filter_by(id=answer["asset_id"]).one()
            if asset not in self.assets.values():
                self.add_asset("recording", asset)
        return answer

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

        analysis = REPPAnalysis(config=sms_tapping)
        output, analysis, is_failed = analysis.do_analysis(
            info, audio_file, title_in_graph, output_plot
        )
        output = json.dumps(output, cls=NumpySerializer)
        analysis = json.dumps(analysis, cls=NumpySerializer)
        
        return {
            "failed": is_failed["failed"],
            "reason": is_failed["reason"],
            "output": output,
            "analysis": analysis,
            "stim_name": stim_name,
        }

class TapTrialISO(TapTrialAnalysisISO):
    time_estimate = 10
    
    def show_trial(self, experiment, participant):
        info = self.get_info()
        duration_rec = info["stim_duration"]
        trial_number = self.position + 1
        return ModularPage(
            f"trial_main_page_iso_{self.id}",
            AudioPrompt(
                self.assets["stimulus_audio"].url,
                Markup(
                    f"""
                    <br><h3>Tap in time to the beat.</h3>
                    Attempt number {trial_number} out of {NUM_TRIALS_PER_PARTICIPANT_ISO} attempts.
                    """
                ),
            ),
            AudioRecordControl(
                duration=duration_rec,
                show_meter=False,
                controls=False,
                auto_advance=True, # auto advance to next trial after tapping
                bot_response_media=self.get_bot_response_media(),
            ),
            time_estimate=duration_rec + 5,
            show_next_button=False, # hide next button during tapping trial
            progress_display=ProgressDisplay(
                show_bar=True,
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
                        "Uploading audio...",
                        "orange",
                        persistent=True,
                    ),
                ],
            ),
        )

    def get_bot_response_media(self):
        return {
            "iso_800ms": "boot_responses/example_iso_slow_tap.wav",
            "iso_600ms": "boot_responses/example_iso_fast_tap.wav",
        }[self.definition["stim_name"]]

class TapTrialAnalysisExplore(AudioRecordTrial, StaticTrial):
    def format_answer(self, raw_answer, **kwargs):
        """Add the Recording asset to trial.assets so async_post_trial can find it.

        The AudioRecordControl creates the Recording but does not add it to trial.assets.
        RecordTrial.recording looks in trial.assets, so we must add it here before
        async_post_trial runs (which triggers analyse_recording).
        """
        answer = raw_answer
        if isinstance(answer, dict) and answer.get("supports_record_trial") and "asset_id" in answer:
            asset = ExperimentAsset.query.filter_by(id=answer["asset_id"]).one()
            if asset not in self.assets.values():
                self.add_asset("recording", asset)
        return answer

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
        
               # Create a config object with MIN_RAW_TAPS and MAX_RAW_TAPS from experiment settings
        config = ConfigWithThresholds(sms_tapping)
        
        # Use beat detection analysis
        # Pass the stimulus info and config to the analysis function
        output, analysis, is_failed = do_beat_detection_analysis(
            audio_file, title_in_graph, output_plot, stim_info=info, config=config)
                
        # Extract the quality results from is_failed
        is_failed_flag = is_failed.get("failed", True)
        reason = is_failed.get("reason", "Analysis failed")

        extracted_onsets_json = json.dumps(output, cls=NumpySerializer)
        analysis_json = json.dumps(analysis, cls=NumpySerializer)

        # scoring
        TARGET_RHYTHM = [int(char) for char in stim_name]
        tapping_iois = analysis["tapping_iois"]
        num_taps_detected = analysis["num_taps_detected"]

        # print code to debug
        print(f"DEBUG analyze_recording: stim_name={stim_name}, TARGET_RHYTHM={TARGET_RHYTHM}")
        print(f"DEBUG analyze_recording: tapping_iois={tapping_iois}, type={type(tapping_iois)}")
        print(f"DEBUG analyze_recording: num_taps_detected={num_taps_detected}")
        print(f"DEBUG analyze_recording: analysis keys={list(analysis.keys()) if isinstance(analysis, dict) else 'N/A'}")

        score = scoring_function(TARGET_RHYTHM, tapping_iois, mode=self.score_mode)
        
        
        return {
            "failed": is_failed_flag,
            "reason": reason,
            "extracted_onsets": extracted_onsets_json,
            "analysis": analysis_json,
            "tapping_iois": tapping_iois,
            "num_taps_detected": num_taps_detected,
            "target": TARGET_RHYTHM,
            "score": score,
            "stim_name": stim_name,
        }

    def get_bot_response_media(self):
        return {
            "111": "boot_responses/example_silence_10sec.wav",
            "121": "boot_responses/example_silence_10sec.wav",
            "112": "boot_responses/example_silence_10sec.wav",
            "332": "boot_responses/example_silence_10sec.wav",
            "233": "boot_responses/example_silence_10sec.wav",
            "323": "boot_responses/example_silence_10sec.wav",
        }[self.definition["stim_name"]]

class TapTrialExplore(TapTrialAnalysisExplore):
    time_estimate = DURATION_ESTIMATED_TRIAL
    
    def show_trial(self, experiment, participant):
        info = self.get_info()
        # stim_name = info["stim_name"]
        # TARGET_RHYTHM = [int(char) for char in stim_name]

        duration_rec = info["stim_duration"]
        total_main_trials = NUM_TRIALS_PER_TARGET * 4

        explore_trial_numbers = participant.var.get("explore_trial_numbers", {})
        trial_key = str(self.id)

        if trial_key not in explore_trial_numbers:
            next_trial_number = participant.var.get("explore_trial_counter", 0) + 1
            explore_trial_numbers[trial_key] = next_trial_number
            participant.var.set("explore_trial_numbers", explore_trial_numbers)
            participant.var.set("explore_trial_counter", next_trial_number)

        overall_trial_number = explore_trial_numbers[trial_key]

        return ModularPage(
            f"trial_main_page_explore_{self.id}",
            AudioPrompt(
                self.assets["stimulus_audio"].url,
                Markup(
                    f"""
                    <br><h3>Find the hidden rhythm</h3>
                    <b>Produce 4 taps. Explore different rhythms.</b>
                    <br><br>
                    <i>Attempt number {overall_trial_number} out of {total_main_trials} attempts.</i>
                    """
                ),
            ),
            AudioRecordControl(
                duration=duration_rec,
                show_meter=False,
                controls=False,
                auto_advance=True, # auto advance to next trial after tapping
                bot_response_media=self.get_bot_response_media(),
            ),
            time_estimate=duration_rec + 5,
            show_next_button=False, # hide next button during tapping trial
            progress_display=ProgressDisplay(
                show_bar=True,
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
                        "Uploading audio...",
                        "orange",
                        persistent=True,
                    ),
                ],
            ),
        )

    def gives_feedback(self, experiment, participant):
        return True

    def show_feedback(self, experiment, participant):
        output_analysis = self.analysis
        num_taps_detected = output_analysis["num_taps_detected"]
        score = output_analysis["score"]

        self.var.set("score", score)
        self.var.set("num_taps_detected", num_taps_detected)
        self.var.set("target_rhythm", "".join(str(x) for x in output_analysis["target"]))
        self.var.set("stim_name", output_analysis["stim_name"])

        if num_taps_detected == TARGET_NUM_TAPS:
            if self.score_mode == "punishment":
                return InfoPage(
                    Markup(
                        f"""
                        <h3>Feedback</h3>
                        <hr>
                        <div style="font-size: 1.4em; text-align: center;">Your penalty score is</div>
                        <div style="font-size: 3em; font-weight: bold; text-align: center; margin: 0.35em 0;">{score}</div>
                        <hr>
                        """
                    ),
                    time_estimate=2,
                )
            else:
                return InfoPage(
                    Markup(
                       f"""
                        <h3>Feedback</h3>
                        <hr>
                        <div style="font-size: 1.4em; text-align: center;">Your score is</div>
                        <div style="font-size: 3em; font-weight: bold; text-align: center; margin: 0.35em 0;">{score}</div>
                        <hr>
                        """
                    ),
                    time_estimate=2,
                )
        else:
            return InfoPage(
                Markup(
                    f"""
                    <h3>We did not detect the correct number of taps</h3>
                    <hr>
                    We detected <b>{num_taps_detected}</b> taps, but you should produce <b>exactly 4 taps</b> during the green phase.
                    <br><br>
                    Please try again and remember:
                    <ol>
                        <li>Tap only during the green phase.</li>
                        <li>Do not tap during the red phases.</li>
                        <li>You may use either your left or right hand.</li>
                        <li>Tap on the surface of your laptop using your index finger.</li>
                    </ol>
                    <hr>
                    """
                ),
                time_estimate=2,
            )

class RewardTapTrialExplore(TapTrialExplore):
    score_mode = "reward"


class PunishmentTapTrialExplore(TapTrialExplore):
    score_mode = "punishment"


class FamiliarisationTapTrial1(TapTrialAnalysisExplore):
    score_mode = "reward"
    time_estimate = DURATION_ESTIMATED_TRIAL
    def show_trial(self, experiment, participant):
        info = self.get_info()
        duration_rec = info["stim_duration"]

        return ModularPage(
            f"trial_main_page_familiarisation_{self.id}",
            AudioPrompt(
                self.assets["stimulus_audio"].url,
                Markup(
                    """
                    <br><h3>Familiarisation</h3>
                    Your goal is simple: produce <b>exactly 4 taps</b> during the <b>green</b> phase.
                    <br><br>
                    Do not tap during the red phases.
                    <br><br>
                    You may use either your left or right hand.
                    <br><br>
                    <i>Practice attempt</i>
                    """
                ),
            ),
            AudioRecordControl(
                duration=duration_rec,
                show_meter=False,
                controls=False,
                auto_advance=True, # auto advance to next trial after tapping
                bot_response_media=self.get_bot_response_media(),
            ),
            time_estimate=duration_rec + 5,
            show_next_button=False, # hide next button during tapping trial
            progress_display=ProgressDisplay(
                show_bar=True,
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
                        "Uploading audio...",
                        "orange",
                        persistent=True,
                    ),
                ],
            ),
        )

    def gives_feedback(self, experiment, participant):
        return True
    
    def show_feedback(self, experiment, participant):
        output_analysis = self.analysis
        num_taps_detected = output_analysis["num_taps_detected"]
        self.var.set("num_taps_detected", num_taps_detected)

        participant.var.set(
            "familiarisation_first_attempts",
            participant.var.get("familiarisation_first_attempts", 0) + 1
        )

        if num_taps_detected == TARGET_NUM_TAPS:
            participant.var.set("familiarisation_first_success", True)
            return InfoPage(
                Markup(
                    f"""
                    <h3>Attempt</h3>
                    <hr>
                    We detected <b>{num_taps_detected}</b> taps.
                    <br><br>
                    Great, have another go to make sure you are comfortable with tapping.
                    <br><br>
                    Press <b>Next</b> when you are ready.
                    <hr>
                    """
                ),
                time_estimate=2,
            )
        else:
            return InfoPage(
                Markup(
                    f"""
                    <h3>We did not detect the correct number of taps</h3>
                    <hr>
                    We detected <b>{num_taps_detected}</b> taps in the rhythm, but we asked you to produce a rhythm with <b>{TARGET_NUM_TAPS}</b> taps.
                    <br><br>
                    Please try to do one or more of the following:
                    <ol>
                        <li>Make sure you are in a quiet environment.</li>
                        <li>Tap on the surface of your laptop using your index finger.</li>
                        <li>Do not tap during the beeps at the start and end of the recording.</li>
                    </ol>
                    <b>If you don't improve your performance, the experiment will terminate.</b>
                    <hr>
                    """
                ),
                time_estimate=2,
            )

class FamiliarisationTapTrial2(FamiliarisationTapTrial1):
    time_estimate = DURATION_ESTIMATED_TRIAL
    def show_feedback(self, experiment, participant):
        output_analysis = self.analysis
        num_taps_detected = output_analysis["num_taps_detected"]
        self.var.set("num_taps_detected", num_taps_detected)

        participant.var.set(
            "familiarisation_second_attempts",
            participant.var.get("familiarisation_second_attempts", 0) + 1
        )

        if num_taps_detected == TARGET_NUM_TAPS:
            participant.var.set("familiarisation_second_success", True)
            return InfoPage(
                Markup(
                    f"""
                    <h3>Familiarisation complete</h3>
                    <hr>
                    We detected <b>{num_taps_detected}</b> taps.
                    <br><br>
                    Great. You have now completed the familiarisation phase successfully twice.
                    <br><br>
                    Press Next when you are ready to begin training.
                    <hr>
                    """
                ),
                time_estimate=2,
            )
        else:
            return InfoPage(
                Markup(
                    f"""
                    <h3>Oops, something went wrong.</h3>
                    <hr>
                    Please re-read the instructions and make sure you produce <b>exactly 4 taps</b> during the “green” phase.
                    <br><br>
                    We detected <b>{num_taps_detected}</b> taps.
                    <br><br>
                    The task consists of three phases:
                    <ol>
                        <li>In the first phase, a cue will be played back. You don't need to do anything while listening to it. Please remain silent.</li>
                        <li>In the second (main) phase, <b>you tap</b>. Your goal is to produce <b>exactly 4 taps, neither more nor less</b>.</li>
                        <li>In the last phase, you should stop tapping and remain silent.</li>
                    </ol>
                    <br><br>
                    The three phases will be represented by a progress bar that will be red during phases 1 and 3 and green during phase 2.
                    <br><br>
                    Try to do one or more of the following:
                    <ol>
                        <li>Make sure you are in a quiet environment.</li>
                        <li>Tap on the surface of your laptop using your index finger.</li>
                        <li>Do not tap during the beeps at the start and end of the recording.</li>
                    </ol>
                    <br><br>
                    <b>Unfortunately, if you do not improve your performance, the experiment will soon terminate.</b>
                    <hr>
                    """
                ),
                time_estimate=2,
            )

# TrainingTapTrial will inherit score persistence from TapTrialExplore.show_feedback

class TrainingTapTrial(TapTrialExplore):
    score_mode = "reward"
    def show_trial(self, experiment, participant):
        info = self.get_info()
        duration_rec = info["stim_duration"]

        training_trial_number = self.position + 1

        return ModularPage(
            f"trial_main_page_training_{self.id}",
            AudioPrompt(
                self.assets["stimulus_audio"].url,
                Markup(
                    f"""
                    <br><h3>Training</h3>
                    <b>Produce 4 taps. Explore different rhythms!</b>
                    <br><br>
                    <i>Training trial {training_trial_number} out of 5.</i>
                    """
                ),
            ),
            AudioRecordControl(
                duration=duration_rec,
                show_meter=False,
                controls=False,
                auto_advance=True, # auto advance to next trial after tapping
                bot_response_media=self.get_bot_response_media(),
            ),
            time_estimate=duration_rec + 5,
            show_next_button=False, # hide next button during tapping trial
            progress_display=ProgressDisplay(
                show_bar=True,
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
                        "Uploading audio...",
                        "orange",
                        persistent=True,
                    ),
                ],
            ),
        )


def welcome():
    return InfoPage(
        Markup(
            """
            <h3>Welcome</h3>
            <hr>
            In this experiment, you will have to tap on the surface of your laptop with your index finger.
            <br><br>
            We will monitor your responses throughout the experiment.
            <br><br>
            Press <b>Next</b> when you are ready to start.
            <hr>
            """
        ),
        time_estimate=3,
    )

class RegularTappingTrial(TapTrialAnalysisExplore):
    score_mode = "reward"
    time_estimate = DURATION_ESTIMATED_TRIAL

    def show_trial(self, experiment, participant):
        info = self.get_info()
        duration_rec = info["stim_duration"]
        regular_trial_number = self.position + 1

        return ModularPage(
            f"trial_main_page_regular_tapping_{self.id}",
            AudioPrompt(
                self.assets["stimulus_audio"].url,
                Markup(
                    f"""
                    <br><h3>Keeping Regular Tapping</h3>
                    <b>Produce exactly 4 taps during the green phase.</b>
                    <br><br>
                    Aim to produce <b>4 regular intervals</b>, with the same interval between taps.
                    <br><br>
                    Try to reproduce exactly the same regular tapping pattern on every attempt.
                    <br><br>
                    <i>Attempt {regular_trial_number} out of 10.</i>
                    """
                ),
            ),
            AudioRecordControl(
                duration=duration_rec,
                show_meter=False,
                controls=False,
                auto_advance=True, # auto advance to next trial after tapping
                bot_response_media=self.get_bot_response_media(),
            ),
            time_estimate=duration_rec + 5,
            show_next_button=False, # hide next button during tapping trial
            progress_display=ProgressDisplay(
                show_bar=True,
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
                        "Uploading audio...",
                        "orange",
                        persistent=True,
                    ),
                ],
            ),
        )

    def gives_feedback(self, experiment, participant):
        return False

class FamiliarisationIntroPage(InfoPage):
    def on_arrival(self, experiment, participant):
        participant.var.set("familiarisation_first_success", False)
        participant.var.set("familiarisation_second_success", False)
        participant.var.set("familiarisation_first_attempts", 0)
        participant.var.set("familiarisation_second_attempts", 0)
        participant.var.set("explore_trial_counter", 0)
        participant.var.set("explore_trial_numbers", {})

# Familiarisation task
familiarisation_explore_tapping = FamiliarisationIntroPage(
    Markup(
        """
        <h3>Practice how to tap on your laptop</h3>
        <hr>
        Please always tap on the surface of your laptop using your index finger. You can use either your right or left hand, whichever is more comfortable for you.
        <br><br>
        Do not tap on the keyboard or trackpad, and do not tap using your nails or any other object.
        <br><br>
        Your goal is to produce <b>exactly 4 taps</b> during the <b>green</b> phase.
        <br><br>
        Remember:
        <ol>
            <li>Tap on the surface of your laptop using your index finger.</li>
            <li>Do not tap on any key or trackpad.</li>
            <li>Do not tap with your fingernail or any object.</li>
        </ol>
        <br>
        Press <b>Next</b> when you are ready to begin.
        """
    ),
    time_estimate=5,
)
### Attempt until they produce 4 taps (if not correct, then reread the instructions to make sure that they produce exactly 4 taps during the green phase)

training_explore_tapping = InfoPage(
    Markup(
        """
        <h3>Training</h3>
        <hr>
        Congratulations, you are now ready to try how the task works.
        <br><br>
        Your goal is to guess a hidden rhythmic pattern. You will have <b>5 attempts</b>. In each attempt, you must produce 4 taps, varying the time between taps (shorter or longer pauses).
        <br><br>
        After each attempt, you will see a score from 0 to 100. The score tells you how close your tapping was to the hidden pattern. Use the score to guide your learning.
        <br><br>
        This training uses a different hidden rhythm than the main experiment, so you will need to learn again from the beginning later.
        <br><br>
        Don’t worry if you cannot find the rhythm in 5 attempts. The goal here is just to understand how the task works.
        <br><br>
        Press <b>Next</b> when you are ready to start.
        <hr>
        """
    ),
    time_estimate=5,
)

familiarisation_trial_1 = StaticTrialMaker(
    id_="familiarisation_trial_1",
    trial_class=FamiliarisationTapTrial1,
    nodes=nodes_familiarisation,
    expected_trials_per_participant=5,
    max_trials_per_participant=5,
    allow_repeated_nodes=True,
    n_repeat_trials=0,
    target_n_participants=NUM_PARTICIPANTS,
    recruit_mode="n_participants",
    check_performance_at_end=False,
)
familiarisation_trial_1.time_estimate = DURATION_ESTIMATED_TRIAL + 2

familiarisation_trial_2 = StaticTrialMaker(
    id_="familiarisation_trial_2",
    trial_class=FamiliarisationTapTrial2,
    nodes=nodes_familiarisation_second,
    expected_trials_per_participant=5,
    max_trials_per_participant=5,
    allow_repeated_nodes=True,
    n_repeat_trials=0,
    target_n_participants=NUM_PARTICIPANTS,
    recruit_mode="n_participants",
    check_performance_at_end=False,
)
familiarisation_trial_2.time_estimate = DURATION_ESTIMATED_TRIAL + 2

familiarisation_loop_module_1 = Module(
    "familiarisation_loop_module_1",
    familiarisation_trial_1,
)

familiarisation_loop_module_2 = Module(
    "familiarisation_loop_module_2",
    familiarisation_trial_2,
)

training_trials = StaticTrialMaker(
    id_="training_trials",
    trial_class=TrainingTapTrial,
    nodes=nodes_training,
    expected_trials_per_participant=5,
    max_trials_per_participant=5,
    allow_repeated_nodes=True,
    n_repeat_trials=0,
    target_n_participants=NUM_PARTICIPANTS,
    recruit_mode="n_participants",
    check_performance_at_end=False,
)
training_trials.time_estimate = 5 * (DURATION_ESTIMATED_TRIAL + 2)

regular_tapping_trials = StaticTrialMaker(
    id_="regular_tapping_trials",
    trial_class=RegularTappingTrial,
    nodes=nodes_regular_tapping,
    expected_trials_per_participant=10,
    max_trials_per_participant=10,
    allow_repeated_nodes=True,
    n_repeat_trials=0,
    target_n_participants=NUM_PARTICIPANTS,
    recruit_mode="n_participants",
    check_performance_at_end=False,
)
regular_tapping_trials.time_estimate = 10 * (DURATION_ESTIMATED_TRIAL + 2)

instructions_explore_tapping = InfoPage(
    Markup(
        f"""
        <h3>Main Task</h3>
        <hr>
        Congratulations, you are now ready for the main task.
        <br><br>
        Your goal is still to discover a hidden rhythmic pattern. In each attempt, you must produce 4 taps while varying their timing.
        <br><br>
        The main task has <b>4 sets of {NUM_TRIALS_PER_TARGET} attempts</b>, for a total of <b>{NUM_TRIALS_PER_TARGET * 4}</b> attempts.
        <br><br>
        {(
            'The first two sets use the <b>reward</b> version of the task, where scores range from <b>0 to 100</b> and indicate how close you are to <b>rhythm solution 1</b>.<br>'
            'The last two sets use the <b>punishment</b> version of the task, where scores range from <b>-100 to 0</b> and indicate how close you are to a new <b>rhythm solution 2</b>.'
            if MAIN_TASK_ORDER == "reward_first"
            else 'The first two sets use the <b>punishment</b> version of the task, where scores range from <b>-100 to 0</b> and indicate how close you are to <b>rhythm solution 1</b>.<br>'
                'The last two sets use the <b>reward</b> version of the task, where scores range from <b>0 to 100</b> and indicate how close you are to a new <b>rhythm solution 2</b>.'
        )}
        <br><br>
        There will be a short break after each set of <b>{NUM_TRIALS_PER_TARGET}</b> attempts.
        <br><br>
        Press <b>Next</b> when you are ready to start.
        <hr>
        """
    ),
    time_estimate=10,
)


### Momentary subjective states page questionnaire
def make_momentary_subjective_states_page(label_suffix):
    return ModularPage(
        f"momentary_subjective_states_{label_suffix}",
        Markup(
            """
            <h3>How do you feel right now?</h3>
            <hr>
            Read each statement and then enter the appropriate number to the right of the statement to indicate how you feel right now, at this moment.
            <br><br>
            Numbers go from <b>1 (not at all)</b> to <b>5 (very much so)</b>.
            <hr>
            """
        ),
        SurveyJSControl(
            {
                "pages": [
                    {
                        "name": "momentary_subjective_states",
                        "elements": [
                            {
                                "type": "text",
                                "name": "energised",
                                "title": "I feel energised right now:",
                                "inputType": "number",
                                "isRequired": True,
                                "min": 1,
                                "max": 5,
                            },
                            {
                                "type": "text",
                                "name": "heart_racing",
                                "title": "My heart is racing:",
                                "inputType": "number",
                                "isRequired": True,
                                "min": 1,
                                "max": 5,
                            },
                            {
                                "type": "text",
                                "name": "concerned_about_performing_poorly",
                                "title": "I'm concerned about performing poorly:",
                                "inputType": "number",
                                "isRequired": True,
                                "min": 1,
                                "max": 5,
                            },
                            {
                                "type": "text",
                                "name": "irritated",
                                "title": "I feel irritated right now:",
                                "inputType": "number",
                                "isRequired": True,
                                "min": 1,
                                "max": 5,
                            },
                        ],
                    }
                ],
                "completeText": "Next",
                "showQuestionNumbers": "off",
                "questionErrorLocation": "bottom",
            },
            bot_response=lambda: {
                "energised": 3,
                "heart_racing": 3,
                "concerned_about_performing_poorly": 3,
                "irritated": 3,
            },
        ),
        time_estimate=12,
    )


momentary_subjective_states_1 = make_momentary_subjective_states_page("1")
momentary_subjective_states_2 = make_momentary_subjective_states_page("2")
momentary_subjective_states_3 = make_momentary_subjective_states_page("3")
momentary_subjective_states_4 = make_momentary_subjective_states_page("4")
momentary_subjective_states_5 = make_momentary_subjective_states_page("5")
###

regular_tapping_intro = InfoPage(
    Markup(
        """
        <br><h3>Keeping Regular Tapping</h3>
        <hr>
        Now your final goal is different: produce <b>exactly 4 taps</b> during the <b>green</b> phase.
        <br><br>
        But aim to produce <b>4 regular intervals</b>, with the same interval between taps.
        <br><br>
        Try to reproduce exactly the same regular tapping pattern every attempt.
        <br><br>
        You will not receive a score for this, but try to be consistent in your tapping.
        <br><br>
        Press <b>Next</b> when you are ready to start.
        <hr>
        """
    ),
    time_estimate=5,
)

break_after_reward_1 = InfoPage(
    Markup(
        f"""
        <h3>Break</h3>
        <hr>
        You have completed the first set of {NUM_TRIALS_PER_TARGET} attempts in the reward task.
        <br><br>
        Please take a short break to rest if needed.
        <br><br>
        Press <b>Next</b> when you are ready to continue.
        <hr>
        """
    ),
    time_estimate=10,
)

break_after_reward_2 = InfoPage(
    Markup(
        f"""
        <h3>Break</h3>
        <hr>
        You have completed the second set of {NUM_TRIALS_PER_TARGET} attempts in the reward task.
        <br><br>
        {('The next part will use the punishment version of the task.' if MAIN_TASK_ORDER == "reward_first" else 'You have now finished the reward part of the task.')}
        <br><br>
        Please take a short break to rest if needed.
        <br><br>
        Press <b>Next</b> when you are ready to continue.
        <hr>
        """
    ),
    time_estimate=10,
)

break_after_punishment_1 = InfoPage(
    Markup(
       f"""
        <h3>Break</h3>
        <hr>
        You have completed the first set of {NUM_TRIALS_PER_TARGET} attempts in the punishment task.
        <br><br>
        Please take a short break to rest if needed.
        <br><br>
        Press <b>Next</b> when you are ready to continue.
        <hr>
        """
    ),
    time_estimate=10,
)

break_after_punishment_2 = InfoPage(
    Markup(
       f"""
        <h3>Break</h3>
        <hr>
        You have completed the second set of {NUM_TRIALS_PER_TARGET} attempts in the punishment task.
        <br><br>
        {('The next part will use the reward version of the task.' if MAIN_TASK_ORDER == "punishment_first" else 'You have now finished the punishment part of the task.')}
        <br><br>
        Please take a short break to rest if needed.
        <br><br>
        Press <b>Next</b> when you are ready to continue.
        <hr>
        """
    ),
    time_estimate=10,
)

custom_end_page = InfoPage(
    Markup(
        """
        <h3>That's the end of the experiment</h3>
        <hr>
        Thank you for taking part.
        <br><br>
        Please press <b>Next</b> to finish the study.
        <hr>
        """
    ),
    time_estimate=3,
)

reward_explore_tapping_target1a = StaticTrialMaker(
    id_="reward_explore_tapping_target1a",
    trial_class=RewardTapTrialExplore,
    nodes=reward_nodes_target1a,
    expected_trials_per_participant=NUM_TRIALS_PER_TARGET,
    max_trials_per_participant=NUM_TRIALS_PER_TARGET,
    allow_repeated_nodes=True,
    n_repeat_trials=0,
    target_n_participants=NUM_PARTICIPANTS,
    recruit_mode="n_participants",
    check_performance_at_end=False,
)
reward_explore_tapping_target1a.time_estimate = NUM_TRIALS_PER_TARGET * (DURATION_ESTIMATED_TRIAL + 2)

reward_explore_tapping_target1b = StaticTrialMaker(
    id_="reward_explore_tapping_target1b",
    trial_class=RewardTapTrialExplore,
    nodes=reward_nodes_target1b,
    expected_trials_per_participant=NUM_TRIALS_PER_TARGET,
    max_trials_per_participant=NUM_TRIALS_PER_TARGET,
    allow_repeated_nodes=True,
    n_repeat_trials=0,
    target_n_participants=NUM_PARTICIPANTS,
    recruit_mode="n_participants",
    check_performance_at_end=False,
)
reward_explore_tapping_target1b.time_estimate = NUM_TRIALS_PER_TARGET * (DURATION_ESTIMATED_TRIAL + 2)

reward_explore_tapping_target2a = StaticTrialMaker(
    id_="reward_explore_tapping_target2a",
    trial_class=RewardTapTrialExplore,
    nodes=reward_nodes_target2a,
    expected_trials_per_participant=NUM_TRIALS_PER_TARGET,
    max_trials_per_participant=NUM_TRIALS_PER_TARGET,
    allow_repeated_nodes=True,
    n_repeat_trials=0,
    target_n_participants=NUM_PARTICIPANTS,
    recruit_mode="n_participants",
    check_performance_at_end=False,
)
reward_explore_tapping_target2a.time_estimate = NUM_TRIALS_PER_TARGET * (DURATION_ESTIMATED_TRIAL + 2)

reward_explore_tapping_target2b = StaticTrialMaker(
    id_="reward_explore_tapping_target2b",
    trial_class=RewardTapTrialExplore,
    nodes=reward_nodes_target2b,
    expected_trials_per_participant=NUM_TRIALS_PER_TARGET,
    max_trials_per_participant=NUM_TRIALS_PER_TARGET,
    allow_repeated_nodes=True,
    n_repeat_trials=0,
    target_n_participants=NUM_PARTICIPANTS,
    recruit_mode="n_participants",
    check_performance_at_end=False,
)
reward_explore_tapping_target2b.time_estimate = NUM_TRIALS_PER_TARGET * (DURATION_ESTIMATED_TRIAL + 2)

punishment_explore_tapping_target1a = StaticTrialMaker(
    id_="punishment_explore_tapping_target1a",
    trial_class=PunishmentTapTrialExplore,
    nodes=punishment_nodes_target1a,
    expected_trials_per_participant=NUM_TRIALS_PER_TARGET,
    max_trials_per_participant=NUM_TRIALS_PER_TARGET,
    allow_repeated_nodes=True,
    n_repeat_trials=0,
    target_n_participants=NUM_PARTICIPANTS,
    recruit_mode="n_participants",
    check_performance_at_end=False,
)
punishment_explore_tapping_target1a.time_estimate = NUM_TRIALS_PER_TARGET * (DURATION_ESTIMATED_TRIAL + 2)

punishment_explore_tapping_target1b = StaticTrialMaker(
    id_="punishment_explore_tapping_target1b",
    trial_class=PunishmentTapTrialExplore,
    nodes=punishment_nodes_target1b,
    expected_trials_per_participant=NUM_TRIALS_PER_TARGET,
    max_trials_per_participant=NUM_TRIALS_PER_TARGET,
    allow_repeated_nodes=True,
    n_repeat_trials=0,
    target_n_participants=NUM_PARTICIPANTS,
    recruit_mode="n_participants",
    check_performance_at_end=False,
)
punishment_explore_tapping_target1b.time_estimate = NUM_TRIALS_PER_TARGET * (DURATION_ESTIMATED_TRIAL + 2)

punishment_explore_tapping_target2a = StaticTrialMaker(
    id_="punishment_explore_tapping_target2a",
    trial_class=PunishmentTapTrialExplore,
    nodes=punishment_nodes_target2a,
    expected_trials_per_participant=NUM_TRIALS_PER_TARGET,
    max_trials_per_participant=NUM_TRIALS_PER_TARGET,
    allow_repeated_nodes=True,
    n_repeat_trials=0,
    target_n_participants=NUM_PARTICIPANTS,
    recruit_mode="n_participants",
    check_performance_at_end=False,
)
punishment_explore_tapping_target2a.time_estimate = NUM_TRIALS_PER_TARGET * (DURATION_ESTIMATED_TRIAL + 2)

punishment_explore_tapping_target2b = StaticTrialMaker(
    id_="punishment_explore_tapping_target2b",
    trial_class=PunishmentTapTrialExplore,
    nodes=punishment_nodes_target2b,
    expected_trials_per_participant=NUM_TRIALS_PER_TARGET,
    max_trials_per_participant=NUM_TRIALS_PER_TARGET,
    allow_repeated_nodes=True,
    n_repeat_trials=0,
    target_n_participants=NUM_PARTICIPANTS,
    recruit_mode="n_participants",
    check_performance_at_end=False,
)
punishment_explore_tapping_target2b.time_estimate = NUM_TRIALS_PER_TARGET * (DURATION_ESTIMATED_TRIAL + 2)

########################################################################################################################
########################################################################################################################
class Exp(psynet.experiment.Experiment):
    label = "Tapping Experiment"
    asset_storage = LocalStorage()

    config = {
        "initial_recruitment_size": INITIAL_RECRUITMENT_SIZE,
        "auto_recruit": AUTO_RECRUIT,
        "show_reward": False, # disable reward display with bonus
        "title": "Tapping experiment (Chrome browser, ~14 mins)",
        "description": "This is a tapping experiment. You will be asked to listen to rhythms and synchronize to the beat by tapping with your finger.",
        "contact_email_on_error": "m.angladatort@gold.ac.uk",
        "organization_name": "Max Planck Institute for Empirical Aesthetics",
    }
    if DEBUG:
        timeline = Timeline(
            NoConsent(),
            welcome(),
            REPPVolumeCalibrationMusic(),
            REPPMarkersTest(),
            familiarisation_explore_tapping,
            while_loop(
               "repeat_familiarisation_until_first_success",
               lambda participant: (
                   not participant.var.get("familiarisation_first_success", False)
                   and participant.var.get("familiarisation_first_attempts", 0) < 5
                                    ),
               familiarisation_loop_module_1,
               expected_repetitions=5,
               fix_time_credit=False,
            ),
            while_loop(
               "repeat_familiarisation_until_second_success",
               lambda participant: (
                   not participant.var.get("familiarisation_second_success", False)
                   and participant.var.get("familiarisation_second_attempts", 0) < 5
                                 ),
                familiarisation_loop_module_2,
                expected_repetitions=5,
                fix_time_credit=False,
            ),
            training_explore_tapping,
            training_trials,
            instructions_explore_tapping,
            *(
                [
                    momentary_subjective_states_1,
                    reward_explore_tapping_target1a,
                    momentary_subjective_states_2,
                    break_after_reward_1,
                    reward_explore_tapping_target1b,
                    momentary_subjective_states_3,
                    break_after_reward_2,
                    punishment_explore_tapping_target2a,
                    momentary_subjective_states_4,
                    break_after_punishment_1,
                    punishment_explore_tapping_target2b,
                    momentary_subjective_states_5,
                ]
                if MAIN_TASK_ORDER == "reward_first"
                else [
                    momentary_subjective_states_1,
                    punishment_explore_tapping_target1a,
                    momentary_subjective_states_2,
                    break_after_punishment_1,
                    punishment_explore_tapping_target1b,
                    momentary_subjective_states_3,
                    break_after_punishment_2,
                    reward_explore_tapping_target2a,
                    momentary_subjective_states_4,
                    break_after_reward_1,
                    reward_explore_tapping_target2b,
                    momentary_subjective_states_5,
                ]
            ),
            regular_tapping_intro,
            regular_tapping_trials,
            custom_end_page,
            SuccessfulEndPage(),
        )
    else:
        timeline = Timeline(
            NoConsent(),
            # welcome(),
            # REPPVolumeCalibrationMusic(),
            # REPPMarkersTest(),
            # familiarisation_explore_tapping,
            while_loop(
               "repeat_familiarisation_until_first_success",
               lambda participant: (
                   not participant.var.get("familiarisation_first_success", False)
                   and participant.var.get("familiarisation_first_attempts", 0) < 5
                                    ),
               familiarisation_loop_module_1,
               expected_repetitions=5,
               fix_time_credit=False,
            ),
            while_loop(
               "repeat_familiarisation_until_second_success",
               lambda participant: (
                   not participant.var.get("familiarisation_second_success", False)
                   and participant.var.get("familiarisation_second_attempts", 0) < 5
                                 ),
                familiarisation_loop_module_2,
                expected_repetitions=5,
                fix_time_credit=False,
            ),
            training_explore_tapping,
            training_trials,
            instructions_explore_tapping,
            *(
                [
                    momentary_subjective_states_1,
                    reward_explore_tapping_target1a,
                    momentary_subjective_states_2,
                    break_after_reward_1,
                    reward_explore_tapping_target1b,
                    momentary_subjective_states_3,
                    break_after_reward_2,
                    punishment_explore_tapping_target2a,
                    momentary_subjective_states_4,
                    break_after_punishment_1,
                    punishment_explore_tapping_target2b,
                    momentary_subjective_states_5,
                ]
                if MAIN_TASK_ORDER == "reward_first"
                else [
                    momentary_subjective_states_1,
                    punishment_explore_tapping_target1a,
                    momentary_subjective_states_2,
                    break_after_punishment_1,
                    punishment_explore_tapping_target1b,
                    momentary_subjective_states_3,
                    break_after_punishment_2,
                    reward_explore_tapping_target2a,
                    momentary_subjective_states_4,
                    break_after_reward_1,
                    reward_explore_tapping_target2b,
                    momentary_subjective_states_5,
                ]
            ),
            regular_tapping_intro,
            regular_tapping_trials,
            custom_end_page,
            SuccessfulEndPage()
        )
