import time
from pathlib import Path
from copy import deepcopy
import numpy as np
import slab
import pickle
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import datetime
from freefield import DIR, Processors, camera
import logging
logging.basicConfig(level=logging.INFO)
slab.Signal.set_default_samplerate(48828)  # default samplerate for generating sounds, filters etc.
# Initialize global variables:
CAMERAS = None
PROCESSORS = Processors()
CALIBRATIONFILE = Path()
CALIBRATIONDICT = {}  # calibration to equalize levels
TABLE = pd.DataFrame()  # numbers and coordinates of all loudspeakers


def initialize_setup(setup, default_mode=None, proc_list=None, zbus=True, connection="GB", camera_type="flir"):
    """
    Initialize the processors and load table and calibration for setup.

    We are using two different 48-channel setups. A 3-dimensional 'dome' and
    a horizontal 'arc'. For each setup there is a table describing the position
    and channel of each loudspeaker as well as calibration files. This function
    loads those files and stores them in global variables. This is necessary
    for most of the other functions to work.

    Args:
        setup: determines which files to load, can be 'dome' or 'arc'
        default_mode: initialize the setup using one of the defaults, see Processors.initialize_default
        proc_list: if not using a default, specify the processors in a list, see processors.initialize_processors
        zbus: whether or not to initialize the zbus interface
        connection: type of connection to processors, can be "GB" (optical) or "USB"
        camera_type: kind of camera that is initialized. Can be "webcam", "flir" or None
    """

    # TODO: put level and frequency equalization in one common file
    global CALIBRATIONDICT, CALIBRATIONFILE, TABLE, PROCESSORS, CAMERAS
    # initialize processors
    if bool(proc_list) == bool(default_mode):
        raise ValueError("You have to specify a proc_list OR a default_mode")
    if proc_list is not None:
        PROCESSORS.initialize(proc_list, zbus, connection)
    elif default_mode is not None:
        PROCESSORS.initialize_default(default_mode)
    if camera_type is not None:
        CAMERAS = camera.initialize_cameras(camera_type)
    # get the correct speaker table and calibration files for the setup
    if setup == 'arc':
        CALIBRATIONFILE = DIR / 'data' / Path('calibration_arc.pkl')
        table_file = DIR / 'data' / 'tables' / Path('speakertable_arc.txt')
    elif setup == 'dome':
        CALIBRATIONFILE = DIR / 'data' / Path('calibration_dome.pkl')
        table_file = DIR / 'data' / 'tables' / Path('speakertable_dome.txt')
    else:
        raise ValueError("Unknown setup! Use 'arc' or 'dome'.")
    logging.info(f'Speaker configuration set to {setup}.')
    # lambdas provide default values of 0 if azi or ele are not in the file
    TABLE = pd.read_csv(table_file, dtype={"index_number": "Int64", "channel": "Int64", "analog_proc": "category",
                         "azi": float, "ele": float, "bit": "Int64", "digital_proc": "category"})
    logging.info('Speaker table loaded.')
    if CALIBRATIONFILE.exists():
        with open(CALIBRATIONFILE, 'rb') as f:
            CALIBRATIONDICT = pickle.load(f)
        logging.info('Frequency-calibration filters loaded.')
    else:
        logging.warning('Setup not calibrated...')


# Wrappers for Processor operations read, write, trigger and halt:
def write(tag, value, procs):
    PROCESSORS.write(tag=tag, value=value, procs=procs)


def read(tag, proc, n_samples=1):
    value = PROCESSORS.read(tag=tag, proc=proc, n_samples=n_samples)
    return value

def play(kind='zBusA', proc=None):
    PROCESSORS.trigger(kind=kind, proc=proc)


def halt():
    PROCESSORS.halt()


def wait_to_finish_playing(proc="all", tag="playback"):
    """
    Busy wait until the processors finished playing.

    For this function to work, the rcx-circuit must have a tag that is 1
    while output is generated and 0 otherwise. The default name for this
    kind of tag is "playback". "playback" is read repeatedly for each processors
    followed by a short sleep if the value is 1.

    Args:
        proc (str, list of str): name(s) of the processor(s) to wait for.
        tag (str): name of the tag that signals if something is played
    """
    if proc == "all":
        proc = list(PROCESSORS.procs.keys())
    elif isinstance(proc, str):
        proc = [proc]
    logging.info(f'Waiting for {tag} on {proc}.')
    while any(PROCESSORS.read(tag, n_samples=1, proc=p) for p in proc):
        time.sleep(0.01)
    logging.info('Done waiting.')


def wait_for_button() -> None:
    while not PROCESSORS.read(tag="response", proc="RP2"):
        time.sleep(0.1)  # wait until button is pressed


def play_and_wait() -> None:
    PROCESSORS.play()
    wait_to_finish_playing()


def play_and_wait_for_button() -> None:
    play_and_wait()
    wait_for_button()


def get_speaker(index_number=None, coordinates=None):
    """
    Either return the speaker at given coordinates (azimuth, elevation) or the
    speaker with a specific index number.

    Args:
        index_number (int): index number of the speaker
        coordinates (list of floats): azimuth and elevation of the speaker

    Returns:
        int: index number of the speaker (0 to 47)
        int: channel of the processor the speaker is attached to (1 to 24)
        int: index of the processor the speaker is attached to (1 or 2)
        float: azimuth angle of the target speaker
        float: elevation angle of the target speaker
        int: integer value of the bitmask for the LED at speaker position
        int: index of the processor the LED is attached to (1 or 2)
    """
    row = pd.DataFrame()
    if (index_number is None and coordinates is None) or (index_number is not None and coordinates is not None):
        raise ValueError("You have to specify a the index OR coordinates of the speaker!")
    if index_number is not None:
        row = TABLE.loc[(TABLE.index_number == index_number)]
    elif coordinates is not None:
        if len(coordinates) != 2:
            raise ValueError("Coordinates must have two elements: azimuth and elevation!")
        row = TABLE.loc[(TABLE.azi == coordinates[0]) & (TABLE.ele == coordinates[1])]
    if len(row) > 1:
        logging.warning("More or then one row returned!")
    elif len(row) == 0:
        logging.warning("No entry found that matches the criterion!")
    return row


def get_speaker_list(speaker_list):
    """
    Specify a list of either indices or coordinates and call get_speaker()
    for each element of the list.

    Args:
        speaker_list (list or list of lists): indices or coordinates of speakers.

    Returns:
        list of lists: rows from _table corresponding to the list.
            each sub list contains all the variable returned by get_speaker()
    """
    speakers = pd.DataFrame()
    if (all(isinstance(x, int) for x in speaker_list) or  # list contains indices
            all(isinstance(x, np.int64) for x in speaker_list)):
        speakers = [get_speaker(index_number=i) for i in speaker_list]
        speakers = [df.set_index('index_number') for df in speakers]
        speakers = pd.concat(speakers)
    elif (all(isinstance(x, tuple) for x in speaker_list) or  # list contains coords
          all(isinstance(x, list) for x in speaker_list)):
        speakers = [get_speaker(coordinates=i) for i in speaker_list]
        speakers = [df.set_index('index_number') for df in speakers]
        speakers = pd.concat(speakers)
    if len(speaker_list) == 0:
        logging.warning("No speakers found that match the criteria!")
    return speakers


def all_leds():
    # Temporary hack: return all speakers from the table which have a LED attached
    return TABLE.dropna()


def shift_setup(delta_azi, delta_ele):
    """
    Shift the setup (relative to the lister) by adding some delta value
    in azimuth and elevation. This can be used when chaning the position of
    the chair where the listener is sitting - moving the chair to the right
    is equivalent to shifting the setup to the left. Changes are not saved to
    the speaker table.

    Args:
        delta_azi (float): azimuth by which the setup is shifted, positive value means shifting right
        delta_ele (float): elevation by which the setup is shifted, positive value means shifting up
    """
    global TABLE
    TABLE.azi += delta_azi  # azimuth
    TABLE.ele += delta_ele  # elevation
    logging.info(f"shifting the loudspeaker array by {delta_azi} in azimuth and {delta_ele} in elevation")


def set_signal_and_speaker(signal, speaker, calibrate=True):
    """
    Load a signal into the processor buffer and set the output channel to match the speaker.
    The processor is chosen automatically depending on the speaker.

        Args:
            signal (array-like): signal to load to the buffer, must be one-dimensional
            speaker : speaker to play the signal from, can be index number or [azimuth, elevation]
            calibrate (bool): if True (=default) apply loudspeaker equalization
    """
    signal = slab.Sound(signal)
    if isinstance(speaker, (list, tuple)):
        speaker = get_speaker(coordinates=speaker)
    elif isinstance(speaker, (int, np.int64, np.int32)):
        speaker = get_speaker(index_number=speaker)
    elif isinstance(speaker, pd.Series):
        pass
    else:
        raise ValueError(f"Input {speaker} for argument speaker is not valid! \n"
                         "Specify either an index number or coordinates of the speaker!")
    if calibrate:
        logging.info('Applying calibration.')  # apply level and frequency calibration
        to_play = apply_calibration(signal, speaker)
    else:
        to_play = signal
    PROCESSORS.write(tag='chan', value=speaker.channel.iloc[0], procs=speaker.analog_proc)
    PROCESSORS.write(tag='data', value=to_play.data, procs=speaker.analog_proc)
    other_procs = list(TABLE["analog_proc"].unique())
    other_procs.remove(speaker.analog_proc.iloc[0])  # set the analog output of other procs to non existent number 99
    PROCESSORS.write(tag='chan', value=99, procs=other_procs)


def apply_calibration(signal, speaker, level=True, frequency=True):
    """
    Apply level correction and frequency equalization to a signal

    Args:
        signal: signal to calibrate
        speaker: index number, coordinates or row from the speaker table. Determines which calibration is used
    Returns:
        slab.Sound: calibrated copy of signal
    """
    if not bool(CALIBRATIONDICT):
        logging.warning("Setup is not calibrated! Returning the signal unchanged...")
        return signal
    else:
        signal = slab.Sound(signal)
        if isinstance(speaker, (int, np.int64, np.int32)):
            speaker = get_speaker(index_number=speaker)
        elif isinstance(speaker, (list, tuple)):
            speaker = get_speaker(coordinates=speaker)
        elif not isinstance(speaker, (pd.Series, pd.DataFrame)):
            raise ValueError("Argument speaker must be a index number, coordinates or table row of a speaker!")
        speaker_calibration = CALIBRATIONDICT[str(speaker.index_number.iloc[0])]
        calibrated_signal = deepcopy(signal)
        if level:
            calibrated_signal.level *= speaker_calibration["level"]
        if frequency:
            calibrated_signal = speaker_calibration["filter"].apply(calibrated_signal)
        return calibrated_signal


def get_recording_delay(distance=1.6, sample_rate=48828, play_from=None, rec_from=None):
    """
        Calculate the delay it takes for played sound to be recorded. Depends
        on the distance of the microphone from the speaker and on the processors
        digital-to-analog and analog-to-digital conversion delays.

        Args:
            distance (float): distance between listener and speaker array in meters
            sample_rate (int): sample rate under which the system is running
            play_from (str): processor used for digital to analog conversion
            rec_from (str): processor used for analog to digital conversion

    """
    n_sound_traveling = int(distance / 343 * sample_rate)
    if play_from:
        if play_from == "RX8":
            n_da = 24
        elif play_from == "RP2":
            n_da = 30
        else:
            logging.warning(f"dont know D/A-delay for processor type {play_from}...")
            n_da = 0
    else:
        n_da = 0
    if rec_from:
        if rec_from == "RX8":
            n_ad = 47
        elif rec_from == "RP2":
            n_ad = 65
        else:
            logging.warning(f"dont know A/D-delay for processor type {rec_from}...")
            n_ad = 0
    else:
        n_ad = 0
    return n_sound_traveling + n_da + n_ad


def get_headpose(convert=True, average=True, n=1):
    """Wrapper for the get headpose method of the camera class"""
    if isinstance(CAMERAS, camera.Cameras):
        ele, azi = CAMERAS.get_headpose(convert=convert, average=average, n=n)
        return ele, azi
    else:
        logging.warning("Cameras were not initialized...")
        return False
    

def check_pose(fix=(0, 0), var=10):
    """
    Check if the head pose is directed towards the fixation point

    Args:
        fix: azimuth and elevation of the fixation point
        var: degrees, the pose is allowed to deviate from the fixation point in azimuth and elevations
    Returns:
        bool: True if difference between pose and fix is smaller than var, False otherwise
    """

    if isinstance(CAMERAS, camera.Cameras):
        ele, azi = CAMERAS.get_headpose(convert=True, average=True, n=1)
        if azi is np.nan:  # if the camera is not calibrated in one direction, NaN will be returned -> ignore
            azi = fix[0]
        if ele is np.nan:
            ele = fix[1]
        if np.abs(azi - fix[0]) > var or np.abs(ele - fix[1]) > var:
            return False
        else:
            return True
    else:
        logging.warning("Cameras were not initialized...")
        return False


# functions implementing complete procedures:
def play_start_sound(speaker=23):
    """
    Load and play the sound that signals the start and end of an experiment/block
    """
    start = slab.Sound.read(DIR/"data"/"sounds"/"start.wav")
    set_signal_and_speaker(signal=start, speaker=speaker)
    play_and_wait()


def play_warning_sound(duration=.5, speaker=23):
    """
    Load and play the sound that signals a warning (for example if the listener is in the wrong position)
    """
    warning = slab.Sound.clicktrain(duration=duration)
    set_signal_and_speaker(signal=warning, speaker=speaker)
    play_and_wait()


def calibrate_camera(targets, n_reps=1, n_images=5):
    """
    Calibrate all cameras by lighting up a series of LEDs and estimate the pose when the head is pointed
    towards the currently lit LED. This results in a list of world and camera coordinates which is used to
    calibrate the cameras.

    Args:
        targets (pandas DataFrame): rows from the speaker table. The speakers must have a LED attached
        n_reps(int): number of repetitions for each target
        n_images(int): number of images taken for each head pose estimate
    Returns:
        pandas DataFrame: camera and world coordinates acquired (calibration is performed automatically)
    """
    if not isinstance(CAMERAS, camera.Cameras):
        raise ValueError("Camera must be initialized before calibration!")
    coords = pd.DataFrame(columns=["azi_cam", "azi_world", "ele_cam", "ele_world", "cam", "frame", "n"])
    if not PROCESSORS.mode == "cam_calibration":  # initialize setup in camera calibration mode
        PROCESSORS.initialize_default(mode="cam_calibration")
    targets = [targets.loc[i] for i in targets.index]
    seq = slab.Trialsequence(n_reps=n_reps, conditions=targets)
    for trial in seq:
        logging.info(f"trial nr {seq.this_n}: \n target at elevation of {trial.ele} and azimuth of {trial.azi}")
        PROCESSORS.write(tag="bitmask", value=int(trial.bit), procs=trial.digital_proc)
        wait_for_button()
        pose = CAMERAS.get_headpose(average=False, convert=False, n=n_images)
        pose.insert(0, "n", seq.this_n)
        pose = pose.rename(columns={"azi": "azi_cam", "ele": "ele_cam"})
        pose.insert(2, "ele_world", trial.ele)
        pose.insert(4, "azi_world", trial.azi)
        pose = pose.dropna()
        coords = coords.append(pose, ignore_index=True, sort=True)
        PROCESSORS.write(tag="bitmask", value=0, procs=trial.digital_proc)
    CAMERAS.calibrate(coords, plot=True)
    return coords


def localization_test_freefield(targets, duration=0.5, n_reps=1, n_images=5, visual=False):
    """
    Run a basic localization test where the same sound is played from different
    speakers in randomized order, without playing the same position twice in
    a row. After every trial the presentation is paused and the listener has
    to localize the sound source by pointing the head towards the source and
    pressing the response button. The cameras need to be calibrated before the
    test! After every trial the listener has to point to the middle speaker at
    0 elevation and azimuth and press the button to indicate the next trial.

    Args:
        targets : rows from the speaker table or index numbers of the speakers.
        duration (float): duration of the noise played from the target positions in seconds
        n_reps(int): number of repetitions for each target
        n_images(int): number of images taken for each head pose estimate
        visual(bool): If True, light a LED at the target position - the speakers must have a LED attached
    Returns:
        instance of slab.Trialsequence: the response is stored in the data attribute as tuples with (azimuth, elevation)
    """
    if not isinstance(CAMERAS, camera.Cameras) and CAMERAS.calibration is not None:
        raise ValueError("Camera must be initialized and calibrated before localization test!")
    if not PROCESSORS.mode == "loctest_freefield":
        PROCESSORS.initialize_default(mode="loctest_freefield")
    PROCESSORS.write(tag="playbuflen", value=int(slab.signal._default_samplerate*duration), procs=["RX81", "RX82"])
    if visual is True:
        if targets.bit.isnull.sum():
            raise ValueError("All speakers must have a LED attached for a test with visual cues")
    targets = [targets.loc[i] for i in targets.index]  # make list from data frame
    seq = slab.Trialsequence(targets, n_reps, kind="non_repeating")
    play_start_sound()
    for trial in seq:
        wait_for_button()
        while check_pose(fix=[0, 0]) is None:  # check if head is in position
            play_warning_sound()
            wait_for_button()
        sound = slab.Sound.pinknoise(duration=duration)
        set_signal_and_speaker(signal=sound, speaker=trial.index_number)
        seq = _loctest_trial(trial, seq, visual, n_images)
    play_start_sound()
    return seq


def localization_test_headphones(targets, signals, n_reps=1, n_images=5, visual=False):
    """
    Run a basic localization test where previously recorded/generated binaural sound are played via headphones.
    The procedure is the same as in localization_test_freefield().

    Args:
        targets : rows from the speaker table or index numbers of the speakers.
        signals (array-like) : binaural sounds that are played. Must be ordered corresponding to the targets (first
            element of signals is played for the first row of targets etc.). If the elements of signals are
            instances of slab.Precomputed, a random one is drawn in each trial (useful if you don't want to repeat
            the exact same sound in each trial)
        n_reps(int): number of repetitions for each target
        n_images(int): number of images taken for each head pose estimate
        visual(bool): If True, light a LED at the target position - the speakers must have a LED attached
    Returns:
        instance of slab.Trialsequence: the response is stored in the data attribute as tuples with (azimuth, elevation)
    """

    if not isinstance(CAMERAS, camera.Cameras) and CAMERAS.calibration is not None:
        raise ValueError("Camera must be initialized and calibrated before localization test!")
    if not PROCESSORS.mode == "loctest_headphones":
        PROCESSORS.initialize_default(mode="loctest_headphones")
    if not len(signals) == len(targets):
        raise ValueError("There must be one signal for each target!")
    if visual is True:
        if targets.bit.isnull.sum():
            raise ValueError("All speakers must have a LED attached for a test with visual cues")
    targets = [targets.loc[i] for i in targets.index]  # make list from data frame
    seq = slab.Trialsequence(targets, n_reps, kind="non_repeating")
    play_start_sound()
    for trial in seq:
        signal = signals[trial.index_number]  # get the signal corresponding to the target
        if isinstance(signal, slab.Precomputed):  # if signal is precomputed, pick a random one
            signal = signal[np.random.randint(len(signal))]
        try:
            signal = slab.Binaural(signal)
        except IndexError:
            logging.warning("Binaural sounds must have exactly two channels!")
        wait_for_button()
        while check_pose(fix=[0, 0]) is None:  # check if head is in position
            play_warning_sound()
            wait_for_button()
        # write sound into buffer
        PROCESSORS.write(tag="playbuflen", value=signal.nsamples, procs="RP2")
        PROCESSORS.write(tag="data_l", value=signal.left.data.flatten(), procs="RP2")
        PROCESSORS.write(tag="data_r", value=signal.right.data.flatten(), procs="RP2")
        seq = _loctest_trial(trial, seq, visual, n_images)
    play_start_sound()
    return seq


def _loctest_trial(trial, seq, visual, n_images):
    """do a single trial in a localization test experiment: turn on LED (optional), play and wait for button press,
     get head pose, turn led of, write response in trial sequence and return the sequence"""
    if visual is True:  # turn LED on
        PROCESSORS.write(tag="bitmask", value=trial.bit, procs=trial.digital_proc)
    play_and_wait_for_button()
    pose = CAMERAS.get_headpose(convert=True, average=True, n=n_images)
    if visual is True:  # turn LED off
        PROCESSORS.write(tag="bitmask", value=0, procs=trial.digital_proc)
    seq.add_response(pose)
    return seq


def equalize_speakers(speakers="all", target_speaker=23, bandwidth=1/10, db_tresh=80,
                      low_cutoff=200, high_cutoff=16000, alpha=1.0, plot=False, test=True):
    """
    Equalize the loudspeaker array in two steps. First: equalize over all
    level differences by a constant for each speaker. Second: remove spectral
    difference by inverse filtering. For more details on how the
    inverse filters are computed see the documentation of slab.Filter.equalizing_filterbank
    """
    global CALIBRATIONDICT
    logging.info('Starting calibration.')
    if not PROCESSORS.mode == "play_rec":
        PROCESSORS.initialize_default(mode="play_and_record")
    sig = slab.Sound.chirp(duration=0.05, from_frequency=low_cutoff, to_frequency=high_cutoff)
    if speakers == "all":  # use the whole speaker table
        speaker_list = TABLE
    elif isinstance(speakers, list):  # use a subset of speakers
        speaker_list = get_speaker_list(speakers)
    else:
        raise ValueError("Argument speakers must be a list of interers or 'all'!")
    calibration_lvls = _level_equalization(sig, speaker_list, target_speaker, db_tresh)
    filter_bank, rec = _frequency_equalization(sig, speaker_list, target_speaker, calibration_lvls,
                                               bandwidth, low_cutoff, high_cutoff, alpha, db_tresh)
    # if plot:  # save plot for each speaker
    #     for i in range(rec.nchannels):
    #         _plot_equalization(target_speaker, rec.channel(i),
    #                            fbank.channel(i), i)
    for i in range(TABLE.shape[0]):  # write level and frequency equalization into one dictionary
        CALIBRATIONDICT[str(i)] = {"level": calibration_lvls[i], "filter": filter_bank.channel(i)}
    if CALIBRATIONFILE.exists():  # move the old calibration to the log folder
        date = datetime.datetime.now().strftime("_%Y-%m-%d-%H-%M-%S")
        rename_previous = DIR / 'data' / Path("log/" + _calibration_file.stem + date + _calibration_file.suffix)
        CALIBRATIONFILE.rename(rename_previous)
    with open(CALIBRATIONFILE, 'wb') as f:  # save the newly recorded calibration
        pickle.dump(CALIBRATIONDICT, f, pickle.HIGHEST_PROTOCOL)
    logging.info('Calibration completed.')


def _level_equalization(sig, speaker_list, target_speaker, db_thresh):
    """
    Record the signal from each speaker in the list and return the level of each
    speaker relative to the target speaker(target speaker must be in the list)
    """
    rec = []
    for i in range(speaker_list.shape[0]):
        row = speaker_list.loc[i]
        rec.append(play_and_record(row.index_number, sig, apply_calibration=False))
        if row.index_number == target_speaker:
            target = rec[-1]
    rec = slab.Sound(rec)
    rec.data[:, rec.level < db_thresh] = target.data  # thresholding
    return target.level / rec.level


def _frequency_equalization(sig, speaker_list, target_speaker, calibration_lvls, bandwidth,
                            low_cutoff, high_cutoff, alpha, db_thresh):
    """
    play the level-equalized signal, record and compute and a bank of inverse filter
    to equalize each speaker relative to the target one. Return filterbank and recordings
    """
    rec = []
    for i in range(speaker_list.shape[0]):
        row = speaker_list.loc[i]
        modulated_sig = deepcopy(sig)  # copy signal and correct for lvl difference
        modulated_sig.level *= calibration_lvls[row.index_number]
        rec.append(play_and_record(row.index_number, modulated_sig, apply_calibration=False))
        if row.index_number == target_speaker:
            target = rec[-1]
    rec = slab.Sound(rec)
    # set recordings which are below the threshold or which are from exluded speaker
    # equal to the target so that the resulting frequency filter will be flat
    rec.data[:, rec.level < db_thresh] = target.data

    filter_bank = slab.Filter.equalizing_filterbank(target=target, signal=rec, low_cutoff=low_cutoff,
                                                    high_cutoff=high_cutoff, bandwidth=bandwidth, alpha=alpha)
    # check for notches in the filter:
    transfer_function = filter_bank.tf(show=False)[1][0:900, :]
    if (transfer_function < -30).sum() > 0:
        logging.warning(f"The filter for speaker {row.index_number} at azimuth {row.azi} and elevation {row.ele} /n"
                        "contains deep notches - adjust the equalization parameters!")

    return filter_bank, rec


def check_equalization(sig, speakers="all", max_diff=5, db_thresh=80):
    """
    Test the effectiveness of the speaker equalization
    """
    fig, ax = plt.subplots(3, 2, sharex=True)
    # recordings without, with level and with complete (level+frequency) equalization
    rec_raw, rec_lvl_eq, rec_freq_eq = [], [], []
    if speakers == "all":  # use the whole speaker table
        speaker_list = TABLE
    elif isinstance(speakers, list):  # use a subset of speakers
        speaker_list = get_speaker_list(speakers)
    else:
        raise ValueError("Speakers must be 'all' or a list of indices/coordinates!")
    for i in range(speaker_list.shape[0]):
        row = speaker_list.loc[i]
        sig2 = apply_calibration(sig, speaker=row.index_number, level=True, frequency=False)  # only level equalization
        sig3 = apply_calibration(sig, speaker=row.index_number, level=True, frequency=True)  # level and frequency
        rec_raw.append(play_and_record(row.index_number, sig, calibrate=False))
        rec_lvl_eq.append(play_and_record(row.index_number, sig2, calibrate=False))
        rec_freq_eq.append(play_and_record(row.index_number, sig3, calibrate=False))
    for i, rec in enumerate([rec_raw, rec_lvl_eq, rec_freq_eq]):
        rec = slab.Sound(rec)
        rec.data = rec.data[:, rec.level > db_thresh]
        rec.spectrum(axes=ax[i, 0], show=False)
        spectral_range(rec, plot=ax[i, 1], thresh=max_diff, log=False)
    plt.show()

    return slab.Sound(rec_raw), slab.Sound(rec_lvl_eq), slab.Sound(rec_freq_eq)


def spectral_range(signal, bandwidth=1 / 5, low_cutoff=50, high_cutoff=20000, thresh=3,
                   plot=True, log=True):
    """
    Compute the range of differences in power spectrum for all channels in
    the signal. The signal is devided into bands of equivalent rectangular
    bandwidth (ERB - see More&Glasberg 1982) and the level is computed for
    each frequency band and each channel in the recording. To show the range
    of spectral difference across channels the minimum and maximum levels
    across channels are computed. Can be used for example to check the
    effect of loud speaker equalization.
    """
    # TODO: this really should be part of the slab.Sound file
    # generate ERB-spaced filterbank:
    fbank = slab.Filter.cos_filterbank(length=1000, bandwidth=bandwidth,
                                       low_cutoff=low_cutoff, high_cutoff=high_cutoff,
                                       samplerate=signal.samplerate)
    center_freqs, _, _ = slab.Filter._center_freqs(low_cutoff, high_cutoff, bandwidth)
    center_freqs = slab.Filter._erb2freq(center_freqs)
    # create arrays to write data into:
    levels = np.zeros((signal.nchannels, fbank.nchannels))
    max_level, min_level = np.zeros(fbank.nchannels), np.zeros(fbank.nchannels)
    for i in range(signal.nchannels):  # compute ERB levels for each channel
        levels[i] = fbank.apply(signal.channel(i)).level
    for i in range(fbank.nchannels):  # find max and min for each frequency
        max_level[i] = max(levels[:, i])
        min_level[i] = min(levels[:, i])
    difference = max_level - min_level
    if plot is True or isinstance(plot, Axes):
        if isinstance(plot, Axes):
            ax = plot
        else:
            fig, ax = plt.subplots(1)
        # frequencies where the difference exceeds the threshold
        bads = np.where(difference > thresh)[0]
        for y in [max_level, min_level]:
            if log is True:
                ax.semilogx(center_freqs, y, color="black", linestyle="--")
            else:
                ax.plot(center_freqs, y, color="black", linestyle="--")
        for bad in bads:
            ax.fill_between(center_freqs[bad - 1:bad + 1], max_level[bad - 1:bad + 1],
                            min_level[bad - 1:bad + 1], color="red", alpha=.6)
    return difference


def play_and_record(speaker_nr, sig, compensate_delay=True, compensate_level=True, calibrate=False):
    """
    Play the signal from a speaker and return the recording. Delay compensation
    means making the buffer of the recording processor n samples longer and then
    throwing the first n samples away when returning the recording so sig and
    rec still have the same legth. For this to work, the circuits rec_buf.rcx
    and play_buf.rcx have to be initialized on RP2 and RX8s and the mic must
    be plugged in.
    Parameters:
        speaker_nr: integer between 1 and 48, index number of the speaker
        sig: instance of slab.Sound, signal that is played from the speaker
        compensate_delay: bool, compensate the delay between play and record
    Returns:
        rec: 1-D array, recorded signal
    """
    if PROCESSORS.mode == "play_birec":
        binaural = True  # 2 channel recording
    elif PROCESSORS.mode == "play_rec":
        binaural = False  # record single channel
    else:
        raise ValueError("Setup must be initialized in mode 'play_rec' or 'play_birec'!")
    PROCESSORS.write(tag="playbuflen", value=sig.nsamples, procs=["RX81", "RX82"])
    if compensate_delay:
        n_delay = get_recording_delay(play_from="RX8", rec_from="RP2")
        n_delay += 50  # make the delay a bit larger, just to be sure
    else:
        n_delay = 0
    PROCESSORS.write(tag="playbuflen", value=sig.nsamples, procs=["RX81", "RX82"])
    PROCESSORS.write(tag="playbuflen", value=sig.nsamples + n_delay, procs="RP2")
    set_signal_and_speaker(sig, speaker_nr, calibrate)
    play_and_wait()
    if binaural is False:  # read the data from buffer and skip the first n_delay samples
        rec = PROCESSORS.read(tag='data', proc='RP2', n_samples=sig.nsamples + n_delay)[n_delay:]
        rec = slab.Sound(rec)
    else:  # read data for left and right ear from buffer
        rec_l = PROCESSORS.read(tag='datal', proc='RP2', n_samples=sig.nsamples + n_delay)[n_delay:]
        rec_r = PROCESSORS.read(tag='datar', proc='RP2', n_samples=sig.nsamples + n_delay)[n_delay:]
        rec = slab.Binaural([rec_l, rec_r])
    if compensate_level:
        if binaural:
            iid = rec.left.level - rec.right.level
            rec.level = sig.level
            rec.left.level += iid
        else:
            rec.level = sig.level
    return rec
