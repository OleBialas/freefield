import time
from pathlib import Path
import glob
from copy import deepcopy
import numpy as np
import slab
from typing import Union, Optional
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from freefield import camera
import pandas as pd
import datetime
from freefield import DIR
from freefield import Devices as Devs
import logging

logging.basicConfig(level=logging.WARNING)
slab.Signal.set_default_samplerate(48828)  # default samplerate for generating sounds, filters etc.
# Initialize global variables:
Cameras = None
Devices = Devs()
_calibration_freqs = None  # filters for frequency equalization
_calibration_lvls = None  # calibration to equalize levels
_table = pd.DataFrame()  # numbers and coordinates of all loudspeakers


def initialize_setup(setup, default_mode=None, device_list=None, zbus=True, connection="GB", camera_type="flir"):
    """
    Initialize the devices and load table and calibration for setup.

    We are using two different 48-channel setups. A 3-dimensional 'dome' and
    a horizontal 'arc'. For each setup there is a table describing the position
    and channel of each loudspeaker as well as calibration files. This function
    loads those files and stores them in global variables. This is necessary
    for most of the other functions to work.

    Args:
        setup: determines which files to load, can be 'dome' or 'arc'
        default_mode: initialize the setup using one of the defaults, see devices.initialize_default
        device_list: if not using a default, specify the devices in a list, see devices.initialize_devices
        zbus: whether or not to initialize the zbus interface
        connection: type of connection to devices, can be "GB" (optical) or "USB"
        camera_type: kind of camera that is initialized. Can be "webcam", "flir" or None
    """

    # TODO: put level and frequency equalization in one common file
    global _calibration_freqs, _calibration_lvls, _table, Devices, Cameras
    # initialize devices
    if bool(device_list) == bool(default_mode):
        raise ValueError("You have to specify a device_list OR a default_mode")
    if device_list is not None:
        Devices.initialize_devices(device_list, zbus, connection)
    elif default_mode is not None:
        Devices.initialize_default(default_mode)
    if camera_type is not None:
        _cameras = camera.initialize_cameras(camera_type)
    # get the correct speaker table and calibration files for the setup
    if setup == 'arc':
        freq_calibration_file = DIR / 'data' / Path('frequency_calibration_arc.npy')
        lvl_calibration_file = DIR / 'data' / Path('level_calibration_arc.npy')
        table_file = DIR / 'data' / 'tables' / Path('speakertable_arc.txt')
    elif setup == 'dome':
        freq_calibration_file = DIR / 'data' / Path('frequency_calibration_dome.npy')
        lvl_calibration_file = DIR / 'data' / Path('level_calibration_dome.npy')
        table_file = DIR / 'data' / 'tables' / Path('speakertable_dome.txt')
    else:
        raise ValueError("Unknown device! Use 'arc' or 'dome'.")
    logging.info(f'Speaker configuration set to {setup}.')
    # lambdas provide default values of 0 if azi or ele are not in the file
    _table = pd.read_csv(table_file, dtype={"index_number": "Int64", "channel": "Int64", "analog_proc": "category",
                         "azi": float, "ele": float, "bit": "Int64", "digital_proc": "category"})
    logging.info('Speaker table loaded.')
    if freq_calibration_file.exists():
        _calibration_freqs = slab.Filter.load(freq_calibration_file)
        logging.info('Frequency-calibration filters loaded.')
    else:
        logging.warning('Setup not frequency-calibrated...')
    if lvl_calibration_file.exists():
        _calibration_lvls = np.load(lvl_calibration_file)
        logging.info('Level-calibration loaded.')
    else:
        logging.warning('Setup not level-calibrated...')


def wait_to_finish_playing(proc: str = "all", tag: str = "playback") -> None:
    """
    Busy wait until the devices finished playing.

    For this function to work, the rcx-circuit must have a tag that is 1
    while output is generated and 0 otherwise. The default name for this
    kind of tag is "playback". "playback" is read repeatedly for each device
    followed by a short sleep if the value is 1.

    Args:
        proc (str, list of str): name(s) of the processor(s) to wait for.
        tag (str): name of the tag that signals if something is played
    """
    if proc == "all":
        proc = list(Devices.procs.keys())
    elif isinstance(proc, str):
        proc = [proc]
    logging.info(f'Waiting for {tag} on {proc}.')
    while any(Devices.read(tag, n_samples=1, proc=p) for p in proc):
        time.sleep(0.01)
    logging.info('Done waiting.')


def wait_for_button() -> None:
    while not Devices.read(tag="response", proc="RP2"):
        time.sleep(0.1)  # wait until button is pressed


def play_and_wait() -> None:
    Devices.trigger()
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
        int: channel of the device the speaker is attached to (1 to 24)
        int: index of the device the speaker is attached to (1 or 2)
        float: azimuth angle of the target speaker
        float: elevation angle of the target speaker
        int: integer value of the bitmask for the LED at speaker position
        int: index of the device the LED is attached to (1 or 2)
    """
    row = pd.DataFrame()
    if (index_number is None and coordinates is None) or (index_number is not None and coordinates is not None):
        raise ValueError("You have to specify a the index OR coordinates of the speaker!")
    if index_number is not None:
        row = _table.loc[(_table.index_number == index_number)]
    elif coordinates is not None:
        if len(coordinates) != 2:
            raise ValueError("Coordinates must have two elements: azimuth and elevation!")
        row = _table.loc[(_table.azi == coordinates[0]) & (_table.ele == coordinates[1])]
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
    return _table.dropna()


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
    global _table
    _table.azi += delta_azi  # azimuth
    _table.ele += delta_ele  # elevation
    logging.info("shifting the loudspeaker array by % s degree in azimuth / n"
                 "and % s degree in elevation" % (delta_azi, delta[1]))


def set_signal_and_speaker(signal, speaker, apply_calibration=True):
    """
    Load a signal into the device buffer and set the output channel to match the speaker.
    The device is chosen automatically depending on the speaker.

        Args:
            signal (array-like): signal to load to the buffer, must be one-dimensional
            speaker : speaker to play the signal from, can be index number or [azimuth, elevation]
            apply_calibration (bool): if True (=default) apply loudspeaker equalization
    """

    signal = slab.Sound(signal)
    if isinstance(speaker, list):
        speaker = get_speaker(coordinates=speaker)
    elif isinstance(speaker, int):
        speaker = get_speaker(index_number=speaker)
    to_play = deepcopy(signal)  # copy the signal so the original is not changed
    if apply_calibration:
        if _calibration_freqs is None or _calibration_lvls is None:
            logging.warning("Setup is not calibrated!")
        elif isinstance(_calibration_freqs, slab.Filter) and isinstance(_calibration_lvls, np.ndarray):
            logging.info('Applying calibration.')  # apply level and frequency calibration
            to_play.level *= _calibration_lvls[int(speaker)]
            to_play = _calibration_freqs.channel(int(speaker)).apply(to_play)
    Devices.write(tag='chan', value=speaker.channel.iloc[0], procs=speaker.analog_proc)
    Devices.write(tag='data', value=to_play.data, procs=speaker.analog_proc)
    other_procs = list(_table["analog_proc"].unique())
    other_procs.remove(speaker.analog_proc)  # set the analog output of other procs to non existent number 99
    Devices.write(tag='chan', value=99, procs=speaker.analog_proc)


def get_recording_delay(distance=1.6, sample_rate=48828, play_device=None, rec_device=None):
    """
        Calculate the delay it takes for played sound to be recorded. Depends
        on the distance of the microphone from the speaker and on the devices
        digital-to-analog and analog-to-digital conversion delays.

        Args:
            distance (float): distance between listener and speaker array in meters
            sample_rate (int): sample rate under which the system is running
            play_device (str): device used for digital to analog conversion
            rec_device (str): device used for analog to digital conversion

    """
    n_sound_traveling = int(distance / 343 * sample_rate)
    if play_device:
        if play_device == "RX8":
            n_da = 24
        elif play_device == "RP2":
            n_da = 30
        else:
            logging.warning(f"dont know D/A-delay for device type {play_device}...")
            n_da = 0
    else:
        n_da = 0
    if rec_device:
        if rec_device == "RX8":
            n_ad = 47
        elif rec_device == "RP2":
            n_ad = 65
        else:
            logging.warning(f"dont know A/D-delay for device type {rec_device}...")
            n_ad = 0
    else:
        n_ad = 0
    return n_sound_traveling + n_da + n_ad


def check_pose(fix=(0, 0), var=10):
    """
    Check if the head pose is directed towards the fixation point

    Args:
        fix: azimuth and elevation of the fixation point
        var: degrees, the pose is allowed to deviate from the fixation point in azimuth and elevations
    Returns:
        bool: True if difference between pose and fix is smaller than var, False otherwise
    """

    if isinstance(Cameras, camera.Cameras):
        ele, azi = Cameras.get_headpose(convert=True, average=True, n=1)
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
    if not isinstance(Cameras, camera.Cameras):
        raise ValueError("Camera must be initialized before calibration!")
    coords = pd.DataFrame(columns=["azi_cam", "azi_world", "ele_cam", "ele_world", "cam", "frame", "n"])
    if not Devices.mode == "cam_calibration":  # initialize setup in camera calibration mode
        Devices.initialize_default(mode="cam_calibration")
    targets = [targets.loc[i] for i in targets.index]
    seq = slab.Trialsequence(n_reps=n_reps, conditions=targets)
    for trial in seq:
        logging.info(f"trial nr {seq.this_n}: \n target at elevation of {trial.ele} and azimuth of {trial.azi}")
        Devices.write(tag="bitmask", value=int(trial.bit), procs=trial.digital_proc)
        wait_for_button()
        pose = Cameras.get_headpose(average=False, convert=False, n=n_images)
        pose.insert(0, "n", seq.this_n)
        pose = pose.rename(columns={"azi": "azi_cam", "ele": "ele_cam"})
        pose.insert(2, "ele_world", trial.ele)
        pose.insert(4, "azi_world", trial.azi)
        pose = pose.dropna()
        coords = coords.append(pose, ignore_index=True, sort=True)
        Devices.write(tag="bitmask", value=0, procs=trial.digital_proc)
    Cameras.calibrate(coords, plot=True)
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
        targets (pandas DataFrame): rows from the speaker table.
        duration (float): duration of the noise played from the target positions in seconds
        n_reps(int): number of repetitions for each target
        n_images(int): number of images taken for each head pose estimate
        visual(bool): If True, light a LED at the target position - the speakers must have a LED attached
    Returns:
        pandas DataFrame: camera and world coordinates acquired (calibration is performed automatically)
    """
    if not isinstance(Cameras, camera.Cameras) or Cameras.calibration is not None:
        raise ValueError("Camera must be initialized and calibrated before localization test!")
    if not Devices.mode == "loctest_freefield":
        Devices.initialize_default(mode="loctest_freefield")
    warning = slab.Sound.clicktrain(duration=duration)
    Devices.write(tag="playbuflen", value=int(slab.signal._default_samplerate*duration), procs=["RX81", "RX82"])
    if visual is True:
        if targets.bit.isnull.sum():
            raise ValueError("All speakers must have a LED attached for a test with visual cues")
    speakers = [targets.loc[i] for i in targets.index]
    seq = slab.Trialsequence(speakers, n_reps, kind="non_repeating")
    start = slab.Sound.read(DIR/"data"/"sounds"/"start.wav").channel(0)
    set_signal_and_speaker(signal=start, speaker=23)
    play_and_wait()
    for trial in seq:
        wait_for_button()
        while check_pose(fix=[0, 0]) is None:  # check if head is in position
            set_signal_and_speaker(signal=warning, speaker=23)
            play_and_wait_for_button()
        sound = slab.Sound.pinknoise(duration=duration)
        set_signal_and_speaker(signal=sound, speaker=trial.index_number)
        if visual is True:  # turn LED on
            Devices.write(tag="bitmask", value=trial.bit, procs=trial.digital_proc)
        play_and_wait_for_button()
        pose = Cameras.get_headpose(convert=True, average=True, n=n_images)
        if visual is True:  # turn LED off
            Devices.write(tag="bitmask", value=0, procs=trial.digital_proc)
        seq.add_response(pose)
    set_signal_and_speaker(signal=start, speaker=23)  # play sound to signal end
    play_and_wait()
    return seq


def localization_test_headphones(folder, speakers, n_reps=1, visual=False):
    folder = Path(folder)
    if not folder.exists():
        raise ValueError("Folder does not exist!")
    else:
        if not _mode == "localization_test_headphones":
            initialize_devices(mode="localization_test_headphones")
        speakers = speakers_from_list(speakers)
        seq = slab.Trialsequence(speakers, n_reps, kind="non_repeating")
    if visual is True:  # check if speakers have a LED
        if any(np.isnan([s[-1] for s in speakers])):
            raise ValueError("At least one of the selected speakers does not have a LED attached!")
    if camera._cal is None:
        raise ValueError("Camera must be calibrated before localization test!")
    response = _response.copy()
    # play start signal :
    set_variable(variable="playbuflen", value=_signal.nsamples, proc="RP2")
    set_variable(variable="data_l", value=_signal.data.flatten(), proc="RP2")
    set_variable(variable="data_r", value=_signal.data.flatten(), proc="RP2")
    trigger()
    # wait for button press to start the sequence
    while not get_variable(variable="response", proc="RP2"):
        time.sleep(0.01)
    for trial in seq:  # loop trough trial sequence
        _, _, _, azi, ele, bit, proc_bit = trial
        trial = {"azi_target": azi, "ele_target": ele}
        file = Path(np.random.choice(glob.glob(str(folder / ("*azi%s_ele%s*" % (azi, ele))))))
        sound = slab.Binaural(slab.Sound(file))
        # write sound into buffer
        set_variable(variable="playbuflen", value=sound.nsamples, proc="RP2")
        set_variable(variable="data_l", value=sound.left.data.flatten(), proc="RP2")
        set_variable(variable="data_r", value=sound.right.data.flatten(), proc="RP2")
        if visual is True:  # turn on the LED
            set_variable(variable="bitmask", value=bit, proc=proc_bit)
        trigger()
        # wait for response, then estimate the headpose
        while not get_variable(variable="response", proc="RP2"):
            time.sleep(0.01)
        ele, azi = camera.get_headpose(convert=True, average=True, target=(azi, ele))
        if visual is True:  # turn of the LED
            set_variable(variable="bitmask", value=0, proc=proc_bit)
        # append response
        trial["azi_response"], trial["ele_response"] = azi, ele
        response = response.append(trial, ignore_index=True)
        # wait till head is back in position, send warning if not
        head_in_position = False
        while not head_in_position:
            while not get_variable(variable="response", proc="RP2"):
                time.sleep(0.01)
            head_in_position = check_pose()
    # play sound to signal end
    set_variable(variable="playbuflen", value=_signal.nsamples, proc="RP2")
    set_variable(variable="data_l", value=_signal.data.flatten(), proc="RP2")
    set_variable(variable="data_r", value=_signal.data.flatten(), proc="RP2")
    trigger()
    return response


def equalize_speakers(speakers="all", target_speaker=23, bandwidth=1 / 10,
                      low_cutoff=200, high_cutoff=16000, alpha=1.0, plot=False, test=True, exclude=None):
    """
    Equalize the loudspeaker array in two steps. First: equalize over all
    level differences by a constant for each speaker. Second: remove spectral
    differeces by inverse filtering. Argument exlude can be a list of speakers that
    will be excluded from the frequency equalization. For more details on how the
    inverse filters are computed see the documentation of slab.Filter.equalizing_filterbank
    """
    # TODO: check filter shift compensation
    global _calibration_freqs, _calibration_lvls
    printv('Starting calibration.')
    if not _mode == "play_and_record":
        initialize_devices(mode="play_and_record")
    sig = slab.Sound.chirp(duration=0.05, from_freq=low_cutoff, to_freq=high_cutoff)
    sig.level = 95
    if speakers == "all":  # use the whole speaker table
        speaker_list = _table
    else:  # use a subset of speakers
        speaker_list = speakers_from_list(speakers)
    _calibration_lvls = _level_equalization(sig, speaker_list, target_speaker)
    fbank, rec = _frequency_equalization(
        sig, speaker_list, target_speaker, bandwidth, low_cutoff, high_cutoff, alpha, exclude)
    if plot:  # save plot for each speaker
        for i in range(rec.nchannels):
            _plot_equalization(target_speaker, rec.channel(i),
                               fbank.channel(i), i)

    if _freq_calibration_file.exists():
        date = datetime.datetime.now().strftime("_%Y-%m-%d-%H-%M-%S")
        rename_previous = \
            _location.parent / Path("log/" + _freq_calibration_file.stem + date
                                    + _freq_calibration_file.suffix)
        _freq_calibration_file.rename(rename_previous)

    if _lvl_calibration_file.exists():
        date = datetime.datetime.now().strftime("_%Y-%m-%d-%H-%M-%S")
        rename_previous = \
            _location.parent / Path("log/" + _lvl_calibration_file.stem + date
                                    + _lvl_calibration_file.suffix)
        _lvl_calibration_file.rename(rename_previous)
    np.save(_lvl_calibration_file, _calibration_lvls)
    fbank.save(_freq_calibration_file)  # save as 'calibration_arc.npy' or dome.
    printv('Calibration completed.')
    if test:
        test_equalization(sig)


def _level_equalization(sig, speaker_list, target_speaker):
    """
    Record the signal from each speaker in the list and return the level of each
    speaker relative to the target speaker(target speaker must be in the list)
    """
    rec = []
    for row in speaker_list:
        rec.append(play_and_record(row[0], sig, apply_calibration=False))
        if row[0] == target_speaker:
            target = rec[-1]
    rec = slab.Sound(rec)
    rec.data[:, rec.level < _rec_tresh] = target.data  # thresholding
    return target.level / rec.level


def _frequency_equalization(sig, speaker_list, target_speaker, bandwidth,
                            low_cutoff, high_cutoff, alpha, exclude=None):
    """
    play the level-equalized signal, record and compute and a bank of inverse filter
    to equalize each speaker relative to the target one. Return filterbank and recordings
    """
    rec = []
    for row in speaker_list:
        modulated_sig = deepcopy(sig)  # copy signal and correct for lvl difference
        modulated_sig.level *= _calibration_lvls[int(row[0])]
        rec.append(play_and_record(row[0], modulated_sig, apply_calibration=False))
        if row[0] == target_speaker:
            target = rec[-1]
    rec = slab.Sound(rec)
    # set recordings which are below the threshold or which are from exluded speaker
    # equal to the target so that the resulting frequency filter will be flat
    rec.data[:, rec.level < _rec_tresh] = target.data
    if exclude is not None:
        for e in exclude:
            print("excluding speaker %s from frequency equalization..." % (e))
            rec.data[:, e] = target.data[:, 0]
    fbank = slab.Filter.equalizing_filterbank(target=target, signal=rec, low_cutoff=low_cutoff,
                                              high_cutoff=high_cutoff, bandwidth=bandwidth, alpha=alpha)
    # check for notches in the filter:
    dB = fbank.tf(plot=False)[1][0:900, :]
    if (dB < -30).sum() > 0:
        printv("WARNING! The filter contains large notches! You might want to adjust the \n"
               " alpha parameter or set plot=True to see the speakers affected...")
    return fbank, rec


def test_equalization(sig, speakers="all", max_diff=5):
    """
    Test the effectiveness of the speaker equalization
    """
    fig, ax = plt.subplots(3, 2, sharex=True)
    # recordings without, with level and with complete (level+frequency) equalization
    rec_raw, rec_lvl_eq, rec_freq_eq = [], [], []
    if speakers == "all":  # use the whole speaker table
        speaker_list = _table
    else:  # use a subset of speakers
        speaker_list = speakers_from_list(speakers)
    for row in speaker_list:
        sig2 = deepcopy(sig)
        rec_raw.append(play_and_record(row[0], sig2))
        sig2.level *= _calibration_lvls[int(row[0])]
        rec_lvl_eq.append(play_and_record(row[0], sig2))
        sig2 = _calibration_freqs.channel(int(row[0])).apply(sig2)
        # rec_freq_eq.append(play_and_record(row[0], sig2))
        # this should do the same thing:
        rec_freq_eq.append(play_and_record(row[0], sig, apply_calibration=True))
    for i, rec in enumerate([rec_raw, rec_lvl_eq, rec_freq_eq]):
        rec = slab.Sound(rec)
        rec.data = rec.data[:, rec.level > _rec_tresh]
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


def binaural_recording(sound, speaker_nr, compensate_delay=True, apply_calibration=True):
    if _mode != "binaural_recording":
        initialize_devices(mode="binaural_recording")
    level = sound.level
    rec = play_and_record(speaker_nr, sound, compensate_delay, apply_calibration)
    iid = rec.left.level - rec.right.level
    rec.level = level  # correct for level attenuation
    rec.left.level += iid  # keep interaural intensity difference
    return rec


def play_and_record(speaker_nr, sig, compensate_delay=True,
                    apply_calibration=False):
    """
    Play the signal from a speaker and return the recording. Delay compensation
    means making the buffer of the recording device n samples longer and then
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
    # TODO use binaural class for binaural recordings

    if _mode == "binaural_recording":
        binaural = True  # 2 channel recording
    elif _mode == "play_and_record":
        binaural = False  # record single channle
    else:
        raise ValueError("Setup must be initalized in 'play_and_record' for "
                         "single or 'binaural' for two channel recording!"
                         "\n current mode is %s" % (_mode))
    set_variable(variable="playbuflen", value=sig.nsamples, proc="RX8s")
    if compensate_delay:
        n_delay = get_recording_delay(play_device="RX8", rec_device="RP2")
        n_delay += 50  # make the delay a bit larger, just to be sure
    else:
        n_delay = 0
    set_variable(variable="playbuflen", value=sig.nsamples, proc="RX8s")
    set_variable(variable="playbuflen", value=sig.nsamples + n_delay, proc="RP2")
    set_signal_and_speaker(sig, speaker_nr, apply_calibration)
    trigger()  # start playing and wait
    wait_to_finish_playing(proc="all")
    if binaural is False:
        rec = get_variable(variable='data', proc='RP2',
                           n_samples=sig.nsamples + n_delay)[n_delay:]
        rec = slab.Sound(rec)
    if binaural is True:
        recl = get_variable(variable='datal', proc='RP2',
                            n_samples=sig.nsamples + n_delay)[n_delay:]
        recr = get_variable(variable='datar', proc='RP2',
                            n_samples=sig.nsamples + n_delay)[n_delay:]
        rec = slab.Binaural([recl, recr])
    return rec
