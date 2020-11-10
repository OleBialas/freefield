import time
from pathlib import Path
import glob
from copy import deepcopy
import numpy as np
import slab
from typing import Union
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from freefield import camera
import pandas as pd
import datetime
from freefield import DIR, Devices
import logging

logging.basicConfig(level=logging.WARNING)
# default samplerate for generating sounds, filters etc.
slab.Signal.set_default_samplerate(48828)

_config = None  # either "dome" or "arc"
_devices = Devices()
_calibration_freqs = None  # filters for frequency equalization
_calibration_lvls = None  # calibration to equalize levels
_table = pd.DataFrame()  # numbers and coordinates of all loudspeakers
_rec_tresh = 65  # treshold in dB above which recordings are not rejected
_fix_ele = 0  # fixation points' elevation
_fix_azi = 0  # fixation points' azimuth
_fix_acc = 10  # accuracy for determining if subject looks at fixation point


def initialize_setup(setup: str, default_mode: Union[str, bool] = None, device_list: Union[list, bool] = None,
                     zbus: bool = True, connection: str = "GB") -> None:
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
    """

    # TODO: put level and frequency equalization in one common file
    global _config, _calibration_freqs, _calibration_lvls, _table, _devices
    # initialize devices
    if bool(device_list) == bool(default_mode):
        raise ValueError("You have to specify a device_list OR a default_mode")
    if device_list is not None:
        _devices.initialize_devices(device_list, zbus, connection)
    elif default_mode is not None:
        _devices.initialize_default(default_mode)
    # get the correct speaker table and calibration files for the setup
    if setup == 'arc':
        _config = 'arc'
        freq_calibration_file = DIR / 'data' / Path('frequency_calibration_arc.npy')
        lvl_calibration_file = DIR / 'data' / Path('level_calibration_arc.npy')
        table_file = DIR / 'data' / 'tables' / Path('speakertable_arc.txt')
    elif setup == 'dome':
        _config = 'dome'
        freq_calibration_file = DIR / 'data' / Path('frequency_calibration_dome.npy')
        lvl_calibration_file = DIR / 'data' / Path('level_calibration_dome.npy')
        table_file = DIR / 'data' / 'tables' / Path('speakertable_dome.txt')
    else:
        raise ValueError("Unknown device! Use 'arc' or 'dome'.")
    logging.info(f'Speaker configuration set to {setup}.')
    # lambdas provide default values of 0 if azi or ele are not in the file
    _table = pd.read_csv(table_file)
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
        proc = list(_devices.procs.keys())
    elif isinstance(proc, str):
        proc = [proc]
    logging.info(f'Waiting for {tag} on {proc}.')
    while any(_devices.read(tag, n_samples=1, proc=p) for p in proc):
        time.sleep(0.01)
    logging.info('Done waiting.')


def get_speaker(index_number: Union[int, bool] = None, coordinates: Union[list, bool] = None) -> pd.DataFrame:
    """
    Either return the speaker at given coordinates (azimuth, elevation) or the
    speaker with a specific index number.

    Args:
        index (int): index number of the speaker
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
    if index_number:
        row = _table.loc[(_table.index_number == index_number)]
    elif coordinates:
        if len(coordinates) != 2:
            raise ValueError("Coordinates must have two elements: azimuth and elevation!")
        row = _table.loc[(_table.azi == coordinates[0]) & (_table.ele == coordinates[1])]
    if len(row) > 1:
        logging.warning("More or then one row returned!")
    if len(row) == 0:
        logging.warning("No entry found that matches the criterion!")
    return row


def get_speaker_list(speakers: list) -> pd.DataFrame:
    """
    Specify a list of either indices or coordinates and call get_speaker()
    for each element of the list.

    Args:
        speakers (list or list of lists): indices or coordinates of speakers.

    Returns:
        list of lists: rows from _table corresponding to the list.
            each sub list contains all the variable returned by get_speaker()
    """
    speaker_list = pd.DataFrame()
    if (all(isinstance(x, int) for x in speakers) or  # list contains indices
            all(isinstance(x, np.int64) for x in speakers)):
        speaker_list = [get_speaker(index_number=i) for i in speakers]
        speaker_list = [df.set_index('index') for df in speaker_list]
        speaker_list = pd.concat(speaker_list)
    elif (all(isinstance(x, tuple) for x in speakers) or  # list contains coords
          all(isinstance(x, list) for x in speakers)):
        speaker_list = [get_speaker(coordinates=i) for i in speakers]
        speaker_list = [df.set_index('index') for df in speaker_list]
        speaker_list = pd.concat(speaker_list)
    if len(speaker_list) == 0:
        logging.warning("No speakers found that match the criteria!")
    return speaker_list


def all_leds() -> pd.DataFrame:
    # Temporary hack: return all speakers from the table which have a LED attached
    return _table.dropna()


def shift_setup(delta: tuple) -> None:
    """
    Shift the setup (relative to the lister) by adding some delta value
    in azimuth and elevation. This can be used when chaning the position of
    the chair where the listener is sitting - moving the chair to the right
    is equivalent to shifting the setup to the left. Changes are not saved to
    the speaker table.

    Args:
        delta (tuple of floats): azimuth and elevation by which the setup is
        shifted. Positive values mean shifting right/up, negative values
        mean shifting left/down
    """
    global _table
    _table[:, 3] += delta[0]  # azimuth
    _table[:, 4] += delta[1]  # elevation
    logging.info("shifting the loudspeaker array by % s degree in azimuth / n"
                 "and % s degree in elevation" % (delta[0], delta[1]))


def set_signal_and_speaker(signal=None, speaker=0, apply_calibration=False):
    '''
        Upload a signal to the correct RX8 and channel (channel on the other
        RX8 set to -1). If apply_calibration=True, apply the speaker's inverse
        filter before upoading. 'speaker' can be a speaker number (1-48) or
        a tuple (azimuth, elevation).
        '''
    if isinstance(speaker, tuple):
        speaker, channel, chidx, azimuth, elevation, bitval, bitidx = \
            get_speaker(coordinates=(speaker[0], speaker[1]))
    elif isinstance(speaker, int):
        speaker, channel, chidx, azimuth, elevation, bitval, bitidx = \
            get_speaker(index=speaker)
    toplay = deepcopy(signal)  # copy the signal so the original is not changed
    if apply_calibration:
        if _calibration_freqs is None:
            logging.warning("Setup is not calibrated!")
        else:
            logging.info('Applying calibration.')
        toplay.level *= _calibration_lvls[int(speaker)]
        toplay = _calibration_freqs.channel(
            int(speaker)).apply(toplay)
    _devices.write(['chan', 'data'], [channel, toplay.data], [proc, proc])
    # set the other channel to non existant
    set_variable(variable='chan', value=25, proc=3 - proc)


def get_recording_delay(distance=1.6, samplerate=48828.125, play_device=None,
                        rec_device=None):
    """
        Calculate the delay it takes for played sound to be recorded. Depends
        on the distance of the microphone from the speaker and on the devices
        digital-to-analog and analog-to-digital conversion delays.
        """
    n_sound_traveling = int(distance / 343 * samplerate)
    if play_device:
        if play_device == "RX8":
            n_da = 24
        elif play_device == "RP2":
            n_da = 30
        else:
            raise ValueError("Input %s not understood!" % (play_device))
    else:
        n_da = 0
    if rec_device:
        if rec_device == "RX8":
            n_ad = 47
        elif rec_device == "RP2":
            n_ad = 65
        else:
            raise ValueError("Input %s not understood!" % (rec_device))
    else:
        n_ad = 0
    return n_sound_traveling + n_da + n_ad


def check_pose(target=None, var=10):
    """ Take an image and check if head is pointed to the target position +/- var.
    Returns True or False. NaN values are ignored."""
    if target is None:
        target = (_fix_azi, _fix_ele)
    correct = True
    ele, azi = camera.get_headpose(convert=True, average=True, n_images=1)
    if azi is np.nan:
        pass
    else:
        if np.abs(ele - target[0]) > var:
            correct = False
    if ele is np.nan:
        pass
    else:
        if np.abs(ele - target[1]) > var:
            correct = False
    return correct

    # functions implementing complete procedures:
    def calibrate_camera(self, n_reps=1):
        # azimuth and elevation of a set of points in camera and world coords
        # one list for each camera
        coords = pd.DataFrame(columns=["ele_cam", "azi_cam", "ele_world",
                                       "azi_world", "cam", "frame", "n"])
        if _cam_type == "web" and targets is None:
            raise ValueError("Define target positions for calibrating webcam!")
        elif _cam_type == "freefield":
            targets = setup.all_leds()  # get the speakers that have a LED attached
            if setup._mode != "camera_calibration":
                setup.initialize_devices(mode="camera_calibration")
        elif _cam_type is None:
            raise ValueError("Initialize Camera before calibration!")
        if not setup._mode == "camera_calibration":  # initialize setup
            setup.initialize_devices(mode="camera_calibration")
        seq = Trialsequence(n_reps=n_reps, conditions=targets)
        while seq.n_remaining:
            target = seq.__next__()
            if _cam_type == "web":  # look at target position and press enter
                ele, azi = target[0], target[1]
                input("point your head towards the target at elevation: %s and "
                      "azimuth %s. \n Then press enter to take an image an get "
                      "the headpose" % (ele, azi))
            elif _cam_type == "freefield":  # light LED and wait for button press
                ele, azi = target[4], target[3]
                proc, bitval = target[6], target[5]
                setup.printv("trial nr %s: speaker at ele: %s and azi: of %s" %
                             (seq.this_n, ele, azi))
                setup.set_variable(variable="bitmask", value=bitval, proc=proc)
                while not setup.get_variable(variable="response", proc="RP2",
                                             supress_print=True):
                    time.sleep(0.1)  # wait untill button is pressed
            pose = get_headpose(average=False, convert=False, cams=cams)
            pose = pose.rename(columns={"ele": "ele_cam", "azi": "azi_cam"})
            pose.insert(0, "n", seq.this_n)
            pose.insert(2, "ele_world", ele)
            pose.insert(4, "azi_world", azi)
            pose = pose.dropna()
            coords = coords.append(pose, ignore_index=True, sort=True)
        if _cam_type == "freefield":
            setup.set_variable(variable="bitmask", value=0, proc="RX8s")


def localization_test_freefield(duration=0.5, n_reps=1, speakers=None, visual=False):
    """
    Run a basic localization test where the same sound is played from different
    speakers in randomized order, without playing the same position twice in
    a row. After every trial the presentation is paused and the listener has
    to localize the sound source by pointing the head towards the source and
    pressing the response button. The cameras need to be calibrated before the
    test! After every trial the listener has to point to the middle speaker at
    0 elevation and azimuth and press the button to iniciate the next trial.
    """
    # TODO: one function for fundamental trial-unit?
    if not _mode == "localization_test_freefield":
        initialize_devices(mode="localization_test_freefield")
    if camera._cal is None:
        raise ValueError("Camera must be calibrated before localization test!")
    warning = slab.Sound.clicktrain(duration=0.4).data.flatten()
    if visual is True:
        speakers = all_leds()
    else:
        speakers = speakers_from_list(speakers)  # should return same format
    seq = slab.Trialsequence(speakers, n_reps, kind="non_repeating")
    response = pd.DataFrame(columns=["ele_target", "azi_target", "ele_response", "azi_response"])
    start = slab.Sound.read("localization_test_start.wav").channel(0)
    # could be one function:
    set_signal_and_speaker(signal=start, speaker=23, apply_calibration=False)
    trigger()
    wait_to_finish_playing()
    while not get_variable(variable="response", proc="RP2"):
        time.sleep(0.01)
    while seq.n_remaining > 0:
        sound = slab.Sound.pinknoise(duration=duration)
        _, ch, proc_ch, azi, ele, bit, proc_bit = seq.__next__()
        # TODO response into trial sequence
        trial = {"azi_target": azi, "ele_target": ele}
        # give dictionary or list of variables ?
        set_variable(variable="chan", value=ch, proc="RX8%s" % int(proc_ch))
        set_variable(variable="chan", value=25, proc="RX8%s" % int(3 - proc_ch))
        set_variable(variable="playbuflen", value=len(sound), proc="RX8s")
        set_variable(variable="data", value=sound.data, proc="RX8s")
        if visual is True:
            set_variable(variable="bitmask", value=bit, proc=proc_bit)
        trigger()
        while not get_variable(variable="response", proc="RP2"):
            time.sleep(0.01)
        ele, azi = camera.get_headpose(convert=True, average=True, target=(azi, ele))
        if visual is True:
            set_variable(variable="bitmask", value=0, proc=proc_bit)
        trial["azi_response"], trial["ele_response"] = azi, ele
        response = response.append(trial, ignore_index=True)
        head_in_position = 0
        while head_in_position == 0:
            while not get_variable(variable="response", proc="RP2"):
                time.sleep(0.01)
            ele, azi = camera.get_headpose(convert=True, average=True, n_images=1)
            if ele is np.nan:
                ele = 0
            if azi is np.nan:
                azi = 0
            if np.abs(ele - _fix_ele) < _fix_acc and np.abs(azi - _fix_azi) < _fix_acc:
                head_in_position = 1
            else:
                print(np.abs(ele - _fix_ele), np.abs(azi - _fix_azi))
                set_variable(variable="data", value=warning, proc="RX8s")
                set_variable(variable="chan", value=1, proc="RX81")
                set_variable(variable="chan", value=25, proc="RX82")
                set_variable(variable="playbuflen", value=len(warning), proc="RX8s")
                trigger()
    # play sound to signal end
    set_signal_and_speaker(signal=start, speaker=23, apply_calibration=False)
    trigger()
    return response


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
