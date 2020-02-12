import time
from scipy import signal
import os
from freefield import setup as fs
from slab.filter import Filter
import slab
import numpy as np
import sys
sys.path.append("C:/Projects/soundtools")
sys.path.append("C:/Projects/freefield_toolbox")
cd = fs._location.parent


def record_transfer_function(sound="chirp", dur=0.04, n_reps=10, samplerate=48828.125, speakers="all", setup="dome"):

    rx8_path = os.path.join(cd, "examples", "play_buf.rcx")
    rp2_path = os.path.join(cd, "examples", "rec_buf.rcx")
    fs.set_speaker_config(setup)
    fs.initialize_devices(RX81_file=rx8_path, RX82_file=rx8_path, RP2_file=rp2_path, ZBus=True)
    out_path = os.path.join(cd, "transfer_functions", setup)
    if speakers == "all":
        speakers = list(range(1, 48))
    if not isinstance(speakers, (list, np.ndarray)):
        raise ValueError("'speakers' must be 'all' or a list/array!")
    else:  # speakers is list, continue...
        sound_in = fs.makesound(duration=dur, kind=sound, samplerate=samplerate)
        sound_in.ramp(duration=dur/10, when="both")
        delay_samples = fs.get_recording_delay(da_delay="RX8", ad_delay="RP2")
        fs.set_variable(variable="playbuflen", value=sound_in.nsamples, proc="RX8s")
        fs.set_variable(variable="data", value=sound_in.data, proc="RX8s")
        fs.set_variable(variable="recbuflen", value=sound_in.nsamples+delay_samples, proc="RP2")
        for i in speakers:
            fname = "speaker_%s_%sms_%s_tf.wav" % (i, int(dur*1000), sound)
            sound_out = np.zeros((n_reps, sound_in.nsamples))
            channel, rx8, azimuth, elevation = fs.speaker_from_number(i)
            fs.set_variable(variable="chan", value=channel, proc="RX8"+str(rx8))
            time.sleep(0.5)
            for n in range(n_reps):
                fs.trigger()
                while fs.get_variable(variable="recording", proc="RP2"):
                    time.sleep(0.01)
                time.sleep(0.5)
                rec = fs.get_variable(
                    variable="data", n_samples=sound_in.nsamples+delay_samples, proc="RP2")
                sound_out[n] = rec[delay_samples::]
            sound_out = slab.Sound(data=sound_out, samplerate=samplerate)
            sound_out.write(os.path.join(out_path, fname), normalise=True)
    sound_in.write(os.path.join(out_path, "%sms_%s.wav" % (int(dur*1000), sound)), normalise=True)
    fs.halt()


def record_hp_tf(sound="chirp", dur=0.04, n_reps=10, samplerate=48828.125):
    rp2_path = os.path.join(cd, "examples", "play_n_rec_headphones.rcx")
    fs.set_speaker_config("dome")
    fs.initialize_devices(RP2_file=rp2_path, ZBus=True)
    out_path = os.path.join(cd, "transfer_functions", "headphones")
    sound_in = fs.makesound(duration=dur, kind=sound, samplerate=samplerate, level=90)
    delay_samples = fs.get_recording_delay(distance=0, da_delay="RP2", ad_delay="RP2")
    fs.set_variable(variable="data_in_left", value=sound_in.data, proc="RP2")
    fs.set_variable(variable="data_in_right", value=sound_in.data, proc="RP2")
    fs.set_variable(variable="buflen", value=sound_in.nsamples+delay_samples, proc="RP2")
    fs.trigger()
    left = fs.get_variable(variable="data_out_left",
                           n_samples=sound_in.nsamples+delay_samples, proc="RP2")
    right = fs.get_variable(variable="data_out_right",
                            n_samples=sound_in.nsamples+delay_samples, proc="RP2")
    sound_out = slab.Sound(data=np.array([left, right]).T, samplerate=samplerate)
    sound_in.write(os.path.join(out_path, "%sms_%s.wav" % (int(dur*1000), sound)), normalise=True)
    sound_out.write(os.path.join(out_path, "k1000_%sms_%s.wav" %
                                 (int(dur*1000), sound)), normalise=True)


def make_inverse_filters(speakers="all", setup="arc"):
    path = "C:\\Projects\\freefield_toolbox\\transfer_functions\\arc\\"
    played_signal = slab.Sound(path+"40ms_chirp.wav")
    for i in range(47):
        recorded_signals = slab.Sound(path+"speaker_1_40ms_chirp_tf.wav")
        filterbank = Filter.equalizing_filterbank(played_signal, recorded_signals)
