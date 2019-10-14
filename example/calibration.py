import numpy as np
import sys
sys.path.append("C:/Projects/soundtools")
sys.path.append("C:/Projects/freefield_toolbox")
import slab
from freefield import setup as fs
import os
cd = setup._location.parent
rx8_path=os.path.join(cd, "play_buf.rcx")
rp2_path=os.path.join(cd, "rec_buf.rcx")

def record_transfer_function(sound="chirp",dur=0.1, samplerate=48828.125, speakers="all", setup="dome"):

    fs.set_speaker_config(setup)
    fs.initialize_devices(rx8_path, rx8_path, rp2_path, ZBus=True)

    if speakers=="all":
        speakers = list(range(1,49))
    if not isinstance(speakers, list) or isinstance(speakers, np.array):
        raise ValueError("'speakers' must be 'all' or a list/array!")
    else: # speakers is list, continue...
        if sound.lower()="chirp":
            sound_in = slab.Sound.chirp(duration=dur, samplerate=samplerate)
        elif sound.lower()="whitenoise":
            sound_in = slab.Sound.whitenoise(duration=dur, samplerate=samplerate)
        elif sound.lower()="pinknoise":
            sound_in = slab.Sound.pinknoise(duration=dur, samplerate=samplerate)
        else:
            raise ValueError("Unkown sound type!")

        fs.set_variable(variable="playbuflen",value=sound_in.nsamples,proc="RX8s")
        fs.set_variable(variable="data",value=sound_in.data ,proc="RX8s")
        fs.set_variable(variable="recbuflen",value=sound_in.nsamples+1000, proc="RP2")
        for i in speakers:
            channel, rx8, azimuth, elevation = fs.speaker_from_number(i)
            fs.set_variable(variable="chan",value=channel, proc="RX8"+str(rx8))
            fs.tigger()
            while fs.get_variable(variable="recording", proc="RP2"):
                time.sleep(0.01)
            sound_out = fs.get_variable(variable="data_ch1",nsamples=sound_in.nsamples+1000, proc="RP2")
