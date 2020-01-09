"""
play sound from loudspeaker and record. Then calculate the difference between
the played and recorded sound an compute an inverse filter to equalize
the difference
"""
try:
    from freefield import setup
except:
    raise ImportError("could not find the freefield toolbox module!")
try:
    import slab
except:
    raise ImportError("could not find the slab toolbox module!")
import os
import time
import numpy as np
cd = setup._location.parent
rx8_path = os.path.join(cd,"examples", "play_buf.rcx")
rp2_path = os.path.join(cd,"examples", "rec_buf.rcx")
for file in [rx8_path, rp2_path]: # Check if the rcx files exist
    if not os.path.isfile(file):
        raise FileNotFoundError("Could not find file: "+file)
_speakers = None
_recordings = None
_sound = None
_samplerate=48828

def init(setup="dome", speakers="all", sound="sweep", dur=0.04):
    """
    To initialize the calbration process, select the setup (dome or arc) and
    the speakers for equalization. speakers can be "all" or a list of integers
    (speaker indices) if you only want to equalize a subset of speakers. The
    variable sound is the input that is send to each speaker.
    """
    global _speakers, _recordings, _sound
    setup.set_speaker_config(setup)
    setup.initialize_devices(RX81_file=rx8_path, RX82_file=rx8_path, RP2_file=rp2_path, ZBus=True)
    # get entries from the speaker table. Each row contains channel, rx8,
    # azimuth and elevation of one speaker
    if speakers=="all":
        _speakers=[setup.speaker_from_number(i) for i in range(1,48)]
    elif isinstance(speakers, list):
        _speakers=[setup.speaker_from_number(i) for i in speakers]
    else:
        raise ValueError("Data type for argument 'speakers' not understood!")
    # generate the sound and load it into the buffer of the device
    if sound=="sweep":
        _sound=slab.Sound.chirp(dur, samplerate=_samplerate)
    elif sound=="noise":
        _sound=slab.Sound.whitenoise(dur, samplerate=_samplerate)
    _recordings = np.zeros([sound.nsamples+1000, len(_speakers)])
    setup.set_variable(variable="playbuflen", value=_sound.nsamples, proc="RX8s")
    # Buffer for recording should be longer to compensate for sound traveling delay
    setup.set_variable(variable="playbuflen", value=_sound.nsamples+1000, proc="RP2")
    setup.set_variable(variable="data", value=_sound.data, proc="RX8s")

def play_and_record():
    """
    Play the sound from each speaker and record the ouput
    """
    global _recordings
    if _speakers==None:
        raise ValueError("procedure be initialized first!")
    else: # set the channel to the speaker, set the channel of the other RX8 to non-existent
        for speaker,i in enumerate(_speakers):
            setup.set_variable(variable="chan", value=speaker[0], proc=speaker[1])
            setup.set_variable(variable="chan", value=25, proc=3-speaker[1])
            setup.trigger()
            while setup.get_variable(variable="playback",proc="RP2"):
                time.sleep(0.01)
            _recordings[i] = setup.get_variable(variable="data", n_samples=_sound.nsamples+1000, proc="RP2")
        _recordings = slab.Sound(_recordings, samplerate=_samplerate)

def make_inverse_filter():

    filterbank = slab.Filter.equalizing_filterbank(_sound, _recordings)
