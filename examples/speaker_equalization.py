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
import numpy
cd = setup._location.parent
rx8_path=os.path.join(cd,"examples", "play_buf.rcx")
rp2_path=os.path.join(cd,"examples", "rec_buf.rcx")
for file in [rx8_path, rp2_path]: # Check if the rcx files exist
    if not os.path.isfile(file):
        raise FileNotFoundError("Could not find file: "+file)
_speakers = None
_sound = None
_isinit = False

def init(setup, speakers="all", sound="sweep"):
    """
    To initialize the calbration process, select the setup (dome or arc) and
    the speakers for equalization. speakers can be "all" or a list of integers
    (speaker indices) if you only want to equalize a subset of speakers. The
    variable sound is the input that is send to each speaker.
    """
    setup.set_speaker_config(setup)
    setup.initialize_devices(RX81_file=rx8_path, RX82_file=rx8_path, RP2_file=rp2_path, ZBus=True)
    # get entrys from the speaker table. Each row contains channel, rx8,
    # azimuth and elevation of one speaker
    if speakers="all":
        _speakers=[setup.speaker_from_number(i) for i in range(1,48)]
    else:
        _speakers=[setup.speaker_from_number(i) for i in range(1,48)]


def play_and_record():
    """
    Play the sound from each speaker and record the ouput
    """
    if not _isinit:
        raise ValueError("procedure be initialized first!")
    else:
        for speker in speakers:

        setup.set_variable(variable="playbuflen",value=sound_in.nsamples,proc="RX8s")
        setup.set_variable(variable="data",value=sound_in.data ,proc="RX8s")
        setup.set_variable(variable="recbuflen",value=sound_in.nsamples+delay_samples, proc="RP2")
