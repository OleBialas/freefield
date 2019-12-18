"""
play sound from loudspeaker and record. Then calculate the difference between
the played and recorded sound an compute an inverse filter to equalize
the difference
"""
from freefield import setup
import os
cd = setup._location.parent
rx8_path=os.path.join(cd,"examples", "play_buf.rcx")
rp2_path=os.path.join(cd,"examples", "rec_buf.rcx")
for file in [rx8_path, rp2_path]:
    if not os.path.isfile(file):
        raise FileNotFoundError("Could not find file: "+file)
setup.set_speaker_config(setup)
setup.initialize_devices(RX81_file=rx8_path, RX82_file=rx8_path, RP2_file=rp2_path, ZBus=True)

def play_and_record(speakers, sound):
    setup.set_variable(variable="playbuflen",value=sound_in.nsamples,proc="RX8s")
    setup.set_variable(variable="data",value=sound_in.data ,proc="RX8s")
    setup.set_variable(variable="recbuflen",value=sound_in.nsamples+delay_samples, proc="RP2")
