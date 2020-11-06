from freefield import Devices
import numpy as np

def test_devices():
    _devices = Devices()
    for mode in ["play_rec", "play_birec", "loctest_freefield",
                 "loctest_headphones", "cam_calibration"]:
        _devices.initialize_default(mode)
    _devices._procs
    data = _devices.read("tag", 1, proc='RX81')
    assert data == 1
    data = _devices.read("tag", 1000, proc='RX81')
    assert len(data) == 1000
    assert _devices.write("tag", 1, procs=['RX81', 'RX82']) == 1
    _devices.write(["tag1", "tag2", "tag3"], [1, 2, 3],
                   procs=[['RX81', 'RX82'], ['RP2'], ['RX81']])


tag = ["tag1", "tag2", "tag3"]
procs = [['RX81', 'RX82'], ['RP2'], ['RX81']]
value = [1, 2, 3]

if isinstance(tag, list):
    if not len(tag) == len(value) == len(procs):
        raise ValueError("tag, value and procs must be same length!")
    else:
        procnames = [item for sublist in procs for item in sublist]
else:
    tag, value = [tag], [value]
    if isinstance(procs, str):
        procnames = [procs]
    else:
        procnames = procs

# Check if the processors you want to write to are in _procs
if not set(procnames).issubset(_devices._procs.keys()):
    raise ValueError('Can not find some of the specified processors!')

for t, v, proc in zip(tag, value, procs):
    for p in proc:
        if isinstance(v, (list, np.ndarray)):
            flag = _devices._procs[p]._oleobj_.InvokeTypes(
                15, 0x0, 1, (3, 0), ((8, 0), (3, 0), (0x2005, 0)),
                t, 0, v)
        else:
            flag = _devices._procs[p].SetTagVal(t, v)
