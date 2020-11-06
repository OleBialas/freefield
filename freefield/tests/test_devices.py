from freefield import Devices


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

tag = "tag1"
procs = ['RX81', 'RX82']
value = 1
if isinstance(tag, list):
    if not len(tag) == len(value) == len(procs):
        raise ValueError("tag, value and procs must be same length!")
    else:
        procnames = [proc for proc in (sublist for sublist in procs)]
else:
    tag, value = [tag], [value]
    if isinstance(procs, str):
        procnames = [procs]
    else:
        procnames = procs

print(names)

# Check if the processors you want to write to are in _procs
if not set(names).issubset(_devices._procs.keys()):
    raise ValueError('Can not find some of the specified processors!')
