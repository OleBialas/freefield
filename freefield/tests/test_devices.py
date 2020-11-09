from freefield import Devices


def test_devices():
    _devices = Devices()
    for mode in ["play_rec", "play_birec", "loctest_freefield",
                 "loctest_headphones", "cam_calibration"]:
        _devices.initialize_default(mode)
    data = _devices.read("tag", 1, proc='RX81')
    assert data == 1
    data = _devices.read("tag", 1000, proc='RX81')
    assert len(data) == 1000
    assert _devices.write("tag", 1, procs=['RX81', 'RX82']) == 1
    _devices.write(["tag1", "tag2", "tag3"], [1, 2, 3],
                   procs=[['RX81', 'RX82'], ['RP2'], ['RX81']])
