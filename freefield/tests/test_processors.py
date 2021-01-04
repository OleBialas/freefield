from freefield import Processors


def test_devices():
    processors = Processors()
    for mode in ["play_rec", "play_birec", "loctest_freefield",
                 "loctest_headphones", "cam_calibration"]:
        processors.initialize_default(mode)
    data = processors.read("tag", 1, proc='RX81')
    assert data == 1
    data = processors.read("tag", 1000, proc='RX81')
    assert len(data) == 1000
    assert processors.write("tag", 1, procs=['RX81', 'RX82']) == 1
    processors.write(["tag1", "tag2", "tag3"], [1, 2, 3],
                   procs=[['RX81', 'RX82'], ['RP2'], ['RX81']])
