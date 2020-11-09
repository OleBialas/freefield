from freefield import main
import multiprocessing
import time

main.initialize_setup(setup="dome", default_mode="play_rec")


def test_wait():
    p = multiprocessing.Process(target=main.wait_to_finish_playing,
                                name="wait", args=("RX81",))
    p.start()  # run as process with timeout
    time.sleep(5)
    p.terminate()
    p.join()


def test_get_speaker():
    for i in range(47):
        speaker = main.get_speaker(index=i, coordinates=None)
        assert len(speaker) == 7
        assert speaker[0] == i
    for azi, ele in zip(main._table[:, 3], main._table[:,4]):
        speaker = main.get_speaker(coordinates=[azi, ele])
        assert len(speaker) == 7
        assert speaker[3] == azi and speaker[4] == ele

def test_shift_setup():
    pass

def test_set_signal_and_speaker():
    pass


