from freefield import main
import multiprocessing
import time
main.initialize_setup(setup="dome", default_mode="play_rec")


def test_wait():
    p = multiprocessing.Process(target=main.wait_to_finish_playing,
                                name="wait", args=("RX81",))
    p.start()
    time.sleep(5)
    p.terminate()
    p.join()


def test_get_speaker():
    # get single speakers
    for i in range(47):
        speaker = main.get_speaker(index_number=i, coordinates=None)
        assert speaker.index_number == i
    for azi, ele in zip(main._table.azi, main._table.ele):
        speaker = main.get_speaker(coordinates=[azi, ele])
        assert speaker.azi == azi
        assert speaker.ele == ele


def test_shift_setup():
    pass


def test_set_signal_and_speaker():
    pass


