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
        assert speaker.index_number.iloc[0] == i
    for azi, ele in zip(main._table.azi, main._table.ele):
        speaker = main.get_speaker(coordinates=[azi, ele])
        assert speaker.azi.iloc[0] == azi
        assert speaker.ele.iloc[0] == ele
    # get lists of speakers:
    speaker_list = [4, 16, 32, 45]
    speakers = main.get_speaker_list(speaker_list)
    assert len(speakers) == len(speaker_list)
    speaker_list = [(-52.5, 25), (-35, -12.5), (0, -12.5)]
    speakers = main.get_speaker_list(speaker_list)
    assert len(speakers) == len(speaker_list)


def test_shift_setup():
    pass


def test_set_signal_and_speaker():
    pass


