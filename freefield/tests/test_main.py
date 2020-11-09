from freefield import main
import multiprocessing
import time

main.initialize_setup(setup="dome", default_mode="play_rec")

def test_wait():
    p = multiprocessing.Process(target=main._devices.wait_to_finish_playing,
                                name="wait", args=())
    p.start()  # run as process with timeout
    time.sleep(5)
    p.terminate()
    p.join()
    pass
