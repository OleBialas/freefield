import time
from psychopy import logging
from psychopy.hardware import brainproducts

logging.console.setLevel(logging.DEBUG)
rcs = brainproducts.RemoteControlServer()
rcs.open('testExp',
         workspace='filepath',
         participant='S0021')
rcs.openRecorder()
time.sleep(2)
rcs.mode = 'monitor' # or 'impedance', or 'default'
rcs.startRecording()
time.sleep(2)
rcs.sendAnnotation('124', 'STIM')
time.sleep(1)
rcs.pauseRecording()
time.sleep(1)
rcs.resumeRecording()
time.sleep(1)
rcs.stopRecording()
time.sleep(1)
rcs.mode = 'default'  # stops monitoring mode