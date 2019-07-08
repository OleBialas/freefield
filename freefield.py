'''
Functions and classes for working with the freefield dome and arc.
'''

import numpy as np
import slab
import win32com.client
import csv
import numpy as np
import os

# thoughts on the software architecture:
# Ideally, ZB, RX8, and RP2 are internal variables that the user never needs to use directly (but we can if needed). We would not need a freefield class for this, because there can never be more than one instance anyway.
# set_device function should be called in the beginning to choose arc or dome. Internal variables _calibration_filter and _speakertable are set accordingly. Other functions should not need to read the _device variable.
# provide functions for reading (or waiting for!) response from button box, flashlight, and headtracker

# internal variables here:
_procs = dict(RX81=None, RX82=None, RP2=None, ZBus=None) # dict might be better because you can call objects with a string
# Use like so: _procs.RX81, _procs.RX82, ...
_setup = None # ('arc' or 'dome')
_calibration_file = None
_calibration_filter = None
_speakertable = None


def set_setup(setup='arc'):
	'''
	Set the freefield setup to use (arc or dome).
	'''
	global _device, _calibration_file, _calibration_filter, _speakertable
	if setup == 'arc':
		_setup = 'arc'
		_calibration_file = 'calibration_filter_arc.npy'
		_speakertable = _read_table('speakertable_arc.txt')
	elif setup == 'dome':
		_setup = 'dome'
		_calibration_file = 'calibration_filter_dome.npy'
		_speakertable = _read_table('speakertable_dome.txt')
	else:
		raise ValueError("Unknown device! Use 'arc' or 'dome'.")
	_calibration_filter =  "load filter here"#slab.Filter.load(_calibration_file)

def initialize_devices(rcx_file_name_RX8_1=None, rcx_file_name_RX8_2=None, rcx_file_name_RP2=None, ZBus=False):
	'''
	Initialize the ZBus, RX8s, and RP2.
	'''
	global _procs
	if not _device:
		raise ValueError("Please set device to 'arc' or 'dome' before initialization!")

	if rcx_file_name_RX8_1:
		_procs["RX81"] = _initialize_processor("RX8", rcx_file_name_RX8_1, 1)
	if rcx_file_name_RX8_2:
		_procs["RX82"] = _initialize_processor("RX8", rcx_file_name_RX8_2, 2)
	if rcx_file_name_RX8_2:
		_procs["RP2"] = _initialize_processor("RP2", rcx_file_name_RP2, 1)
	if Zbus:
		_procs["ZBus"] = _initialize_zbus()

def set(variable, value, proc='RX8s'):
	'''
	Set a variable on a processor to a value. Setting will silently fail if
	variable does not exist in the rco file. The function will use SetTagVal
	or WriteTagV correctly, depending on whether len(value) == 1 or is > 1.
	proc can be 'RX81', 'RX82', 'RX8s', or 'RP2'. 'RX8s' sends the value to
	all RX8 processors.
	Example:
	set('stimdur', 90, proc='RX8s')
	'''
	if type(value) == list or type(value) == np.ndarray:
		_procs[proc]._oleobj_.InvokeTypes(15, 0x0, 1, (3, 0), ((8, 0), (3, 0), (0x2005, 0)), variable, 0, value) # = WriteTagV
	else:
		_procs[proc].SetTagVal(variable, value)
	pass

def get(variable=None,n_samples=1, proc='RX81'):
	'''
	Get the value of a variable from a processor. Returns None if variable
	does not exist in the rco file. [Can we get single items and arrays automatically?]
	proc can be 'RX81', 'RX82', or 'RP2'.
	Example:
	get('playing', proc='RX81')
	'''
	if n_samples >1:
		value = np.asarray(_procs[proc].ReadTagV(variable, 0, n_samples))
	else:
		value = _procs[proc].GetTagVal(variable)
	return value

def halt(proc="all"):
	'''
	Halt specified processor. If "all" (default), halt all processors
	'''
	if proc=="all":
		for p in _procs.values():
			p.Halt()
	else:
		_procs[proc].Halt()
	pass

def trigger(trig='zBusA', proc=None):
	'''
	Send a trigger. Options are "soft", "zBusA" and "zBusB". For using
	the software trigger a processor must be specified. For using
	zBus triggers, the zBus must be initialized first.
	'''
	if 'soft' in trig.lower():
		if not proc:
			raise ValueError('Proc needs to be specified for SoftTrig!')
		_procs[proc].SoftTrg()
	if "zbus" in trig.lower() and not _procs["ZBus"]:
		raise ValueError('ZBus needs to be initialized first!')
	if trig.lower()=="zbusa":
		_procs["ZBUS"].zBusTrigA(0, 0, 20)
	elif trig.lower()=="zbusb":
		_procs["ZBUS"].zBusTrigB(0, 0, 20)
	else:
		raise ValueError("Unknown trigger type! Must be 'soft', 'zBusA' or 'zBusB'!")

def wait_to_finish_playing():
	'''
	Wait for a signal from a buffer in RX81 to end playing.
	Relies on 'playing' tag attached to Schmidt trigger.
	'''
	while get(variable='playing'):
		# pause 0.01 secs
		pass

def speaker_from_direction(azimuth=0, elevation=0):
	'''
	Returns the speaker number corresponding to a given azimuth and elevation
	and the processor that speaker is attached to.
	'''
	table = filter_table(azimuth=[str(azimuth)], elevation=[str(elevation)])
	proc = table["proc"]
	speaker = table["index"]
	return speaker, proc

# other functions to access the freefield table here
def _read_table(fname):
	handle = open(fname, encoding="utf8")
	reader = csv.reader(handle)
	headers = reader.__next__()
	table = {}
	for h in headers:
		table[h] = []
	for row in reader:
		for h, v in zip(headers, row):
			table[h].append(v)
	return table

def filter_table(**kwargs):
	"""
	Read table and filter for keyword arguments. Only accepts lists of strings.
	"""
	table=_speakertable
	if len(kwargs)==0:
		raise ValueError("Need a keyword to filter table!")
	for title, values in kwargs.items():
		tmp = {}
		for key in _speakertable:
			tmp[key] = []
		for value in values:
			pos = np.where(np.asanyarray(table[title]) == value)[0]
			for j in pos:
				for key in table.keys():
					tmp[key].append(table[key][j])
		table = tmp
	return table

def set_signal_and_speaker(signal=None, speaker_number=0, apply_calibration=True):
	'''
	Upload a signal to the correct RX8 (signal on the other one stays the same)
	and channel. If apply_calibration=True, apply the speaker's inverse filter
	before upoading.
	'''
	if apply_calibration:
		signal = _calibration_filter.channel(speaker_number).apply(signal)
	set(variable='chan', value=speaker_number, proc='RX81')
	set(variable='chan', value=speaker_number-24, proc='RX82')
	# TODO: this assumes higher chan numbers are on RX82 - might not be true for dome?

# functions implementing complete procedures
def calibrate():
	'''
	Calibrate all speakers in the array by presenting a sound from each one,
	recording, computing inverse filters, and saving the calibration file.
	'''
	slab.Signal.set_default_samplerate(48828.125)
	sig = slab.Sound.chirp(duration=10000, from_freq=100, to_freq=None, kind='quadratic')
	initialize(rcx_file_name_RX8_1='calibration_RX8.rco', rcx_file_name_RX8_2='calibration_RX8.rco', rcx_file_name_RP2='calibration_RP2.rco')
	input('Set up microphone. Press any key to start calibration...')
	set(variable='signal', value=sig, proc='RX8s')
	recording = numpy.zeros((sig.nsamples,48))
	for speaker in range(48):
		set(variable='chan', value=speaker+1, proc='RX8s')
		for i in range(10):
			trigger() # zBusA by default
			wait_to_finish_playing()
			if i == 0: # first iteration
				rec = get(variable='recording', proc='RP2')
			else:
				rec = rec + get(variable='recording', proc='RP2')
		recording[:,speaker] = rec / 10 # averaging
	recording = slab.Sound(recording) # make a multi-channel sound objects
	#filt = # make inverse filter
	# rename old filter file, if it exists, by appending current date
	filt.save(_calibration_file) # save filter file to 'calibration_arc.npy' or dome.

def _initialize_processor(device_type, rcx_file, index, connection="GB"):

	try:
		RP = win32com.client.Dispatch('RPco.X')
	except win32com.client.pythoncom.com_error as e:
		print("Error:", e)
		return -1

	if device_type == "RP2":
		if RP.ConnectRP2(connection, index):
			print("Connected to RP2")
		elif device_type == "RX8":
			if RP.ConnectRX8(connection, index):
				print("Connected to RX8")
		else:
			print("Error: unknown device type!")
			return -1

	if not RP.ClearCOF():
		print("ClearCOF failed")
		return -1

	if RP.LoadCOF(rcx_file):
		print("Circuit {0} loaded".format(rcx_file))
	else:
		print("Failed to load {0}".format(rcx_file))
		return -1

	if RP.Run():
		print("Circuit running")
	else:
		print("Failed to run {0}".format(rcx_file))
		return -1

	return RP

def _initialize_zbus(connection="GB"):

	try:
		ZB = win32com.client.Dispatch('ZBUS.x')
	except win32com.client.pythoncom.com_error as e:
		print("Error:", e)
		return -1
	print("Successfully initialized ZBus")

	if ZB.ConnectZBUS(connection):
		print("Connected to ZBUS")
	else:
		print("failed to connect to ZBUS")

	return ZB

class camera():
	"""
	Class to handle the camera. The calibration is done upon initializing.
	Might have to add a calibration method. Acquisition is startet with the
	start() method and runs until stop() is called (bug: once the cam is stopped
	you have to restart the console to start it again). The camera capture along
	with the caluculated angles is constantly displayed. To get the angles describing
	the current headpose use get_pose(). To remove the jitter, the angles are
	calculatet a number of times (default=100) and the averaged.
	"""
	import cv2
	import dlib
	from imutils import face_utils
	from PIL import Image
	import threading

	def __init__(self):
		"do all the configuration stuff at initialization"
		face_landmark_path = 'C:/Projects/freefield_tools/headpose_estimation/shape_predictor_68_face_landmarks.dat'
		im = Image.open('C:/Projects/freefield_tools/headpose_estimation/calib.jpg')
		K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
			 0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
			 0.0, 0.0, 1.0]
		D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]
		self.cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
		self.dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)
		self.object_pts = np.float32([[6.825897, 6.760612, 4.402142],
								[1.330353, 7.122144, 6.903745],
								[-1.330353, 7.122144, 6.903745],
								[-6.825897, 6.760612, 4.402142],
								[5.311432, 5.485328, 3.987654],
								[1.789930, 5.393625, 4.413414],
								[-1.789930, 5.393625, 4.413414],
								[-5.311432, 5.485328, 3.987654],
								[2.005628, 1.409845, 6.165652],
								[-2.005628, 1.409845, 6.165652],
								[2.774015, -2.080775, 5.048531],
								[-2.774015, -2.080775, 5.048531],
								[0.000000, -3.116408, 6.097667],
								[0.000000, -7.415691, 4.070434]])
		self.reprojectsrc = np.float32([[10.0, 10.0, 10.0],
								[10.0, 10.0, -10.0],
								[10.0, -10.0, -10.0],
								[10.0, -10.0, 10.0],
								[-10.0, 10.0, 10.0],
								[-10.0, 10.0, -10.0],
								[-10.0, -10.0, -10.0],
								[-10.0, -10.0, 10.0]])
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor(face_landmark_path)

	def start(self):
		t = threading.Thread(target=self._start)
		t.start()

	def stop(self):
		self.cap.release()

	def get_pose(self, n=100):
		"""
		compute angles n times and average to remove the jitter
		"""
		x, y, z = 0, 0, 0
		count=n
		while count>0:
			_, angles= _get_angles()
			x+= angles[0, 0]
			y+= angles[1, 0]
			z+= angles[2, 0]
			count-=1
		return x/n, y/n, z/n


	def _start(self):
		self.cap = cv2.VideoCapture(0)
		if not self.cap.isOpened():
			print("Unable to connect to camera.")
			return
		while self.cap.isOpened():
			ret, frame = self.cap.read()
			if ret:
				face_rects = self.detector(frame, 0)
				if len(face_rects) > 0:
						self.shape = self.predictor(frame, face_rects[0])
						self.shape = face_utils.shape_to_np(self.shape)

				reprojectdst, euler_angle = self._get_angles()
				for (x, y) in self.shape:
					cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
				cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
					0.75, (0, 0, 0), thickness=2)
				cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
					0.75, (0, 0, 0), thickness=2)
				cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
					0.75, (0, 0, 0), thickness=2)
				cv2.imshow("Head Pose", frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

	def _get_angles(self):
		"This function calculates the angles from the camera image"
		image_pts = np.float32([self.shape[17], self.shape[21], self.shape[22], self.shape[26], self.shape[36],
								self.shape[39], self.shape[42], self.shape[45], self.shape[31], self.shape[35],
								self.shape[48], self.shape[54], self.shape[57], self.shape[8]])
		_, rotation_vec, translation_vec = cv2.solvePnP(self.object_pts, image_pts, self.cam_matrix, self.dist_coeffs)

		reprojectdst, _ = cv2.projectPoints(self.reprojectsrc, rotation_vec, translation_vec, self.cam_matrix,
											self.dist_coeffs)
		reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

		# calc euler angle
		rotation_mat, _ = cv2.Rodrigues(rotation_vec)
		pose_mat = cv2.hconcat((rotation_mat, translation_vec))
		_, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

		return reprojectdst, euler_angle
