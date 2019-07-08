import cv2
import dlib
from imutils import face_utils
from PIL import Image
import threading

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
