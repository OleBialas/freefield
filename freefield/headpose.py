"""
Tools for extracting the headpose from images using deep neural networks
Code is largely taken from this blogpost:
https://towardsdatascience.com/real-time-head-pose-estimation-in-python-e52db1bc606a
the pretrained models are taken from this github repo:
https://github.com/vardanagarwal/Proctoring-AI
"""
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import math
from freefield import DATADIR
import logging
from matplotlib import pyplot as plt

# 3D-model points to which the points extracted from an image are matched:
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye corner
                            (225.0, 170.0, -135.0),      # Right eye corner
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])


class PoseEstimator:
    def __init__(self):
        try:
            self.face_net = cv2.dnn.readNetFromCaffe(
                str(DATADIR/"models"/"prototxt"),
                str(DATADIR/"models"/"caffemodel"))
        except cv2.error:
            logging.warning("could not initialize DNN!")
        try:  # Restore model from the saved_model file.
            self.model = keras.models.load_model(DATADIR/"models"/"pose_model")
        except OSError:
            logging.warning("could not find the trained headpose model...")

        self.detection_result = None
        self.cnn_input_size = 128
        self.marks = None

    def get_faceboxes(self, image, threshold=0.5):
        """
        Get the bounding box of faces in image using dnn.
        """
        rows, cols, _ = image.shape
        confidences, faceboxes = [], []
        self.face_net.setInput(cv2.dnn.blobFromImage(
            image, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False))
        detections = self.face_net.forward()
        for result in detections[0, 0, :, :]:
            confidence = result[2]
            if confidence > threshold:
                x_left_bottom = int(result[3] * cols)
                y_left_bottom = int(result[4] * rows)
                x_right_top = int(result[5] * cols)
                y_right_top = int(result[6] * rows)
                confidences.append(confidence)
                faceboxes.append(
                    [x_left_bottom, y_left_bottom, x_right_top, y_right_top])
        self.detection_result = [faceboxes, confidences]
        return confidences, faceboxes

    def draw_all_result(self, image):
        """Draw the detection result on image"""
        for facebox, conf in self.detection_result:
            cv2.rectangle(image, (facebox[0], facebox[1]),
                          (facebox[2], facebox[3]), (0, 255, 0))
            label = "face: %.4f" % conf
            label_size, base_line = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(image, (facebox[0], facebox[1] - label_size[1]),
                          (facebox[0] + label_size[0],
                           facebox[1] + base_line),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(image, label, (facebox[0], facebox[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    @staticmethod
    def draw_box(image, boxes, box_color=(255, 255, 255)):
        """Draw square boxes on image"""
        for box in boxes:
            cv2.rectangle(image,
                          (box[0], box[1]),
                          (box[2], box[3]), box_color, 3)

    @staticmethod
    def move_box(box, offset):
        """Move the box to direction specified by vector offset"""
        left_x = box[0] + offset[0]
        top_y = box[1] + offset[1]
        right_x = box[2] + offset[0]
        bottom_y = box[3] + offset[1]
        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def get_square_box(box):
        """Get a square box out of the given box, by expanding it."""
        left_x = box[0]
        top_y = box[1]
        right_x = box[2]
        bottom_y = box[3]
        box_width = right_x - left_x
        box_height = bottom_y - top_y
        # Check if box is already a square. If not, make it a square.
        diff = box_height - box_width
        delta = int(abs(diff) / 2)
        if diff == 0:  # Already a square.
            return box
        elif diff > 0:  # Height > width, a slim box.
            left_x -= delta
            right_x += delta
            if diff % 2 == 1:
                right_x += 1
        else:  # Width > height, a short box.
            top_y -= delta
            bottom_y += delta
            if diff % 2 == 1:
                bottom_y += 1

        # Make sure box is always square.
        assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def box_in_image(box, image):
        """Check if the box is in image"""
        r = image.shape[0]  # rows
        c = image.shape[1]  # columns
        return box[0] >= 0 and box[1] >= 0 and box[2] <= c and box[3] <= r

    def extract_cnn_facebox(self, image):
        """Extract face area from image."""
        _, raw_boxes = self.face_detector.get_faceboxes(
            image=image, threshold=0.9)

        a = []
        for box in raw_boxes:
            # Move box down.
            # diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
            offset_y = int(abs((box[3] - box[1]) * 0.1))
            box_moved = self.move_box(box, [0, offset_y])

            # Make box square.
            facebox = self.get_square_box(box_moved)

            if self.box_in_image(facebox, image):
                a.append(facebox)

        return a

    def detect_marks(self, image_np):
        """Detect marks from image"""

        # # Actual detection.
        predictions = self.model.signatures["predict"](
            tf.constant(image_np, dtype=tf.uint8))

        # Convert predictions to landmarks.
        marks = np.array(predictions['output']).flatten()[:136]
        marks = np.reshape(marks, (-1, 2))

        return marks

    @staticmethod
    def draw_marks(image, marks, color=(255, 255, 255)):
        """Draw mark points on image"""
        for mark in marks:
            cv2.circle(image, (int(mark[0]), int(
                mark[1])), 1, color, -1, cv2.LINE_AA)


def draw_annotation_box(img, rotation_vector, translation_vector,
                        camera_matrix, color=(255, 255, 0), line_width=2):
    """Draw a 3D box as annotation of pose"""
    point_3d = []
    dist_coeffs = np.zeros((4, 1))
    rear_size = 1
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = img.shape[1]
    front_depth = front_size*2
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    k = (point_2d[5] + point_2d[8])//2
    return(point_2d[2], k)


# single image pose estimation:
mark_detector = MarkDetector()
cap = cv2.VideoCapture(0)
cap.isOpened()
cap.grab()
img = cap.retrieve()[1]
plt.imshow(img, cmap="gray")
cap.release()

size = img.shape
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array([[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype="double")

faceboxes = mark_detector.extract_cnn_facebox(img)

for facebox in faceboxes:
    face_img = img[facebox[1]: facebox[3], facebox[0]: facebox[2]]
    face_img = cv2.resize(face_img, (128, 128))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    marks = mark_detector.detect_marks([face_img])
    marks *= (facebox[2] - facebox[0])
    marks[:, 0] += facebox[0]
    marks[:, 1] += facebox[1]
    shape = marks.astype(np.uint)
    # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
    image_points = np.array([
                            shape[30],     # Nose tip
                            shape[8],     # Chin
                            shape[36],     # Left eye left corner
                            shape[45],     # Right eye right corne
                            shape[48],     # Left Mouth corner
                            shape[54]      # Right mouth corner
                            ], dtype="double")
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vec, translation_vec) = \
        cv2.solvePnP(model_points, image_points, camera_matrix,
                     dist_coeffs, flags=cv2.SOLVEPNP_UPNP)


# this is the "old" way of getting the pose:
rotation_mat, _ = cv2.Rodrigues(rotation_vec)
pose_mat = cv2.hconcat((rotation_mat, translation_vec))
_, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(pose_mat)
angles[0, 0] = angles[0, 0]*-1

# new way:
(nose_end_point2D, jacobian) = \
    cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vec,
                      translation_vec, camera_matrix, dist_coeffs)

p1 = (int(image_points[0][0]), int(image_points[0][1]))
p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
x1, x2 = draw_annotation_box(img, rotation_vec, translation_vec, camera_matrix)
m = (p2[1] - p1[1])/(p2[0] - p1[0])
ang1 = math.degrees(math.atan(m))
m = (x2[1] - x1[1])/(x2[0] - x1[0])
ang2 = math.degrees(math.atan(-1/m))

# continuous tracking:
mark_detector = MarkDetector()
cap = cv2.VideoCapture(0)
ret, img = cap.read()
size = img.shape
font = cv2.FONT_HERSHEY_SIMPLEX
# 3D model points.


# Camera internals
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array([[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype="double")

cv2.destroyAllWindows()
cap.release()
