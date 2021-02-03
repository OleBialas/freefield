# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from math import *
from freefield import main
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import pandas as pd


def plot_face_detection_marks(image):
    from freefield import headpose
    import cv2
    model = headpose.PoseEstimator(threshold=.99)
    face_boxes = model.extract_cnn_facebox(image)
    face_box = face_boxes[0]
    face_img = image[face_box[1]: face_box[3], face_box[0]: face_box[2]]
    face_img = cv2.resize(face_img, (128, 128))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    marks = model.detect_marks([face_img])
    marks *= (face_box[2] - face_box[0])
    marks[:, 0] += face_box[0]
    marks[:, 1] += face_box[1]
    plt.imshow(image, cmap="gray")
    plt.scatter(marks[:, 0], marks[:, 1], color="red", marker=".")


def plot_sources():

    pass

"""
def plot_dome(speakers="all"):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect("auto")
    # draw sphere
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="r")

    # plot point representing the head of the listener, at the origin of the dome
    ax.scatter([0], [0], [0], color="black", s=100)
    # plot arrow representing gaze direction:
    a = Arrow3D([0, 1], [0, 0], [0, 0], mutation_scale=20,
                lw=1, arrowstyle="-|>", color="k")
    ax.add_artist(a)

    # draw a point for each speaker:
    speakers = main.read_table("dome")
    speakers = speakers[["azi", "ele"]]
    speakers = speakers.where(speakers["azi"] == 0.0).dropna()
    speakers = speakers.append(pd.DataFrame([[0.0, 55.5], [0.0, -55.5]], columns=["azi", "ele"]))
    for index, row in speakers.iterrows():
        spherical_point = SphericalPoint(r=1, theta=row.ele, phi=row.azi)
        print(spherical_point)
        print(spherical_point.degrees())

        point = spherical_point.to_cartesian()
        print(point)
        ax.scatter([point.x], [point.y], [point.z], color="green", s=100)

    plt.show()


class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class Point(object):
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return '(%0.4f, %0.4f, %0.4f)' % (self.x, self.y, self.z)

    def __repr__(self):
        return 'Point(%f, %f, %f)' % (self.x, self.y, self.z)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, f):
        return Point(self.x * f, self.y * f, self.z * f)

    def dist(self, other):
        p = self - other
        return (p.x ** 2 + p.y ** 2 + p.z ** 2) ** 0.5

    def to_spherical(self):
        r = self.dist(Point(0, 0, 0))
        theta = atan2(hypot(self.x, self.y), self.z)
        phi = atan2(self.y, self.x)
        return SphericalPoint(r, theta, phi)


class SphericalPoint(object):
    def __init__(self, r, theta, phi):
        # radial coordinate, zenith angle, azimuth angle
        self.r = r
        self.theta = theta
        self.phi = phi

    def degrees(self):
        return 'SphericalPoint(%.4f, %.4f deg, %.4f deg)' % (self.r, degrees(self.theta) % 360, degrees(self.phi) % 360)

    def __str__(self):
        return '(%0.4f, %0.4f, %0.4f)' % (self.r, self.theta, self.phi)

    def __repr__(self):
        return 'SphericalPoint(%f, %f, %f)' % (self.r, self.theta, self.phi)

    def to_cartesian(self):
        x = self.r * cos(self.phi) * sin(self.theta)
        y = self.r * sin(self.phi) * sin(self.theta)
        z = self.r * cos(self.theta)
        return Point(x, y, z)
"""

if __name__ == '__main__':
    plot_dome()



# def _plot_equalization(target, signal, filt, speaker_nr, low_cutoff=50,
#                        high_cutoff=20000, bandwidth=1/8):
#     """
#     Make a plot to show the effect of the equalizing FIR-filter on the
#     signal in the time and frequency domain. The plot is saved to the log
#     folder (existing plots are overwritten)
#     """
#     row = speaker_from_number(speaker_nr)  # get the speaker
#     signal_filt = filt.apply(signal)  # apply the filter to the signal
#     fig, ax = plt.subplots(2, 2, figsize=(16., 8.))
#     fig.suptitle("Equalization Speaker Nr. %s at Azimuth: %s and "
#                  "Elevation: %s" % (speaker_nr, row[2], row[3]))
#     ax[0, 0].set(title="Power per ERB-Subband", ylabel="A")
#     ax[0, 1].set(title="Time Series", ylabel="Amplitude in Volts")
#     ax[1, 0].set(title="Equalization Filter Transfer Function",
#                  xlabel="Frequency in Hz", ylabel="Amplitude in dB")
#     ax[1, 1].set(title="Filter Impulse Response",
#                  xlabel="Time in ms", ylabel="Amplitude")
#     # get level per subband for target, signal and filtered signal
#     fbank = slab.Filter.cos_filterbank(
#         1000, bandwidth, low_cutoff, high_cutoff, signal.samplerate)
#     center_freqs, _, _ = slab.Filter._center_freqs(low_cutoff, high_cutoff, bandwidth)
#     center_freqs = slab.Filter._erb2freq(center_freqs)
#     for data, name, color in zip([target, signal, signal_filt],
#                                  ["target", "signal", "filtered"],
#                                  ["red", "blue", "green"]):
#         levels = fbank.apply(data).level
#         ax[0, 0].plot(center_freqs, levels, label=name, color=color)
#         ax[0, 1].plot(data.times*1000, data.data, alpha=0.5, color=color)
#     ax[0, 0].legend()
#     w, h = filt.tf(plot=False)
#     ax[1, 0].semilogx(w, h, c="black")
#     ax[1, 1].plot(filt.times, filt.data, c="black")
#     fig.savefig(_location.parent/Path("log/speaker_%s_equalization.pdf"
#                                       % (speaker_nr)), dpi=800)
#     plt.close()
