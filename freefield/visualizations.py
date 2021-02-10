from mpl_toolkits.mplot3d import Axes3D
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


def plot_sources(azimuth, elevation, distance=1.6):
    """Display sources in a 3D plot
    Arguments:
        azimuth (np.ndarray): azimuth of the sources in degree. Must be same length as elevation
        elevation (np.ndarray): elevation of the sources in degree. Must be same length as azimuth
        distance (float | np.ndarray): distance of the sources to the listener. Can be either an array
            with the same length as azimuth and elevation or a single float if all sources have the same
            distance - meaning they are arranged in a sphere or circle. """
    ax = Axes3D(plt.figure())
    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation-90)
    x = distance * np.sin(elevation) * np.cos(azimuth)
    y = distance * np.sin(elevation) * np.sin(azimuth)
    z = distance * np.cos(elevation)
    ax.scatter(x, y, z, c='b', marker='.')
    ax.scatter(0, 0, 0, c='r', marker='o')


if __name__ == '__main__':
    speakers = main.read_table("dome")
    azi = np.array(speakers["azi"].to_list())
    ele = np.array(speakers["ele"].to_list())
    plot_sources(azi, ele)

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
