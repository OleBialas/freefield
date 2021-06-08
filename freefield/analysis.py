import numpy as np
from scipy import stats


def double_to_single_pole(azimuth_double, elevation_double):
    azimuth_double, elevation_double = np.deg2rad(azimuth_double), np.deg2rad(elevation_double)
    azimuth_single = np.arctan(np.sin(azimuth_double) / np.cos(azimuth_double) / np.cos(elevation_double))
    return np.rad2deg(azimuth_single)


def single_pole_to_polar(azimuth, elevation):
    phi = -1 * azimuth
    theta = elevation - 90
    return phi, theta


def polar_to_single_pole(phi, theta):
    azimuth = phi * -1
    elevation = theta + 90
    return azimuth, elevation


def polar_to_cartesian(phi, theta):
    phi, theta = np.deg2rad(phi), np.deg2rad(theta)
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def mean_dir(data, speaker):
    # use vector addition with uncorrected angles:
    # sines, cosines = _sines_cosines(data, speaker)
    # return numpy.rad2deg(sines.sum(axis=1) / cosines.sum(axis=1)).flatten()
    # use regular addition with corrected angles:
    idx = np.where(data[:, 1] == speaker)
    return data[idx, 2:4].mean(axis=1)


def mad(data, speaker, ref_dir=None):
    'Mean absolute difference between reference directions and pointed directions'
    if ref_dir is None:
        ref_dir = mean_dir(data, speaker)
    idx = np.where(data[:,1] == speaker)
    diffs = data[idx,2:4] - ref_dir
    return np.sqrt((diffs**2).sum(axis=2)).mean()


def rmse(data, speaker, ref_dir=None):
    'Vertical and horizontal localization accuracies were quantified by computing the root mean square of the discrep- ancies between perceived and physical locations (RMSE, Hartmann, 1983; Savel, 2009).'
    if ref_dir is None:
        ref_dir = mean_dir(data, speaker)
    idx = np.where(data[:,1] == speaker)
    diffs = data[idx,2:4] - ref_dir
    dist = np.sqrt((diffs**2).sum(axis=2))
    return np.sqrt((dist**2).mean())


def eg(data, speaker_positions=None):
    '''
    Vertical localization performance was also quantified by the EG, defined as the slope of the linear regression of perceived versus physical elevations (Hofman et al., 1998). Perfect localization corresponds to an EG of 1, while random elevation responses result in an EG of 0.'''
    eles = data[:,3]
    if speaker_positions is None:
        return np.percentile(eles, 75) - np.percentile(eles, 25)
    speaker_seq = data[:,1].astype(int) # presented sequence of speaker numbers
    elevation_seq = speaker_positions[speaker_seq,1] # get the elevations for the speakers in the presented sequence
    regression = stats.linregress(eles, elevation_seq)
    return regression.slope
