'''
Experiment Mira:
Is motion adaption caused by ILD/ITD ramps?
'''

# Bibliotheken  importieren

import time
import functools
import numpy
import scipy
import slab

# Voreinstellungen treffen

slab.Signal.set_default_samplerate(44100)
_speaker_positions = numpy.arange(-90, 0.01, 4)
_adaptor_speed = 180
_adaptor_dir = 'left'
_n_adaptors_per_trial = 3
_n_blocks_per_adaptorversion = 2
_after_stim_pause = 0.1
_jnd_diff_thresh = 1.5

_results_file = None  #??? was sagt das aus?
slab.ResultsFile.results_folder = 'Results'


def moving_gaussian(speed=100, width=7.5, SNR=10, direction='left', externalize=True):
    '''
    Make a wide Gauss curve shaped stimulus that moves horizontally across
    a range of virtual loudspeakers. This is the base stimulus of the experiment.
    '''
    if direction == 'left':
        dir = -1
        starting_loc = _speaker_positions[-1]
    else:
        dir = 1
        starting_loc = _speaker_positions[0]

    def loc(time):
        return (speed * time) * dir + starting_loc # =end-location
    # make times vector from speed and positions angle difference
    end_time = _speaker_positions.ptp() / speed  #???
    time_delta = 0.01  # 10 ms
    times = numpy.arange(0, end_time + time_delta, time_delta)
    # step through time, saving speaker amplitudes for each step
    speaker_amps = numpy.zeros((len(_speaker_positions), len(times)))
    for idx, t in enumerate(times): # enumerate erzeugt dictionnairy mit [(0,thing0), (1, thing1), ...] --> also erzeut Index zu Objekten in der Liste
        speaker_amps[:, idx] = scipy.stats.norm.pdf(_speaker_positions, loc=loc(t), scale=width)
    # scale the amplitudes to max 0, min -SNR dB
    maximum = scipy.stats.norm.pdf(0, loc=0, scale=width)
    minimum = speaker_amps.min()
    speaker_amps = numpy.interp(speaker_amps, [minimum, maximum], [-SNR, 0]) #= Lautstärke
    speaker_signals = []
    for i, speaker_position in enumerate(_speaker_positions):
        sig = slab.Binaural.pinknoise(duration=end_time)
        sig = sig.at_azimuth(azimuth=speaker_position)
        sig = sig.envelope(apply_envelope=speaker_amps[i, :], times=times, kind='dB')
        speaker_signals.append(sig)
    sig = speaker_signals[0]
    for speaker_signal in speaker_signals[1:]:  # add sounds
        sig += speaker_signal
    sig /= len(_speaker_positions)
    sig.ramp(duration=end_time/3)  # ramp the sum
    sig.filter(frequency=[500,14000], kind='bp')
    if externalize:
        sig = sig.externalize() # apply smooth KEMAR HRTF to move perceived source outside of the head, kann noch verbessert werden
    sig.level = 75
    return sig

def familiarization():
    '''
    Presents the familiarization stimuli (100% modulation depth, random direction)
    '''
    print('Familiarization: sounds moving left or right are presented.')
    print('The direction should be easy to hear.')
    print('Press 1 for left, 2 for right.')
    input('Press enter to start familiarization (2min)...')
    repeat = 'r'
    while repeat == 'r':
        trials = slab.Trialsequence(
            conditions=['left', 'right'], n_reps=10, kind='random_permutation')
        responses = []
        _results_file.write('familiarization:', tag='time')
        for dir in trials:
            stim = moving_gaussian(speed=_adapter_speed, SNR=100, direction=dir)
            stim.play()  # present
            with slab.psychoacoustics.key() as key:  # and get response
                resp = key.getch()
            if dir == 'left':  # transform response: left = key '1', right = key '2'
                resp = resp == 49
            else:
                resp = resp == 50
            responses.append(resp)
            time.sleep(_after_stim_pause)
        # compute hitrate
        hitrate = sum(responses)/trials.n_trials  #wie funktioniert das???
        print(f'hitrate: {hitrate}')
        _results_file.write(hitrate, tag='hitrate')
        repeat = input('Press enter to continue, "r" to repeat familiarization.')


    return hitrate

def practice_stairs():
    '''
    Presents an easy large-step staircase to practice.
    '''
    print()
    print('This is a practise run. Explain the procedure to the participant.')
    print('Show the running staircase to the participant.')
    print()
    print('One sound is presented in each trial.')
    print('Is this sound moving left or right?')
    print('Press 1 for left, 2 for right.')
    print('The direction will get more and more difficult to hear.')
    input('Press enter to start practice...')
    _results_file.write('familiarization:', tag='time')
    stairs = slab.Staircase(start_val=24, n_reversals=6,
                            step_sizes=[10, 6, 4], min_val=1, max_val=30, n_up=1, n_down=1, n_pretrials=1)
    for trial in stairs:
        direction = numpy.random.choice(('left', 'right'))
        stim = moving_gaussian(speed=_adapter_speed, SNR=trial, direction=direction)
        stairs.present_tone_trial(
            stimulus=stim, correct_key_idx=1 if direction == 'left' else 2, print_info=True)
        stairs.plot()

    thresh = stairs.threshold()
    print(f'threshold: {thresh}')
    _results_file.write(thresh, tag='threshold')
    input('Done. Press enter to continue...')
    stairs.close_plot()

#Pre-make many adaptor instances to speed-up constructing the stimuli.
def make_spatial_adaptors():

    kwargs = {'speed': _adaptor_speed, 'SNR': 100, 'direction': _adaptor_dir, 'externaliz'=True}
    make_adaptor = functools.partial(moving_gaussian, **kwargs)
    return slab.Precomputed(make_adaptor, _n_adaptors_per_trial * 1)

def make_binaural_adaptors():

    kwargs = {'speed': _adaptor_speed, 'SNR': 100, 'direction': _adaptor_dir, 'externaliz'=False}
    make_adaptor = functools.partial(moving_gaussian, **kwargs)
    return slab.Precomputed(make_adaptor, _n_adaptors_per_trial * 1)


def jnd(adaptor):
    '''
    Presents a staricase of moving_gaussian stimuli with varying SNR and returns the threshold.
    This threshold is used in the main experiment as the listener-specific SNR parameter.
    '''
    print()
    if adaptor:
        print(f'{_n_adaptors_per_trial} sounds moving {_adaptor_dir}, followed by one sound moving left or right is presented.')
        print('Is this last sound moving left or right?')
    else:
        print('One sound is presented in each trial.')
        print('Is this sound moving left or right?')
    print('Press 1 for left, 2 for right.')
    print('The direction will get more and more difficult to hear.')
    input('Press enter to start JND estimation...')
    repeat = 'r'
    while repeat == 'r':

        stairs = slab.Staircase(start_val=24, n_reversals=5,
                                    step_sizes=[8, 6, 4, 3, 2, 1], min_val=0, max_val=30, n_up=1, n_down=1, n_pretrials=2)

        _results_file.write('jnd:', tag='time')

        for trial in stairs:
            direction = numpy.random.choice(('left', 'right'))

                #Stimulus erzeugen:
            probe = moving_gaussian(speed=150, SNR=trial, direction=direction)

            if adaptor == None:
                stim = probe

            else:
                adaptation = adaptor.random_choice(
                    n=_n_adaptors_per_trial)  # some variety in the adaptors
                adaptation.append(slab.Binaural.silence(duration=0.3, samplerate=44100)) #wie kurze Pause einbauen
                adaptation.append(probe)  # add probe to list of adaptors
                stim = slab.Sound.sequence(*adaptation)  # concatenate sounds (6 Adaptoren + Probe) in the list



            stairs.present_tone_trial(stimulus=stim, correct_key_idx=0 if direction == 'left' else 1)
#funtioniert manchmal nicht. Alternativ slab.psychoacoustics.input_method = 'figure'oben festlegen,kann mit dem stairsplot verbunden werden
        thresh = stairs.threshold() #n=x berechnet threshold der letzten x Umkehrungen


        #print(f'jnd for {adaptor}: {round(thresh, ndigits=1)}') #??? wie beziehen auf Adaptor-Namen?
        #_results_file.write(thresh, tag=adaptor) ??? geht nicht, weil Adaptor sich auf den Datensatz bezieht und nicht den Variablennamen

        repeat = input('Press r to repeat this threshold measurement. Press enter to continue.')

    return thresh



def main_experiment(subject=None):
    '''
    Beschreibung einfügen
    '''
    global _results_file
    # set up the results file
    if not subject:
        subject = input('Enter subject code: ')
    _results_file = slab.ResultsFile(subject=subject)

    print('Make adaptors and results-table:')
    spatial_adaptor_precomp = make_spatial_adaptors()
    print('spatial_adaptors completed')
    binaural_adaptor_precomp = make_binaural_adaptors()
    print('binaural_adaptors completed')

    adaptor_types = [spatial_adaptor_precomp, binaural_adaptor_precomp, None]
    adaptor_names = ['spatial adaptor', 'binaural adaptor', 'no adaptor']]
    jnds = numpy.zeros((len(adaptor_types),_n_blocks_per_adaptorversion ))
    repeats=list()

    familiarization()
    practice_stairs()

    print()
    print('The main part of the experiment starts now (motion direction thresholds).')
    _results_file.write('main experiment', tag='time')
    for i in range (_n_blocks_per_adaptorversion):
        adaptor_types_seq = slab.Trialsequence (conditions = (0,1,2), n_reps=1)
        for idx in adaptor_types_seq:
            print ('Adaptor: ', adaptor_names[idx])
            thresh=jnd(adaptor = adaptor_types[idx])
            jnds[idx,i]=thresh
            print(f'jnd for {adaptor_names[idx]}: {round(thresh, ndigits=1)}')
            _results_file.write(thresh, tag=adaptor_names[idx])
            
        # save a string representation of the numpy results table
        _results_file.write(str(jnds), tag=f'results round {i}')

        if i == 1:
            if abs(jnds[idx,1] -jnds[idx,0])>_jnd_diff_thresh:
                repeats.append(idx)
                print ('Difference to first JND too large. Marked for repetition!')

    for idx in repeats:
        print ('Repetition of measurement with large diference between JNDs with same conditions')
        jnd(adaptor = adaptor_types[idx])
        # these will just be tagged in the results file, not in the tables





main_experiment(subject='test')


# Ergebnisse speichern
