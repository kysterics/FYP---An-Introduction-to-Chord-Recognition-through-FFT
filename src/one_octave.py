import itertools
import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import music21
import numpy as np
import audio2numpy as a2n
import seaborn as sns

from numpy import copy

from collections import Counter
from itertools import pairwise, chain
from more_itertools import unique_everseen
from random import randrange, sample, seed
from rounders import round_to_figures

from scipy.fft import fft, fftfreq
from scipy.io import wavfile
from scipy.signal import find_peaks
from scipy.signal import decimate

from sklearn.preprocessing import normalize

from music21.pitch import Pitch
from music21.chord import Chord
from music21.duration import Duration
from music21.instrument import Instrument
from music21.note import Note, Rest
from music21.stream import Stream
from music21.tempo import MetronomeMark
from music21.volume import Volume
from music21.dynamics import Dynamic

from music21 import instrument, clef

# for i in range(300):
#     try:
#         print(instrument.instrumentFromMidiProgram(i))
#     except music21.exceptions21.InstrumentException:
#         pass

sns.set()
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def generate_random_chords_within_an_octave(start, stop, n, k, duration=3, rest=1, gm_id=0, make_midi=True):
    """
    :param start: starting octave no. (inclusive)
    :param stop: stopping octave no. (exclusive)
    :param n: number of chords desired
    :param k: number of notes in a chord
    :param duration:
    :param rest:
    :param gm_id: instrument id (starting from 0) [https://en.wikipedia.org/wiki/General_MIDI#Parameter_interpretations]
    :param make_midi: boolean of whether to make midi file for the chords
    :return: list of chords
    """
    # midi_chords = [tuple(sorted(sample(list(np.arange(12) + randrange(start, stop - 12)), k=k))) for i in range(n)]

    midi_chords = [tuple(sorted(sample(list(np.arange(12) + (i + 1) * 12), k=k))) for i in range(start, stop) for j in range(n)]

    if not make_midi:
        return midi_chords

    chords = [Chord([Pitch(midi=note) for note in chord], duration=Duration(duration)) for chord in midi_chords]
    chords_with_rest = [e for chord in chords for e in [clef.bestClef(Stream([chord])),
                                                        chord,
                                                        Rest(rest)]]

    # chords_with_rest = list(itertools.chain.from_iterable(chords_with_rest))

    # Adapted from https://notebook.community/bzamecnik/ml-playground/instrument-classification
    # /music21_generating_pitches
    def set_instrument():
        i = Instrument()
        i.midiProgram = gm_id
        return i

    s = Stream([
        MetronomeMark(number=120),
        set_instrument()
    ])
    s.append(chords_with_rest)
    s.insert(0, Dynamic('fff'))
    s.show()

    return midi_chords


def f_pitch(p):
    if type(p) in [tuple, list, set]:
        d = [440 * 2 ** ((p_i - 69) / 12) for p_i in p]
    else:
        d = 440 * 2 ** ((p - 69) / 12)
    return d


def pitch_f(f):
    if type(f) in [tuple, list, set]:
        d = [str(Pitch(midi=np.round(69 + 12 * np.log2(f_i / 440)))) for f_i in f]
    else:
        d = str(Pitch(midi=np.round(69 + 12 * np.log2(f / 440))))
    return d


def chroma_mapping_m(freqs, signal):
    """
    Same as `chroma_mapping` except that the amplitudes of each octave are multiplied together instead
    """

    f_ref = 254
    binned_freqs = [f * 2 ** math.floor(1 - math.log(f / f_ref, 2)) if f != 0 else 0 for f in np.abs(freqs)]
    data = {}
    for freq, amplitude in zip(binned_freqs, signal):
        if freq in data:
            data[freq] = data.get(freq) * amplitude
        else:
            data[freq] = amplitude
    data.pop(0)  # remove the key with 0 frequency
    data_ordered = zip(*sorted(list(data.items())))
    work = np.array(list(data_ordered))
    # power = [math.floor(1 - math.log(f / f_ref, 2)) if f != 0 else 0 for f in np.abs(freqs)]
    # print(power)
    # print(max(power) + abs(min(power)))
    # work[1] **= 1 / (max(power) + abs(min(power)))
    return work


def fft_with_peaks(audio_file, peak_threshold, cent_threshold, n, method=1):
    """
    :param audio_file:
    :param peak_threshold:
    :param cent_threshold: allowance for deviance from the A440 standard (for each chroma)
    :param n: desired number of peaks
    :param method: 1 - Pure FFT; 2 - HPS of FFT
    :return:
    """
    # y, sample_rate = a2n.audio_from_file('out000.wav')
    sample_rate, y = wavfile.read(audio_file)
    if len(y[0]) == 2:  # only take the left channel if the input file is in stereo
        y = y[:, 1]
        # y = np.mean(y, axis=1)

    if method == 1:
        yf = np.abs(fft(y))
        xf = fftfreq(len(y), 1 / sample_rate)
        # frequency resolution = len(y) / sample_rate

        # Let's look at the spectrum with peaks
        peaks, properties = find_peaks(yf, height=peak_threshold, prominence=100)

        while not peaks.any() or len(peaks) < n * 6:
            peak_threshold *= .9
            peaks, properties = find_peaks(yf, height=peak_threshold, prominence=10)

        freqs = np.array_split(xf[peaks[:]], 2)[0]
        peaks = properties['peak_heights']
        chroma_labels = [pitch_f(f) for f in freqs]
        # chroma_labels_freqs = [str(int(f)) for f in freqs]

    if method == 2:

        def hps(dft, m=4):
            hps_len = int(np.ceil(np.size(dft) / (2 ** m)))
            # print(np.size(dft))
            # print(hps_len)
            hpss = np.ones(hps_len)
            for n in range(hps_len):
                for m_i in range(m + 1):
                    hpss[n] *= np.absolute(dft[(2 ** m_i) * n])
            # print(hps)

            # hpss = np.ones(18391)
            # for h in range(m + 1):
            #     dec = decimate(np.absolute(dft), 2 ** h)
            #     hpss[:len(dec)] *= dec
            # print(hpss)
            # return hpss

            return hpss

        yf = np.abs(fft(y))
        # yf = hps(yf)
        # yf *= 100.0 / yf.max()
        xf = fftfreq(len(yf), 1 / sample_rate)[:len(yf)]

        xf = fftfreq(len(yf), 1 / sample_rate)
        xf, yf = chroma_mapping_m(xf, yf)

        # Let's look at the spectrum with peaks
        peak_threshold = 10
        freq_threshold = 1 / 2  # allowance for deviance from the A440 standard (for each chroma)
        peaks, properties = find_peaks(np.abs(yf), height=peak_threshold)

        freqs = np.array_split(xf[peaks[:]], 2)[0]  # only taking the positive frequencies
        peaks = properties['peak_heights']
        chroma_labels = [pitch_f(f) for f in freqs]

    # Sieving out unwanted peaks
    d = {}  # dictionary with chroma_labels as keys, and freqs and peaks as values
    for freq, peak, cl in zip(freqs, peaks, chroma_labels):
        if freq < 10 or freq > 20000:
            continue
        if Pitch(cl).frequency < freq:
            freq_threshold = Pitch(cl).frequency * (2 ** (cent_threshold / 1200) - 1)
        else:
            freq_threshold = Pitch(cl).frequency * (1 - 2 ** (-cent_threshold / 1200))

        # only add freq and peak to their chroma label if
        if abs(Pitch(cl).frequency - freq) < freq_threshold and peak > d.get(cl, [0, 0])[1]:
            d[cl] = (freq, peak)

    try:
        chroma_labels, freqs, peaks = list(d.keys()), *zip(*d.values())
    except ValueError:
        chroma_labels, freqs, peaks = ([],) * 3

    # if plot:
    #     try:
    #         min_sep = .64 * np.log(freqs[-1] / freqs[0]) / min(np.log(y / x) for x, y in pairwise(freqs))
    #     except ValueError:
    #         min_sep = 6.4
    #
    #     if min_sep < 6.4:
    #         min_sep += 3.6
    #
    #     plt.rcParams['figure.figsize'] = [min_sep, 4.8]
    #     plt.semilogx(xf, np.abs(yf), '-', freqs, peaks, 'x')
    #     plt.subplots_adjust(bottom=0.2)
    #
    #     chroma_freqs_labels = [f'{f}\n~{cl[:-1]}$_{cl[-1]}$\n[{int(Pitch(cl).frequency)}]'
    #                            for cl, f in zip(chroma_labels, map(int, freqs))]
    #
    #     plt.minorticks_off()
    #     # plt.xticks(rotation=0, fontsize='small')
    #     plt.xticks(freqs, chroma_freqs_labels)
    #     plt.xlim(freqs[0] / 2 ** (2 / 12), freqs[-1] * 2 ** (2 / 12))
    #     plt.title(f"{audio_file} (height={peak_threshold})")
    #     plt.xlabel('Frequency (log scale)')
    #     plt.ylabel('Power')
    #     plt.show()

    # try:
    #     min_sep = .64 * np.log(freqs[-1] / freqs[0]) / min(np.log(y / x) for x, y in pairwise(freqs))
    # except ValueError:
    #     min_sep = 6.4
    #
    # if min_sep < 6.4:
    #     min_sep += 3.6
    #
    # plt.rcParams['figure.figsize'] = [min_sep, 4.8]

    # plt.plot(xf, np.abs(yf), '-', freqs, peaks, 'x')
    # plt.subplots_adjust(bottom=0.2)
    #
    # chroma_freqs_labels = [f'{f}\n~{cl[:-1]}$_{cl[-1]}$\n[{int(Pitch(cl).frequency)}]'
    #                        for cl, f in zip(chroma_labels, map(int, freqs))]
    #
    # plt.minorticks_off()
    # # plt.xticks(rotation=0, fontsize='small')
    # plt.xticks(freqs, chroma_freqs_labels)
    # plt.xlim(freqs[0] / 2 ** (2 / 12), freqs[-1] * 2 ** (2 / 12))
    # plt.xscale('log')
    # plt.title(f"{audio_file} (height={peak_threshold})")
    # plt.xlabel('Frequency (log scale)')
    # plt.ylabel('Power')
    # plt.show()

    return xf, yf, chroma_labels, freqs, peaks


def main(method, midi_chords, peak_threshold, cent_threshold, plot_lst=[], plot=False):
    """
    Checks whether the identified chords match with the actual ones and plot if any does not
    :param method: 1 - Pure FFT; 2 - HPS of FFT
    :param midi_chords: list of chords (in MIDI numbers)
    :param peak_threshold: to maximize the accuracy of peaks
    :param cent_threshold: deviance in cents from the A440 standard allowed
    :param plot_lst: list of chords by numbers (in positive integers) to plot the FFT of
    :param plot: Boolean of whether to plot or not
    :return: None
    """
    # data = [pitch_f(f_pitch(chord)) for chord in midi_chords]
    data = [[str(Pitch(midi=note)) for note in chord] for chord in midi_chords]

    data_exp = list(chain.from_iterable(data))
    # bins = [i * 12 for i in range(11)]
    # plt.hist(data_exp)
    # plt.show()

    plt.rcParams['figure.figsize'] = [30, 4.8]
    res = Counter(data_exp)
    res = sorted(res.items(), key=lambda pair: pair[0][-1])
    plt.bar(*zip(*res))
    plt.xticks(rotation=90, fontsize='small')
    plt.show()

    stop = len(midi_chords)
    p = len(str(stop))  # miscellaneous: padding for numbers
    n = len(midi_chords[0])  # n-note chord

    tf_count = []
    f_data = []  # notes that are not recognized
    f_data_count = 0
    for i in range(stop):
        # audio file split using the following sox command:
        # sox 30.wav 30_3__.wav silence 1 0.1 1% 1 0.1 1% : newfile : restart
        audio_file = f'{stop}/{stop}_{n}__{str(i + 1).zfill(p)}.wav'

        try:
            xf, yf, chroma_labels, freqs, peaks = fft_with_peaks(audio_file,
                                                                 peak_threshold,
                                                                 cent_threshold,
                                                                 n,
                                                                 method=method)
        except TypeError:
            print(str(i + 1).zfill(p) + ':', 'True chord:', data[i], 'Identified chord:', '[]')
            continue

        if chroma_labels[:n] == data[i]:
            tf_count.append(True)
            print(str(i + 1).zfill(p) + ':', 'Match')
        else:
            tf_count.append(False)
            print(str(i + 1).zfill(p) + ':', 'True chord:', data[i], 'Identified chord:', chroma_labels[:n])

            if chroma_labels:
                f_data.append([f'{cl[:-1]}$_{cl[-1]}$' for cl in set(data[i]) - set(chroma_labels[:n])])
            else:
                f_data.append([f'{cl[:-1]}$_{cl[-1]}$' for cl in data[i]])

            f_data_count += sum(int(j[-1]) for j in data[i])

            if plot and (i + 1 in plot_lst):
                # Setting graph width
                try:
                    # min/max separation between peaks
                    min_sep = min(np.log(y / x) for x, y in pairwise(freqs))
                    max_sep = np.log(freqs[-1] / freqs[0])

                    width = .64 * max_sep / min_sep
                    if width < 6.4:
                        width = 6.4
                except ValueError:
                    width = 6.4

                plt.rcParams['figure.figsize'] = [width, 4.8]

                # Plotting FFT graph of chords that failed to match
                plt.semilogx(xf, yf, '-', freqs, peaks, 'x')
                plt.subplots_adjust(bottom=0.2)

                chroma_freqs_labels = [f'{f}\n~{cl[:-1]}$_{cl[-1]}$\n[{int(Pitch(cl).frequency)}]'
                                       for cl, f in zip(chroma_labels, map(int, freqs))]

                plt.minorticks_off()
                plt.xticks(freqs, chroma_freqs_labels)
                plt.xlim(freqs[0] / 2 ** (13 / 12), freqs[-1] * 2 ** (2 / 12))
                if method == 1:
                    plt.title(f'Pure FFT on {audio_file} (height={peak_threshold})')
                if method == 2:
                    plt.title(f'HPS of FFT on {audio_file} (height={peak_threshold})')
                plt.xlabel('Frequency (log scale)')
                plt.ylabel('Power')
                plt.figtext(.79, .82, 'Frequencies in [ ] are in 12-TET tuned relative to A440')
                plt.show()

    print('Accuracy:', f'{tf_count.count(True)}/{stop} =', tf_count.count(True)/stop)

    f_data = list(chain.from_iterable(f_data))
    print('Average octave:', f'{f_data_count / len(f_data)}')

    # octave_labels = [j for j in range(max(sorted(set(f_data))) + 1)]
    # bins = [k for j in octave_labels for k in (j - .4, j + .4)]
    # plt.hist(f_data, bins=bins, histtype='bar', align='mid')
    # plt.xticks(octave_labels)

    # plt.rcParams['figure.figsize'] = [16.4, 4.8]
    # res = Counter(f_data).most_common()
    # plt.bar(*zip(*res))

    # plt.rcParams['figure.figsize'] = [20, 4.8]
    # res = Counter(f_data)
    # res = sorted(res.items(), key=lambda pair: pair[0][-2])
    # plt.bar(*zip(*res))
    # plt.show()

    plt.rcParams['figure.figsize'] = [6.4, 4.8]
    fig, ax = plt.subplots()
    res = Counter([j[-1] for i in data for j in i])
    x1, y1 = zip(*sorted(res.items()))
    plt.bar(x1, y1, width=.7, color='C4')
    plt.ylim(0, max(y1) * 1.2)

    res = Counter([i[-2] for i in f_data])
    x2, y2 = zip(*sorted(res.items()))
    plt.bar(x2, y2, width=.7, color='C8', hatch='/')

    ax.bar_label(ax.containers[1],
                 labels=[f' {round_to_figures(100 * failed/total, 3)}%' for total, failed in zip(y1, y2)],
                 label_type='center',
                 fontsize=10.5,
                 rotation=45)

    plt.legend(['Recognized notes', 'Unrecognized notes'])
    # plt.title(f'Recognition result of 1000 random 3-note piano chords within an octave')
    plt.title(f'Note count against octave number')
    plt.xlabel('Octave number')
    plt.ylabel('Note count')
    plt.savefig('arst')
    plt.show()

    # plt.rcParams['figure.figsize'] = [6.4, 4.8]
    # res = Counter([j[-1] for i in data for j in i])
    # res = sorted(res.items(), key=lambda pair: pair[0])
    # plt.bar(*zip(*res), width=.7, edgecolor='C0')
    # plt.bar(0, 0, width=.7, color='C8', edgecolor='C8')
    # # plt.ylim(0, 600)
    #
    # res = Counter([i[-2] for i in f_data])
    # res = sorted(res.items(), key=lambda pair: pair[0])
    # plt.bar(*zip(*res), hatch='//', width=.7, color='C8', edgecolor='C0')
    #
    # plt.legend(['All notes', 'Unrecognized notes'])

    # handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * 2
    # labels = ["pi = {0:.4g}".format(np.pi),
    #           "root(2) = {0:.4g}".format(np.sqrt(2))]
    #
    # # create the legend, supressing the blank space of the empty line symbol and the
    # # padding between symbol and label by setting handlelenght and handletextpad
    # plt.legend(handles, labels, loc='best', fontsize='small',
    #            fancybox=True, framealpha=0.7,
    #            handlelength=0, handletextpad=0)

    # plt.show()


if __name__ == '__main__':
    seed(314)
    # generate_random_chords_within_an_octave(12, 108, 1000, 3, gm_id=1)
    # lst = generate_random_chords_within_an_octave(21, 109, 3000, 3, gm_id=1, make_midi=True)
    lst = generate_random_chords_within_an_octave(0, 8, 100, 3, duration=3, rest=1, gm_id=1, make_midi=True)

    # If this is deployed in a situation where one does not know have answers to compare to,
    # then one workaround could be to first identify a handful of chords by hand in order to tune the thresholds
    # such that they can match more chords within the same audio sample

    main(1, lst, 155, 20, plot_lst=[163, 164], plot=True)

    # lst_exp = list(chain.from_iterable(lst))
    # bins = [i * 12 for i in range(12)]
    # plt.hist(lst_exp, bins=bins)
    # plt.show()

    # main(2, plot=False)  # unfinished

# conditions for method 1 to work
# 1. Need to know how to set good thresholds
# 2. Knowledge of the tuning (eg if the piano is tuned to the A440 standard)
# 3. Not too noisy
# 4. The range of notes cannot be too extreme
# 5. Piano in tune
# 6. Need to know the number of chords you're looking for

# TODO
# A444 standard shift
#
