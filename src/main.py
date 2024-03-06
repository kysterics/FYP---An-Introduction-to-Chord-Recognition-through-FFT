import matplotlib
import scipy
import math
import shortuuid

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import audio2numpy as a2n
import seaborn as sns
from operator import itemgetter
from itertools import pairwise
from more_itertools import unique_everseen

from scipy.fft import fft, fftfreq
from scipy.io import wavfile
from scipy.io.wavfile import write
from scipy.signal import find_peaks, decimate

from numpy import copy

from music21 import *

sns.set()
matplotlib.rcParams['savefig.dpi'] = 300

# configure.run()  # might be necessary for first use of music21

# I will start from coding up pitch recognition using FT detailed in Chapter 2 from 'Fundamentals of Music
# Processing' by Meinard Müller, looking at different samples like from a tuning fork, piano or a human voice. I
# intend on looking at autocorrelation as well to compare methods if appropriate or time allows. Then for chord
# recognition, I hope to follow the approach detailed by the paper you attached. Everything will be in Python.


# sin_wave = np.sin(2 * np.pi * f * t)
# cos_wave = 2*m.cos(2*np.pi*f*t) + 5*m.cos(2*np.pi*f*2*t)


# plt.show()

# Analog Case


"""
Note recognition
"""
# Partly adapted from https://realpython.com/python-scipy-fft/

# Set up parameters for the test signal
test_freq = 5.0  # Hertz
sample_rate = 44100  # Hertz
duration = 1  # Seconds

N = sample_rate * duration  # Number of samples

# Generate sample signal
t = np.linspace(0, duration, sample_rate * duration, endpoint=False)
y = np.sin(2 * np.pi * test_freq * t)

# Figure
# Plot the signal for checking
# plt.rcParams["figure.figsize"] = (10, 5)
plt.plot(t, y)
plt.title('5 $Hz$ Sine Wave')
plt.xlabel('Time ($s$)')
plt.ylabel('Amplitude')
plt.savefig(shortuuid.uuid())
plt.show()

# FFT of test signal
yf = fft(y)
xf = fftfreq(N, 1 / sample_rate)

# Figure
# Plot FFT results
plt.plot(xf, np.abs(yf))
plt.xlim(-test_freq * 2, test_freq * 2)
plt.title('FFT on 5 $Hz$ Sine Wave')
plt.xlabel('Frequency ($Hz$)')
plt.ylabel('Power')
plt.savefig(shortuuid.uuid())
plt.show()

"""
Chord recognition
"""

# Set up parameters for the test signal
sample_rate = 44100  # Hertz
duration = 2  # Seconds

N = sample_rate * duration  # Number of samples

# To generate the Tristan chord as the sample signal (F3, B3, D#4, G#4)
# Description of the chord
tristan = chord.Chord(['F3', 'B3', 'D#4', 'G#4'])
# tristan.show(fmt='lily.png')
tristan.commonName  # enharmonic equivalent to half-diminished seventh chord
tristan_freqs = [i.frequency for i in tristan.pitches]  # or Equation 1.1 from Ch.1 to generate frequency using MIDI no.

# To generate sample signal
t = np.linspace(0, duration, N, endpoint=False)
# Set up a dot product so the coefficient of each component can be adjusted separately
waves = np.array([np.sin(2 * np.pi * f * t) for f in tristan_freqs])
coeff = np.ones(len(waves))
# coeff = np.arange(4, 0, -1)
y = np.dot(coeff, waves)

# Figure
# Plot the signal
plt.rcParams['figure.figsize'] = (10, 5)
plt.plot(t, y)
plt.title('Tristan chord sample')
plt.xlabel('Time ($s$)')
plt.ylabel('Amplitude')

# Plot the zoomed portion
sub_axes = plt.axes([.78, .75, .2, .2])  # Location for the zoomed portion
sub_axes.plot(t, y, c='k')
# sub_axes.tick_params(axis='x', colors='red')
plt.xlim(.95, 1.05)
plt.savefig(shortuuid.uuid())
plt.show()

# Normalize the signal, so it can be saved as a WAV file
nice_tone = y

# noise_tone = np.sin(2 * np.pi * 4000 * t)
# noise_tone = noise_tone * 0.9
# nice_tone = nice_tone + noise_tone

normalized_tone = np.int16((nice_tone / nice_tone.max()) * 32767)
write('tristan.wav', sample_rate, normalized_tone)

# FFT of test signal
yf = fft(normalized_tone, norm='forward')  # 'forward is chosen so that it gives amplitudes
xf = fftfreq(N, 1 / sample_rate)

# Figure
# Plotting how the real and imaginary parts contribute to the transform
plt.rcParams['figure.figsize'] = [6.4, 6.4]
fig, axs = plt.subplots(3, 1)

axs[0].plot(xf, np.real(yf), c='g')
axs[0].set_title('Using the real part of the transform')

axs[1].plot(xf, np.imag(yf), c='orange')
axs[1].set_title('Using the imaginary part of the transform')

axs[2].plot(xf, np.abs(yf))
axs[2].set_title('Using the norm of the transform')

for i in range(3):
    axs[i].set_xlim(-(tristan_freqs[0] + tristan_freqs[-1]), tristan_freqs[0] + tristan_freqs[-1])
    axs[i].set_ylabel('Amplitude')
    if i == 2:
        plt.xlabel('Frequency ($Hz$)')
    if i != 2:
        axs[i].set_ylim(-4.5e3, 4.5e3)
        axs[i].set_xticklabels([])

fig.tight_layout()
plt.savefig(shortuuid.uuid())
plt.show()

# TODO
# Why are the spikes of different powers when the coefficients are the same?
# What if we add noise to the signal?
# What is the negative x-axis?

# Figure
# Plot FFT results with peaks
peaks, properties = find_peaks(np.abs(yf), height=1000)  # The amplitude threshold is set as 1000 arbitrarily
plt.rcParams['figure.figsize'] = [6.4, 4.8]
plt.plot(xf, np.abs(yf), '-', xf[peaks], properties['peak_heights'], 'x')
plt.xlim(0, tristan_freqs[0] + tristan_freqs[-1])  # only interested in the positive x-axis
plt.title('FFT on the Tristan chord')
plt.xlabel('Frequency ($Hz$)')
plt.ylabel('Amplitude')
plt.savefig(shortuuid.uuid())
plt.show()

# Peak frequencies determined from FFT
xf[peaks[:4]]  # [174.5 247.  311.  415.5]
# Frequencies we started off with
np.round(tristan_freqs, 1)  # [174.6 246.9 311.1 415.3]


# [print("%4.4f    \t %3.4f" % (xf[peaks[i]], properties['peak_heights'][i])) for i in range(len(peaks))]

def chroma_mapping(freqs, signal):
    """
    Squeeze signals from each octave together and add up their contributions
    Returns an array of 2 arrays – frequency and signal.
    """
    # freqs = np.split(freqs, 2)[-1]
    # signal = np.split(signal, 2)[-1]
    # Collapsing frequencies of the same pitch class into the same arbitrarily chosen bin (between C4 and C5)
    f_ref = 254  # 254Hz is chosen as the reference(starting) frequency as it sits between C4 and B3
    binned_freqs = [f * 2 ** math.floor(1 - math.log(f / f_ref, 2)) if f != 0 else 0 for f in np.abs(freqs)]
    data = {}
    for freq, amplitude in zip(binned_freqs, signal):
        data[freq] = data.get(freq, 0) + amplitude
    data.pop(0)  # remove the key with 0 frequency
    data_ordered = zip(*sorted(list(data.items())))
    return np.array(list(data_ordered))


# Figure
# Plot FFT results with peaks
# Note that the amplitude is doubled as the whole x-axis is collapsed into the range
# Note that the peaks sit mostly right on top of their own chroma labels as we are dealing with an artificial signal
# Note that this is in a linear scale
plt.plot(*chroma_mapping(xf, np.abs(yf)))

# Setting up chroma labels
octave = chord.Chord([i for i in range(12)])
octave_freqs = [i.frequency for i in octave.pitches]
chroma_labels = [i.name for i in octave.pitches]
plt.xticks(octave_freqs, chroma_labels)

plt.title('FFT on the Tristan chord')
plt.xlabel('Chroma ($A_{4}$ = 440 $Hz$)')
plt.ylabel('Amplitude (Cumulative)')
plt.savefig(shortuuid.uuid())
plt.show()

# Let's look at another half-diminished seventh chord using the first chord of Chopin Scherzo No.1 taken from
# https://www.youtube.com/watch?v=9VG2a37A64g
y, sample_rate = a2n.audio_from_file('scherzo1_1.wav')
yf = fft(y, norm='forward')
xf = fftfreq(len(y), 1 / sample_rate)

xf_binned, yf_binned = chroma_mapping(xf, np.abs(yf))

# Figure
# Note that the x-axis is on a log scale now
plt.rcParams['figure.figsize'] = [8, 4.8]
plt.plot(xf_binned, yf_binned)
plt.xscale('log')
plt.minorticks_off()
plt.xticks(octave_freqs, chroma_labels)
plt.title('FFT on a real sample of the 1st chord of Chopin Scherzo No.1')
plt.xlabel('Chroma ($A_{4}$ = 440 $Hz$)')
plt.ylabel('Amplitude (Cumulative)')
plt.savefig(shortuuid.uuid())
plt.show()


# Rearranging Equation 1.1 (f_p = 2 ** ((p - 69) / 12) * 440) to obtain the (closest) chroma/pitch from a frequency
def chroma_f(f):
    return str(pitch.Pitch(np.round(69 + 12 * math.log(f / 440, 2)) % 12))


def pitch_f(f):
    return str(pitch.Pitch(np.round(69 + 12 * math.log(f / 440, 2))))


# Verifying results using 'First Chord of Chopin Scherzo No.1.png'
peaks, _ = find_peaks(np.abs(yf), height=0.0010)
peak_pitches = {pitch_f(f) for f in xf[peaks[:]] if f > 0}  # {'G5', 'E6', 'G4', 'E5', 'B5', 'C#5', 'B6'}

# Checking the binned results
peaks, _ = find_peaks(yf_binned, height=0.0050)
peak_chroma = {chroma_f(f) for f in xf_binned[peaks[:]] if f > 0}  # {'B', 'G', 'C#', 'E'}

scherzo1_1 = chord.Chord(peak_chroma)
scherzo1_1.commonName  # half-diminished seventh chord

# Figure
# Note that the thresholds were chosen such that other peaks are not detected
plt.plot(xf, np.abs(yf))
plt.xlim(0, 3000)
plt.title('FFT on a real sample of the 1st chord of Chopin Scherzo No.1')
plt.xlabel('Frequency')
plt.ylabel('Amplitude (Cumulative)')
plt.savefig(shortuuid.uuid())
plt.show()

# # We look at another chord to see the problem more clearly
# y, sample_rate = a2n.audio_from_file('scherzo1_2.wav')
# yf = fft(y, norm='forward')
# xf = fftfreq(len(y), 1 / sample_rate)
#
# # Figure
# plt.plot(xf, np.abs(yf))
# plt.xlim(0, 3000)
# plt.title('FFT on a real sample of the 2nd chord of Chopin Scherzo No.1')
# plt.xlabel('Frequency')
# plt.ylabel('Amplitude (Cumulative)')
# plt.show()
#
# xf_binned, yf_binned = chroma_mapping(xf, np.abs(yf))
#
# # Figure
# plt.rcParams['figure.figsize'] = [8, 4.8]
# plt.plot(xf_binned, yf_binned)
# plt.xticks(octave_freqs, chroma_labels)
# plt.title('FFT on a real sample of the 2nd chord of Chopin Scherzo No.1')
# plt.xlabel('Chroma ($A_{4}$ = 440 $Hz$)')
# plt.ylabel('Amplitude (Cumulative)')
# plt.show()

# Let's look at an oboe's tuning 'A'
# https://www.youtube.com/watch?v=t18EgI2Jsmk
y, sample_rate = a2n.audio_from_file('oboe-A4.wav')

t = np.linspace(0, 88306/sample_rate, 88306, endpoint=False)
plt.plot(t, y)
plt.xlabel('Time ($s$)')
plt.ylabel('Amplitude')
plt.savefig(shortuuid.uuid())
plt.show()

yf = fft(y, norm='forward')
xf = fftfreq(len(y), 1 / sample_rate)

xf_binned, yf_binned = chroma_mapping(xf, np.abs(yf))

# Figure
plt.rcParams['figure.figsize'] = [8, 4.8]
plt.plot(xf_binned, yf_binned)
plt.xscale('log')
plt.minorticks_off()
plt.xticks(octave_freqs, chroma_labels)
plt.title("FFT on a real sample of an oboe's tuning 'A'")
plt.xlabel('Chroma ($A_{4}$ = 440 $Hz$)')
plt.ylabel('Amplitude (Cumulative)')
plt.savefig(shortuuid.uuid())
plt.show()
# Interestingly, 'E' gives the highest peak when the amplitudes are added together, which means this is not an ideal
# strategy if we want to correctly identify a note.

# Let's look at the FFT spectrum (with peaks)
peaks, properties = find_peaks(np.abs(yf), height=0.00015)
# Figure
# Note that the x-axis is on a linear scale now
plt.plot(xf, np.abs(yf), '-', xf[peaks], properties['peak_heights'], 'x')
plt.subplots_adjust(bottom=0.15)

# Setting up chroma labels
# octave = chord.Chord([i + (12 * j) for i in [21, 28] for j in range(4, 10)])
# octave_freqs = [i.frequency for i in octave.pitches]
# chroma_labels = [i.nameWithOctave for i in octave.pitches]

freqs = np.split(xf[peaks[:]], 2)[0]
chroma_labels = [pitch_f(f) for f in freqs]
chroma_labels_freqs = [str(np.round(f, 1)) for f in freqs]
unique_chroma_labels_with_freqs = unique_everseen(zip(freqs, chroma_labels, chroma_labels_freqs), key=itemgetter(1))

freqs, chroma_labels, chroma_labels_freqs = zip(*unique_chroma_labels_with_freqs)
chroma_freqs_labels = [f'{i[1]}\n(~{i[0][:-1]}$_{i[0][-1]}$)' for i in zip(chroma_labels, chroma_labels_freqs)]

plt.xticks(freqs, chroma_freqs_labels)
plt.xlim(0, 4900)
plt.title("FFT on a real sample of an oboe's tuning 'A' (height=0.00015)")
plt.xlabel('Frequency')
plt.ylabel('Amplitude (Cumulative)')
plt.savefig(shortuuid.uuid())
plt.show()


# We see the overtones very clearly and also why E stands out in the last FFT plot

# print(pitch_f(3328))
# print(3328/5)

# Now let's try multiplying instead adding
# Same as `chroma_mapping()` except that the amplitudes of each octave are multiplied together instead
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


m = 4  # The number of overtones we consider
hps = copy(y)
for h in range(2, m + 1):
    dec = decimate(y, h)
    hps[:len(dec)] += dec

yf = fft(hps)
yf **= 1 / m
xf = fftfreq(len(y), 1 / sample_rate)

xf_binned, yf_binned = chroma_mapping_m(xf, np.abs(yf))

# yf = fft(y)
# xf = fftfreq(len(y), 1 / sample_rate)
# xf_binned, yf_binned = chroma_mapping_m(xf, np.abs(yf))
# xf_binned, yf_binned = hps_downsample(xf, freqs[0]), np.abs(yf)
# Figure
plt.rcParams['figure.figsize'] = [8, 4.8]
plt.plot(xf_binned, yf_binned)

# Setting up chroma labels
octave = chord.Chord([i for i in range(12)])
octave_freqs = [i.frequency for i in octave.pitches]
chroma_labels = [i.name for i in octave.pitches]

plt.xscale('log')
plt.minorticks_off()
plt.xticks(octave_freqs, chroma_labels)
plt.title("HPS(ish) of FFT on a real sample of an oboe's tuning 'A'")
plt.xlabel('Chroma ($A_{4}$ = 440 $Hz$)')
plt.ylabel('Amplitude (Cumulative)')
plt.savefig(shortuuid.uuid())
plt.show()
# We still see a spike at 'E' but it is smaller than that of 'A'

peaks, _ = find_peaks(yf_binned, height=8e8)
xf_binned[peaks[:]]  # [437.47423731]

# Let's look at a ('E' + 'A') violin double-stop (often used in tuning) sample taken from
# https://www.youtube.com/watch?v=MvJemGmuN5A
y, sample_rate = a2n.audio_from_file('chaconne.wav')

# m is the number of overtones we consider
m = 4

# # M0
# hps = copy(y)
# for h in range(2, m + 1):
#     dec = decimate(y, h)
#     print(len(dec), len(y))
#     hps[:len(dec)] += dec
#
# yf = fft(hps)
# yf **= 1 / m
# xf = fftfreq(len(y), 1 / sample_rate)
#
# xf_binned, yf_binned = chroma_mapping_m(xf, np.abs(yf))

# M1
hps = copy(y)
for h in range(1, m + 1):
    dec = decimate(y, 2 ** h)
    print(len(dec), len(y))
    hps[:len(dec)] += dec

yf = fft(hps)
yf **= 1 / m
xf = fftfreq(len(y), 1 / sample_rate)

xf_binned, yf_binned = chroma_mapping_m(xf, np.abs(yf))

# # M2
# yf = abs(fft(y))
#
# hps = copy(yf)
# for h in range(1, m + 1):
#     dec = decimate(yf, 2 ** h)
#     hps[:len(dec)] += dec
#
# hps **= 1 / m
# xf = fftfreq(len(y), 1 / sample_rate)
#
# xf_binned, yf_binned = chroma_mapping_m(xf, hps)

# Figure
plt.plot(xf_binned, yf_binned)

# Setting up chroma labels
octave = chord.Chord([i for i in range(12)])
octave_freqs = [i.frequency for i in octave.pitches]
chroma_labels = [i.name for i in octave.pitches]

plt.xscale('log')
plt.minorticks_off()
plt.xticks(octave_freqs, chroma_labels)
plt.title("HPS(ish) of FFT on a real sample of Chaconne's 'E+A' Chord")
plt.xlabel('Chroma A_{5}$')
plt.ylabel('Amplitude (Cumulative)')
plt.savefig(shortuuid.uuid())
plt.show()
