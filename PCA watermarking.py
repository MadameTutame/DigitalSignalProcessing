import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.io.wavfile import read, write
import scipy.signal as sg


"""
    PCA watermarking with IIR filter stabilizing.
    Also I've checked some types of attacks on the signal.
"""


def plot_h(b_coeffs, a_coeffs, label):

    w_iir, h_iir = sg.freqz(b_coeffs, a_coeffs)
    plt.plot(w_iir / np.pi, abs(h_iir))
    plt.title(label)
    plt.show()
    return


def plot_signal_and_spectrum(signal, sampling_freq, label):

    plt.plot(signal)
    plt.title('Amplitudes ' + label)
    plt.show()
    plt.specgram(signal, Fs=sampling_freq)
    plt.title('Spectrum ' + label)
    plt.show()


def plot_correlation(signal, watermark, label):

    h = np.correlate(signal, watermark, mode='valid')
    print('Watermarking start position ' + label + ':', h.argmax())
    plt.plot(h)
    plt.title('Correlation graph ' + label)
    plt.show()


fs, x = read('your_voice.wav')

iir_filter = sg.iirfilter(12, [4000], btype='high',  fs=fs)
b, a = iir_filter[0], iir_filter[1]

w_iir, h_iir = sg.freqz(b, a)
plot_h(b_coeffs=b, a_coeffs=a, label="Non stabilized filter's transmission function")


'''
    Stabilizing
'''
new_roots = []
roots = np.poly1d(b).roots
for root in roots:
    a = root.real
    t = root.imag
    a_ = a / 10
    t_ = t / 10
    new_roots.append(complex(a_, t_))
b = np.poly1d(new_roots, True).coef


plot_signal_and_spectrum(signal=x, sampling_freq=fs, label='of original signal')
plot_h(b_coeffs=b, a_coeffs=a, label='Transmission function of stab. filter')
plot_h(b_coeffs=a, a_coeffs=b, label='Transmission function of reverse filter')

"""
    Filtering by IIR-filter
"""

x = sg.filtfilt(b, a, x)

plot_signal_and_spectrum(signal=x, sampling_freq=fs, label='of filtered signal')

"""
    Creating watermark by using PCA
"""

win_size = 0.01
win_size = int(win_size * fs)

R = 100

Matr = np.zeros((win_size, win_size))
for m in range(0, len(x) - win_size, R):
    vector = np.array(x[m: m + win_size])
    vector = np.reshape(vector, (-1, 1))
    Matr += vector @ vector.T

w, v = np.linalg.eig(Matr)
min_w_ind = np.argmin(w)
PCA = v[:, min_w_ind]

"""
    Marking
"""

P = 50000
coefficient = 400

for m in range(len(PCA)):
    x[P + m] = x[P + m] + coefficient * PCA[m].real

marked = sg.filtfilt(a, b, x)
marked = np.int16(marked)

write('marked.wav', fs, marked)
fs, marked = read('marked.wav')


"""
    Checking watermark
"""


plot_correlation(signal=marked, watermark=PCA, label='of unfiltered signal with watermark')

marked_filtered = sg.filtfilt(b, a, marked)

plot_correlation(signal=marked_filtered, watermark=PCA, label='of filtered signal with watermark')


"""
    Attacks with noise and conversion
"""

noise = np.random.normal(0, 2, len(marked))
noised_marked = marked + noise
noised_marked = sg.filtfilt(b, a, noised_marked)
plot_correlation(signal=noised_marked, watermark=PCA, label='with noise')

wav_sound = AudioSegment.from_wav('marked.wav')
wav_sound.export('marked.mp3', format='mp3')
mp3_sound = AudioSegment.from_mp3('marked.mp3')
mp3_sound.export('converted_marked.wav', format='wav')
_, g_converted = read('converted_marked.wav')
s = sg.filtfilt(b, a, g_converted)
plot_correlation(signal=s, watermark=PCA, label='with conversion')
