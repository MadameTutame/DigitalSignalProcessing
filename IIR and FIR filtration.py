import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt


"""
    IIR and FIR filtration
"""


def plot_signals(recovered, signal):

    plt.plot(signal[:200])
    plt.plot(recovered[:200])
    plt.show()

    return


def plot_h(w, h, label):

    plt.plot(w / np.pi, abs(h))
    plt.title(label)
    plt.show()

    return


def plot_spectrum(signal, sampling_freq, label):

    spec_freq, times, spectrum = sg.spectrogram(signal, sampling_freq)
    plt.pcolormesh(times, spec_freq, spectrum, shading='gouraud')
    plt.title(label)
    plt.show()

    return


def serial_connection(frequencies, sampling_freq, signal):

    iir_filter = sg.iirfilter(11, frequencies, btype='bandpass', fs=sampling_freq)
    fir_filter = sg.firwin(11, frequencies, pass_zero='bandstop', fs=sampling_freq)

    rec_signal = sg.filtfilt(iir_filter[0], iir_filter[1], signal)
    rec_signal = sg.filtfilt(fir_filter, 1, rec_signal)
    w_iir, h_iir = sg.freqz(iir_filter[0], iir_filter[1])
    w_fir, h_fir = sg.freqz(fir_filter)

    plot_h(w_iir, h_iir, 'Serial conn. Transmission function IIR')
    plot_h(w_fir, h_fir, 'Serial conn. Transmission function FIR')
    plot_h(w_fir, h_iir * h_fir, 'Serial conn. Final transmission function')

    plot_signals(rec_signal, signal)

    plot_spectrum(rec_signal, sampling_freq, 'Signal spectrum when serial conn.')

    return


def parallel_connection(frequencies, sampling_freq, signal):

    iir_filter = sg.iirfilter(11, frequencies, btype='bandpass', fs=sampling_freq)

    fir_filter1 = sg.firwin(11, frequencies[0], pass_zero=1, fs=sampling_freq)
    fir_filter2 = sg.firwin(11, frequencies[1], pass_zero=0, fs=sampling_freq)

    fir_filter = fir_filter1 + fir_filter2

    rec_signal = sg.filtfilt(iir_filter[0], iir_filter[1], signal)
    rec_signal = sg.filtfilt(fir_filter, 1, rec_signal)
    w_iir, h_iir = sg.freqz(iir_filter[0], iir_filter[1])
    w_fir1, h_fir1 = sg.freqz(fir_filter1)
    w_fir2, h_fir2 = sg.freqz(fir_filter2)
    w_fir, h_fir = sg.freqz(fir_filter)

    plot_h(w_iir, h_iir, '|| conn. Transmission function IIR')
    plot_h(w_fir1, h_fir1, '|| conn. Transmission function FIR №1')
    plot_h(w_fir2, h_fir2, '|| conn. Transmission function FIR №2')
    plot_h(w_fir, h_fir, '|| conn. Transmission function FIR-ов')
    plot_h(w_fir, h_iir*h_fir, '|| conn. Final transmission function')

    plot_signals(rec_signal, signal)

    plot_spectrum(rec_signal, sampling_freq, 'Signal spectrum when || conn.')

    return


'''
    Creation of a signal
'''

fs = 200
signal_freq = [15, 30, 50, 75, 90]

seconds = 3
x = np.linspace(0, seconds, int(seconds * fs))
y = np.zeros(int(seconds * fs))

for freq in signal_freq:
    y = y + np.sin(2 * np.pi * freq * x)

find_freq = [30, 75]

plot_spectrum(y, fs, 'Original signal spectrum')


serial_connection(frequencies=find_freq, sampling_freq=fs, signal=y)    # Filtering by serial connected filters
parallel_connection(frequencies=find_freq, sampling_freq=fs, signal=y)  # Filtering by parallel (||) connected filters

