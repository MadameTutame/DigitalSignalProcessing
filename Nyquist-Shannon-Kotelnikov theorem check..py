import numpy as np
import matplotlib.pyplot as plt


"""
    Nyquist-Shannon-Kotelnikov theorem check.
"""


seconds = 1
signal_freq = 10
sample_freq = 100

original_deltas = np.linspace(0, seconds, seconds * sample_freq * signal_freq)
original_signal = np.sin(2 * np.pi * signal_freq * original_deltas)

wrong_freq = 13
right_freq = 45

wrong_deltas = np.linspace(0, seconds, seconds * wrong_freq)
wrong_signal = np.sin(2 * np.pi * signal_freq * wrong_deltas)

right_deltas = np.linspace(0, seconds, seconds * right_freq)
right_signal = np.sin(2 * np.pi * signal_freq * right_deltas)


def recovery_by_series(signal, freq):

    new_signal = np.zeros(len(original_deltas))
    counts = np.zeros(len(signal))
    for n in range(len(signal)):
        new_signal += signal[n] * np.sinc((original_deltas - n / freq) * freq)
        counts[n] += n / freq

    plt.plot(original_deltas, original_signal, label='Original')
    plt.plot(original_deltas, new_signal, label='' + str(freq) + ' Hz.')
    plt.title('Recovery_by_series')
    plt.legend(loc="lower right")
    plt.show()


def recovery_by_linspace(signal, freq, deltas):

    new_signal = np.zeros(len(original_deltas))
    for n in range(len(signal)):
        new_signal += signal[n] * np.sinc((original_deltas - deltas[n]) * freq)

    plt.plot(original_deltas, original_signal, label='Original')
    plt.plot(original_deltas, new_signal, label='' + str(freq) + ' Hz.')
    plt.title('Recovery by linspace')
    plt.legend(loc="lower right")
    plt.show()


recovery_by_series(wrong_signal, wrong_freq)
recovery_by_series(right_signal, right_freq)

recovery_by_linspace(wrong_signal, wrong_freq, wrong_deltas)
recovery_by_linspace(right_signal, right_freq, right_deltas)
