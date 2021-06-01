from termcolor import colored
from bitarray import bitarray
from pydub import AudioSegment
import scipy.io.wavfile as wr
import numpy as np


# Bitarray to string
def bits2a(b):
    return ''.join(chr(int(''.join(x), 2)) for x in zip(*[iter(b)]*8))


# Bitarray to bytes
def bitstring_to_bytes(s):
    v = int(s, 2)
    b = bytearray()
    while v:
        b.append(v & 0xff)
        v >>= 8
    return bytes(b[::-1])


# Spline for watermarking
def spline(dot):

    if 1/3 >= dot >= 0:
        return 9 / 2 * (dot ** 2)

    elif 1/3 < dot <= 2/3:
        return -9 * (dot ** 2) + 9 * dot - 3 / 2

    else:
        return 9 / 2 * ((1 - dot) ** 2)


# Watermarking
def watermarking(signal, t_size, spline, mark_in_bits, A):
    g = np.zeros(len(signal))

    for inter in range(len(mark_in_bits)):
        if mark_in_bits[inter] == '0':
            u = 1 - A * spline
        else:
            u = 1 + A * spline

        g[inter * t_size: (inter + 1) * t_size] = list(signal[inter * t_size: (inter + 1) * t_size] * u)

    g[len(mark_in_bits) * t_size: len(signal)] = list(signal[len(mark_in_bits) * t_size: len(signal)])
    return np.int16(g)


# Checking watermark
def check_watermark(marked, mark_length_in_bits, mark, fragment_size, original_signal):

    bits = []
    for position in range(0, mark_length_in_bits):

        power_of_water = 0
        for i in range(position * fragment_size, (position + 1) * fragment_size + 1):
            power_of_water += marked[i] ** 2
        power_of_signal = 0
        for i in range(position * fragment_size, (position + 1) * fragment_size + 1):
            power_of_signal += original_signal[i] ** 2

        if power_of_water > power_of_signal:
            bits.append('1')
        if power_of_water < power_of_signal:
            bits.append('0')

    bits = ''.join(bits)
    recovered = bitstring_to_bytes(bits).decode('utf-8', errors='ignore')
    bar = bitarray()
    bar.fromstring(mark)
    watermark_in_bits = bar.to01()
    print(colored('Equality of the restored and original watermarks?:', 'green'), colored(recovered == mark, 'green'))
    print(colored(recovered, 'green'))
    print(colored(mark, 'green'))
    print(colored('Equality of the restored and original binary watermarks?:', 'blue'))
    print(colored(bits, 'blue'))
    print(colored(watermark_in_bits, 'blue'))
    mistakes = np.sum([bits[i] != watermark_in_bits[i] for i in range(len(bits))])
    print('Number of incorrect bits', mistakes)


def main(interval, A, reduction_factor, path, watermark):

    fs, read = wr.read(path)
    read = np.array(read)
    read = read * reduction_factor

    shape = read.shape
    if len(shape) > 1:
        signal = np.array([read[i][0] for i in range(shape[0])])
    else:
        signal = np.array(read)

    signal = np.int16(signal)
    wr.write('../new_voice.wav', fs, signal)
    fs, signal = wr.read('../new_voice.wav')

    bar = bitarray()
    bar.fromstring(watermark)
    watermark_in_bits = bar.to01()
    length_in_bits = len(watermark_in_bits)

    t_size = int(interval * 0.001 * fs)  # количество амплитуд в интервале
    print('Number of samples in:', interval, 'ms:', t_size)

    t = np.linspace(0, 1, t_size)
    spline_results = np.array([spline(dot) for dot in t])

    marked_signal = watermarking(signal, t_size, spline_results, watermark_in_bits, A)
    wr.write('../marked.wav', fs, marked_signal)

    print('\nWithout attacks')
    fs, g_new = wr.read('../marked.wav')
    check_watermark(g_new, length_in_bits, watermark, t_size, signal)

    print('\nNoise attack')

    noise = np.random.normal(0, 0.01, len(signal))
    g_noised = g_new + noise
    g_noised = np.int16(g_noised)
    check_watermark(g_noised, length_in_bits, watermark, t_size, signal)

    noise = np.random.normal(1, 0.01, len(signal))
    g_noised = g_new + noise
    g_noised = np.int16(g_noised)
    check_watermark(g_noised, length_in_bits, watermark, t_size, signal)

    print('\nAttack by convertation to mp3')
    wav_sound = AudioSegment.from_wav('../marked.wav')
    wav_sound.export('marked.mp3', format='mp3')
    mp3_sound = AudioSegment.from_mp3('../marked.mp3')
    mp3_sound.export('converted_marked.wav', format='wav')
    _, g_converted = wr.read('../converted_marked.wav')
    check_watermark(g_converted, length_in_bits, watermark, t_size, signal)

# Here you can see optimal parameters for correct working with watermarking and checking functions
main(interval=1000,  # time interval in ms
     A=0.5,  # A
     reduction_factor=0.01,
     path='../voice.wav',
     watermark="watermark"
     )
