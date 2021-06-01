from bitarray import bitarray
import scipy.io.wavfile as wr
import numpy as np


"""
    This watermarking type looks like amplitude watermarking, 
    but it uses spectrum amplitudes instead of signal amplitudes.
"""


def bits2a(b):
    return ''.join(chr(int(''.join(x), 2)) for x in zip(*[iter(b)]*8))


def bitstring_to_bytes(s):
    v = int(s, 2)
    b = bytearray()
    while v:
        b.append(v & 0xff)
        v >>= 8
    return bytes(b[::-1])


def watermarking(signal, t_size, mark_in_bits, position):

    epsilon = 0.005
    g = np.zeros(len(signal))
    g[:position] = signal[:position]
    for inter in range(len(mark_in_bits)):

        if mark_in_bits[inter] == '0':
            lamb = 1 / (1 + epsilon)
        else:
            lamb = 1 / (1 - epsilon)

        buf = signal[inter * t_size + position: (inter + 1) * t_size + position]
        fourier = np.fft.fft(buf)
        positions = [1, 2, t_size - 2, t_size - 1]
        power_f_before = 0
        power_f_after = 0
        power_not_f_before = 0

        for i in range(len(fourier)):
            if i in positions:
                power_f_before += np.abs(fourier[i]) ** 2
                fourier[i] = fourier[i] * lamb
                power_f_after += np.abs(fourier[i]) ** 2
            else:
                power_not_f_before += np.abs(fourier[i]) ** 2

        delta = power_f_before - power_f_after
        for i in range(len(fourier)):
            if i not in positions:
                fourier[i] = fourier[i] * np.sqrt((power_not_f_before + delta) / power_not_f_before)

        buf = np.fft.ifft(fourier)
        buf = np.array([x.real for x in buf])

        g[inter * t_size + position: (inter + 1) * t_size + position] = buf

    g[len(mark_in_bits) * t_size + position: len(signal)] = signal[len(mark_in_bits) * t_size + position: len(signal)]
    return np.int16(g)


# Checking watermark
def check_watermark(marked, mark_length_in_bits, mark, t_size, original_signal, position):

    bits = []
    for inter in range(0, mark_length_in_bits):

        buf = marked[inter * t_size + position: (inter + 1) * t_size + position]
        fourier_marked = np.fft.fft(buf)
        fourier_original = np.fft.fft(original_signal[inter * t_size + position: (inter + 1) * t_size + position])
        positions = [1, 2, t_size - 2, t_size - 1]

        power_of_water = 0
        power_of_signal = 0
        for i in range(len(fourier_marked)):
            if i in positions:
                power_of_water += np.abs(fourier_marked[i]) ** 2
                power_of_signal += np.abs(fourier_original[i]) ** 2

        if power_of_water > power_of_signal:
            bits.append('1')
        if power_of_water < power_of_signal:
            bits.append('0')

    bits = ''.join(bits)
    recovered = bitstring_to_bytes(bits).decode('utf-8', errors='ignore')
    bar = bitarray()
    bar.fromstring(mark)
    watermark_in_bits = bar.to01()
    print('Recovered and original watermarks are equal:', recovered == mark)
    print(recovered)
    print(mark)
    print('Recovered and original watermarks in bits:')
    print(bits)
    print(watermark_in_bits)
    mistakes = np.sum([bits[i] != watermark_in_bits[i] for i in range(len(bits))])
    print('Mistakes are in', mistakes, 'bits.')


def main(interval, reduction_factor, path, watermark, position):

    fs, read = wr.read(path)
    read = np.array(read)
    read = read * reduction_factor

    shape = read.shape
    if len(shape) > 1:
        signal = np.array([read[i][0] for i in range(shape[0])])
    else:
        signal = np.array(read)

    signal = np.int16(signal)
    wr.write('new_voice.wav', fs, signal)
    fs, signal = wr.read('new_voice.wav')

    signal = np.array(signal, dtype=np.float64)

    bar = bitarray()
    bar.fromstring(watermark)
    watermark_in_bits = bar.to01()
    length_in_bits = len(watermark_in_bits)

    t_size = int(interval * 0.001 * fs)  # количество амплитуд в интервале
    print('Amplitudes in interval', interval, 'мс', t_size)

    marked_signal = watermarking(signal=signal,
                                 t_size=t_size,
                                 mark_in_bits=watermark_in_bits,
                                 position=position
                                 )

    wr.write('marked.wav', fs, marked_signal)

    print('\nWithout attacks')
    fs, g_new = wr.read('marked.wav')
    g_new = np.float64(g_new)
    check_watermark(marked=g_new,
                    mark_length_in_bits=length_in_bits,
                    mark=watermark,
                    t_size=t_size,
                    original_signal=signal,
                    position=position
                    )

    mean = 0
    dsp = 0.01
    print('\nNoise with mean {} and var {}'.format(mean, dsp))
    noise = np.random.normal(mean, dsp, len(signal))
    g_noised = g_new + noise
    g_noised = np.int16(g_noised)
    g_noised = np.float64(g_noised)
    check_watermark(marked=g_noised,
                    mark_length_in_bits=length_in_bits,
                    mark=watermark,
                    t_size=t_size,
                    original_signal=signal,
                    position=position
                    )

    mean = 10
    dsp = 5
    print('\nNoise with mean {} and var {}'.format(mean, dsp))
    noise = np.random.normal(mean, dsp, len(signal))
    g_noised = g_new + noise
    g_noised = np.int16(g_noised)
    g_noised = np.float64(g_noised)
    check_watermark(marked=g_noised,
                    mark_length_in_bits=length_in_bits,
                    mark=watermark,
                    t_size=t_size,
                    original_signal=signal,
                    position=position
                    )


main(interval=10,
     reduction_factor=1,
     path='voice.wav',
     watermark="YOUR MARK",
     position=30000
     )

