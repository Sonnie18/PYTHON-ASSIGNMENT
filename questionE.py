import numpy as np
import matplotlib.pyplot as plt


def analyze_signal(F1, F2, duration, sampling_freq):
    # Generate time vector
    t = np.linspace(0, duration, int(duration * sampling_freq), False)

    # Generate the signal
    s = np.sin(2 * np.pi * F1 * t) + np.sin(2 * np.pi * F2 * t)

    # Compute the FFT
    fft_result = np.fft.fft(s)

    # Compute the frequencies
    frequencies = np.fft.fftfreq(len(s), 1/sampling_freq)

    # Plot the FFT
    plt.figure(figsize=(8, 6))
    plt.plot(frequencies, np.abs(fft_result))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Fast Fourier Transform')
    plt.show()

    return frequencies, fft_result


# Example usage
F1 = 50
F2 = 120
duration = 1
sampling_freq = 1000

frequencies, fft_result = analyze_signal(F1, F2, duration, sampling_freq)
