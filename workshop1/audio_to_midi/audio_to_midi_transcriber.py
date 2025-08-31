# -*- coding: utf-8 -*-

import librosa
import matplotlib.pyplot as plt
import midiutil
import mido
import numpy as np
import sounddevice as sd
from scipy.ndimage import gaussian_filter1d


def create_spectrogram(audio_data, s_rate):
    # Compute the Short-Time Fourier Transform (STFT) to get a spectrogram
    stft = librosa.stft(audio_data)
    spectrogram = np.abs(stft)

    return spectrogram


def visualise_spectrogram(spectrogram, title="Spectrogram"):
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(
        librosa.amplitude_to_db(spectrogram, ref=np.max),
        y_axis="log",
        x_axis="time",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()


def smooth_spectrogram(spectrogram, sigma=1.0):
    """
    Smooth the spectrogram using Gaussian filter along the time axis.

    :param spectrogram: The input spectrogram array (frequency x time)
    :param sigma: Standard deviation for Gaussian kernel
    :return: Smoothed spectrogram
    """
    # Apply Gaussian filter along axis=1 (time axis) for each frequency bin
    smoothed = gaussian_filter1d(spectrogram, sigma=sigma, axis=1)
    return smoothed


def wave_to_midi(audio_data, s_rate) -> mido.MidiFile | midiutil.MIDIFile:
    """
    Converts audio data to MIDI format.
    This is a placeholder function. You need to implement the actual conversion logic.
    :param audio_data: The audio data to convert.
    :param s_rate: The sample rate of the audio data.
    :return: A MIDI object.
    """

    spectrogram = create_spectrogram(audio_data, s_rate)
    visualise_spectrogram(spectrogram, title="Original Spectrogram")

    # Make smooth
    smoothed_spectrogram = smooth_spectrogram(spectrogram)
    visualise_spectrogram(smoothed_spectrogram, title="Smoothed Spectrogram")

    # Add cutoff threshold
    # Calculate adaptive threshold using local statistics
    window_size = int(s_rate * 0.1)  # 100ms window
    threshold_percentile = 95  # Use 80th percentile as threshold

    # Calculate local threshold for each time frame
    adaptive_threshold = np.zeros(smoothed_spectrogram.shape[1])
    for i in range(smoothed_spectrogram.shape[1]):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(smoothed_spectrogram.shape[1], i + window_size // 2)
        window_data = smoothed_spectrogram[:, start_idx:end_idx]
        adaptive_threshold[i] = np.percentile(window_data, threshold_percentile)

    # Apply adaptive threshold
    thresholded_spectrogram = smoothed_spectrogram.copy()
    for i in range(smoothed_spectrogram.shape[1]):
        thresholded_spectrogram[:, i][
            smoothed_spectrogram[:, i] < adaptive_threshold[i]
        ] = 0

    visualise_spectrogram(thresholded_spectrogram, title="Thresholded Spectrogram")

    plt.show()


if __name__ == "__main__":
    print("Starting...")
    filename = librosa.ex("trumpet")
    audio_data, s_rate = librosa.load(filename, sr=None)
    print("Audio file loaded!")
    result = wave_to_midi(audio_data, s_rate=s_rate)
    # sd.play(audio_data, s_rate)
    # sd.wait()

    if isinstance(result, mido.MidiFile):
        result.save("output.mid")  # Save the MIDI file using mido
    elif isinstance(result, midiutil.MIDIFile):
        with open("output.mid", "wb") as file:
            result.writeFile(file)  # Save the MIDI file using midiutil
    # else:
    # raise TypeError(
    #     "The result must be a mido.MidiFile or midiutil.MIDIFile instance."
    # )

    print("Done. Exiting!")
