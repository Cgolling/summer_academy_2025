# -*- coding: utf-8 -*-

import math

import librosa
import matplotlib.pyplot as plt
import midiutil
import mido
import numpy as np
import sounddevice as sd
from midiutil.MidiFile import MIDIFile
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


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


def find_prominent_frequencies(
    spectrogram, s_rate, height=None, distance=None, prominence=None
):
    """
    Find prominent frequencies in each time window of the spectrogram using peak detection.

    :param spectrogram: The input spectrogram array (frequency x time)
    :param s_rate: Sample rate of the audio
    :param height: Minimum height of peaks (if None, uses adaptive threshold)
    :param distance: Minimum distance between peaks in frequency bins
    :param prominence: Minimum prominence of peaks
    :return: List of arrays, each containing prominent frequency indices for each time frame
    """
    # Get frequency bins from librosa
    n_fft = (spectrogram.shape[0] - 1) * 2  # Reconstruct n_fft from spectrogram shape
    freqs = librosa.fft_frequencies(sr=s_rate, n_fft=n_fft)

    prominent_freqs_per_time = []

    # Process each time frame
    for time_idx in range(spectrogram.shape[1]):
        magnitude_spectrum = spectrogram[:, time_idx]

        # Set default parameters if not provided
        if height is None:
            # Use adaptive height based on local statistics
            adaptive_height = np.percentile(magnitude_spectrum, 75)
        else:
            adaptive_height = height

        if distance is None:
            # Default distance to prevent too many close peaks
            distance = max(1, len(magnitude_spectrum) // 50)

        if prominence is None:
            # Default prominence
            prominence = adaptive_height * 0.1

        # Find peaks in the magnitude spectrum
        peaks, properties = find_peaks(
            magnitude_spectrum,
            height=adaptive_height,
            distance=distance,
            prominence=prominence,
        )

        # Convert frequency bin indices to actual frequencies
        prominent_frequencies = freqs[peaks]

        prominent_freqs_per_time.append(prominent_frequencies)

    return prominent_freqs_per_time


def visualise_found_freqs(prominent_freqs, s_rate):
    # Create a plot for prominent frequencies over time
    plt.figure(figsize=(12, 8))

    # Get time frames
    time_frames = np.arange(len(prominent_freqs))
    hop_length = 512  # Default hop length for librosa.stft
    time_seconds = librosa.frames_to_time(time_frames, sr=s_rate, hop_length=hop_length)

    # Plot each prominent frequency as a scatter plot
    for i, freqs in enumerate(prominent_freqs):
        if len(freqs) > 0:
            plt.scatter(
                [time_seconds[i]] * len(freqs), freqs, c="blue", alpha=0.6, s=10
            )

    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Prominent Frequencies Over Time")
    plt.ylim(0, 4000)  # Limit y-axis to reasonable frequency range
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_fundamental_freqs(fundamental_freqs):
    plt.figure(figsize=(12, 4))
    plt.plot(fundamental_freqs)

    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Fundamental Frequency Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


def wave_to_midi(audio_data, s_rate) -> mido.MidiFile | midiutil.MIDIFile:
    """
    Converts audio data to MIDI format.
    This is a placeholder function. You need to implement the actual conversion logic.
    :param audio_data: The audio data to convert.
    :param s_rate: The sample rate of the audio data.
    :return: A MIDI object.
    """

    spectrogram = create_spectrogram(audio_data, s_rate)
    # visualise_spectrogram(spectrogram, title="Original Spectrogram")

    # Make smooth
    smoothed_spectrogram = smooth_spectrogram(spectrogram)
    # visualise_spectrogram(smoothed_spectrogram, title="Smoothed Spectrogram")

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

    # visualise_spectrogram(thresholded_spectrogram, title="Thresholded Spectrogram")

    prominent_freqs = find_prominent_frequencies(thresholded_spectrogram, s_rate)
    # visualise_found_freqs(prominent_freqs, s_rate)

    fundamental_freqs = [freqs[0] for freqs in prominent_freqs if len(freqs) > 0]
    # plot_fundamental_freqs(fundamental_freqs)

    A4_frequency = 440.0  # = 12th_sqrt(2)
    raw_notes = [
        math.log(freq / A4_frequency) / math.log(2 ** (1 / 12))
        for freq in fundamental_freqs
    ]
    print(raw_notes)

    rounded_notes = [round(note) for note in raw_notes]
    print(rounded_notes)

    midi_pitches = [int(note + 69) for note in rounded_notes]

    plt.show()

    # degrees = [60, 62, 64, 65, 67, 69, 71, 72]  # MIDI note number
    track = 0
    channel = 0
    time = 0  # In beats
    duration = 1  # In beats
    tempo = 60  # In BPM
    volume = 100  # 0-127, as per the MIDI standard

    MyMIDI = MIDIFile(1)  # One track, defaults to format 1 (tempo track
    # automatically created)
    MyMIDI.addTempo(track, time, tempo)

    for pitch in midi_pitches:
        MyMIDI.addNote(track, channel, pitch, time, duration, volume)
        time = time + 1

    with open("v0.mid", "wb") as output_file:
        MyMIDI.writeFile(output_file)


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
