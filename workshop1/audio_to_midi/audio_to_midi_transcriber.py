# -*- coding: utf-8 -*-

import librosa
import midiutil
import mido
import numpy as np
import sounddevice as sd


def create_spectrogram(audio_data, s_rate):
    # Compute the Short-Time Fourier Transform (STFT) to get a spectrogram
    stft = librosa.stft(audio_data)
    spectrogram = np.abs(stft)

    return spectrogram


def visualise_spectrogram(spectrogram):
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(
        librosa.amplitude_to_db(spectrogram, ref=np.max),
        y_axis="log",
        x_axis="time",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.tight_layout()
    plt.show()


def wave_to_midi(audio_data, s_rate) -> mido.MidiFile | midiutil.MIDIFile:
    """
    Converts audio data to MIDI format.
    This is a placeholder function. You need to implement the actual conversion logic.
    :param audio_data: The audio data to convert.
    :param s_rate: The sample rate of the audio data.
    :return: A MIDI object.
    """

    spectrogram = create_spectrogram(audio_data, s_rate)
    visualise_spectrogram(spectrogram)


if __name__ == "__main__":
    print("Starting...")
    filename = librosa.ex("trumpet")
    audio_data, s_rate = librosa.load(filename, sr=None)
    print("Audio file loaded!")
    result = wave_to_midi(audio_data, s_rate=s_rate)
    sd.play(audio_data, s_rate)
    sd.wait()

    if isinstance(result, mido.MidiFile):
        result.save("output.mid")  # Save the MIDI file using mido
    elif isinstance(result, midiutil.MIDIFile):
        with open("output.mid", "wb") as file:
            result.writeFile(file)  # Save the MIDI file using midiutil
    else:
        raise TypeError(
            "The result must be a mido.MidiFile or midiutil.MIDIFile instance."
        )

    print("Done. Exiting!")
