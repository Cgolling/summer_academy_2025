# -*- coding: utf-8 -*-

import librosa
import midiutil
import mido

import sounddevice as sd


def wave_to_midi(audio_data, s_rate) -> mido.MidiFile | midiutil.MIDIFile:
    """
    Converts audio data to MIDI format.
    This is a placeholder function. You need to implement the actual conversion logic.
    :param audio_data: The audio data to convert.
    :param s_rate: The sample rate of the audio data.
    :return: A MIDI object.
    """
    ...


if __name__ == "__main__":
    print("Starting...")
    filename = librosa.ex('trumpet')
    audio_data, s_rate = librosa.load(filename, sr=None)
    print("Audio file loaded!")
    result = wave_to_midi(audio_data, s_rate=s_rate)
    sd.play(audio_data, s_rate)
    sd.wait()

    if isinstance(result, mido.MidiFile):
        result.save("output.mid")  # Save the MIDI file using mido
    elif isinstance(result, midiutil.MIDIFile):
        with open("output.mid", 'wb') as file:
            result.writeFile(file)  # Save the MIDI file using midiutil
    else:
        raise TypeError("The result must be a mido.MidiFile or midiutil.MIDIFile instance.")
    print("Done. Exiting!")
