# utils/audio_transcribe.py
import soundfile as sf
import io
from transformers import pipeline
from scipy.signal import resample
import warnings

def audio_transcribe(audio_file: io.BytesIO) -> str:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Use soundfile to read the buffer and get an array/sample rate
            audio_data, sampling_rate = sf.read(audio_file)
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]

            # resampling audio file for Whisper model
            if sampling_rate != 16000:
                audio_data = resample(audio_data, int(len(audio_data * 16000 / sampling_rate)))
                sampling_rate = 16000
            
            # whisper model setup
            audio_pipeline = pipeline(
                task = "automatic-speech-recognition",
                model = "openai/whisper-small",
                device = 0,
                chunk_length_s = 60,
                batch_size = 100
            )

            res = audio_pipeline(inputs = {"array": audio_data, "sampling_rate" : sampling_rate})

        # returning the transcription
        return res['text']
    except Exception as e:
        return None