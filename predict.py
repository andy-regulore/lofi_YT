import torch
import librosa
import soundfile as sf
from transformers import pipeline

class Predictor:
    def setup(self):
        # Load a pretrained model once at startup
        self.generator = pipeline("text-to-audio", model="facebook/musicgen-small")

    def predict(self, prompt: str, duration: int = 30) -> str:
        """
        Generate a lo-fi track from a text prompt.
        Args:
            prompt (str): Description of the track (e.g. "lo-fi beats with rain ambience")
            duration (int): Length of audio in seconds
        Returns:
            str: Path to generated audio file
        """
        output = self.generator(prompt, forward_params={"duration": duration})
        audio = output[0]["audio"]
        sr = output[0]["sampling_rate"]

        filename = "output.wav"
        sf.write(filename, audio, sr)
        return filename