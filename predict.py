import torch
import numpy as np
import random
from typing import List
from cog import BasePredictor, Input, Path

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write


class Predictor(BasePredictor):
    def setup(self):
        """
        Load MusicGen model once when the container starts.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MusicGen.get_pretrained("facebook/musicgen-small")
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        prompt: str = Input(description="Text description of the music", default="lo-fi hip hop with rain ambience"),
        duration: int = Input(description="Duration in seconds", default=8),
        sample_rate: int = Input(description="Sample rate", default=32000),
        seeds: List[int] = Input(description="List of seeds for variation", default=None),
        album_prefix: str = Input(description="Output filename prefix", default="Album"),
        postprocess: bool = Input(description="Add vinyl crackle effect", default=False),
        output_format: str = Input(description="Audio format", choices=["wav", "mp3", "flac"], default="wav"),
    ) -> List[Path]:
        """
        Generate audio and return Cog Paths (Replicate will upload them).
        """
        results: List[Path] = []

        if not seeds:
            seeds = [None]

        self.model.set_generation_params(duration=duration)

        for seed in seeds:
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

            with torch.no_grad():
                audio = self.model.generate([prompt])
                # audio shape: [B, T] or [B, C, T]
                if audio.ndim == 3:
                    audio = audio[0, 0, :]
                else:
                    audio = audio[0, :]

                if postprocess:
                    crackle = 0.003 * torch.randn_like(audio)
                    audio = (audio + crackle).clamp(-1.0, 1.0)

            out_path = Path(f"{album_prefix}_seed{seed or 'none'}.{output_format}")
            audio_write(out_path, audio.cpu(), sample_rate=sample_rate, strategy="loudness", loudness_compressor=True)
            results.append(out_path)

        return results