import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List
import random

class Predictor:
    def setup(self):
        """
        Called once when the container starts.
        Load the MusicGen model here.
        """
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        self.model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

    def predict(
        self,
        generation_prompt: str = "lo-fi hip hop with rain ambience",
        duration: int = 30,
        sample_rate: int = 32000,
        seeds: List[int] = None,
        album_prefix: str = "Album",
        postprocess: bool = False
    ) -> List[str]:
        """
        Generate audio tracks and return file paths.
        """
        Path("outputs").mkdir(exist_ok=True)
        results = []

        if seeds is None:
            seeds = [None]

        for seed in seeds:
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

            # Run inference with MusicGen
            inputs = self.processor(
                text=[generation_prompt],
                padding=True,
                return_tensors="pt"
            )
            # Use generate_audio safely (no oversized max_new_tokens)
            audio_values = self.model.generate_audio(**inputs)

            # Convert tensor to numpy waveform
            audio = audio_values[0].cpu().numpy().astype(np.float32)

            if postprocess:
                crackle = 0.005 * np.random.randn(len(audio)).astype(np.float32)
                audio = audio + crackle

            filename = f"outputs/{album_prefix}_seed{seed or 'none'}.wav"
            sf.write(filename, audio, sample_rate)
            results.append(filename)

        return results