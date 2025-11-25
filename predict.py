import torch
import random
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List

class Predictor:
    def setup(self):
        """
        Called once when the model is loaded.
        Load heavy models here.
        """
        # Example: load MusicGen from Hugging Face
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        self.model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

    def predict(
        self,
        generation_prompt: str,
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

            # Run inference
            inputs = self.processor(
                text=generation_prompt,
                padding=True,
                return_tensors="pt"
            )
            audio_values = self.model.generate(**inputs, max_length=duration * sample_rate)

            # Convert to numpy
            audio = audio_values.cpu().numpy().astype(np.float32)

            # Optional postprocess (vinyl crackle)
            if postprocess:
                crackle = 0.005 * np.random.randn(duration * sample_rate).astype(np.float32)
                audio = audio + crackle

            filename = f"outputs/{album_prefix}_seed{seed or 'none'}.wav"
            sf.write(filename, audio, sample_rate)
            results.append(filename)

        return results