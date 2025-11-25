import torch
import soundfile as sf
import numpy as np
from pathlib import Path
import random
from typing import List

class Predictor:
    def setup(self):
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        self.model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        prompt: str = "lo-fi hip hop with rain ambience",
        duration: int = 8,
        sample_rate: int = 32000,
        seeds: List[int] = None,
        album_prefix: str = "Album",
        postprocess: bool = False,
        output_format: str = "wav"
    ) -> List[str]:
        Path("outputs").mkdir(exist_ok=True)
        results = []

        if not seeds:
            seeds = [None]

        for seed in seeds:
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

            inputs = self.processor(
                text=[prompt],
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                audio_values = self.model.generate_audio(**inputs)

            audio = audio_values[0].cpu().numpy().astype(np.float32)

            if postprocess:
                crackle = 0.003 * np.random.randn(len(audio)).astype(np.float32)
                audio = audio + crackle

            filename = f"outputs/{album_prefix}_seed{seed or 'none'}.{output_format}"
            sf.write(filename, audio, sample_rate)
            results.append(filename)

        return results