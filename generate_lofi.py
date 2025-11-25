from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=30)
output = model.generate(["lo-fi hip hop with Rhodes piano and vinyl crackle"])
audio_write("lofi_track", output[0].cpu(), model.sample_rate, strategy="loudness")