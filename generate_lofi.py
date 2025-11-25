import yaml
import csv
import os
from datetime import datetime
from audiocraft.models import MusicGen
import torchaudio
from pydub import AudioSegment
import torch

# 1. Load schema (supports .yaml or .yml)
schema_file = None
for candidate in ["lofi_prompts.yaml", "lofi_prompts.yml"]:
    if os.path.isfile(candidate):
        schema_file = candidate
        break

if schema_file is None:
    raise FileNotFoundError("No schema file found (expected lofi_prompts.yaml or lofi_prompts.yml).")

with open(schema_file, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 2. Create Run ID and output folder
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"outputs/run_{run_id}"
os.makedirs(output_dir, exist_ok=True)

# 3. Load MusicGen model
model = MusicGen.get_pretrained("facebook/musicgen-small")

# 4. Prepare CSV log file in append mode
csv_file = "generation_log.csv"
file_exists = os.path.isfile(csv_file)

with open(csv_file, "a", newline="", encoding="utf-8") as log:
    writer = csv.writer(log)

    # Write header only if file is new
    if not file_exists:
        writer.writerow([
            "Run ID","Run Date","Track ID","Prompt","Album Prefix","BPM","Mood","Instruments","Release Tag",
            "Raw File","Final File","Normalize","Fade Out (ms)","Overlay File","Overlay Volume",
            "Duration (s)","Sample Rate","Seed","Status"
        ])

    # 5. Loop through tracks
    for track in config["tracks"]:
        meta = track["metadata"]
        pp = track["post_processing"]

        # Build base prompt with metadata injection
        base_prompt = f"{track['generation_prompt']} at {meta['bpm']} BPM, mood: {meta['mood']}, instruments: {', '.join(meta['instruments'])}"

        # Handle single seed or multiple seeds
        seeds = track.get("seeds", None)
        if seeds is None:
            seeds = [track.get("seed", None)]

        for seed_value in seeds:
            album_prefix = track.get("album_prefix", "")
            track_id = f"{album_prefix}_track{track['id']}_seed{seed_value}" if seed_value is not None else f"{album_prefix}_track{track['id']}"
            prompt = base_prompt

            print(f"Generating track {track_id} ({meta['release_tag']}): {prompt}")

            raw_file = os.path.join(output_dir, f"{track_id}_raw.wav")
            final_file = os.path.join(output_dir, f"{track_id}_final.wav")

            status = "Success"
            try:
                # Set seed globally (instead of passing to set_generation_params)
                if seed_value is not None:
                    torch.manual_seed(seed_value)

                # Set generation parameters (duration only)
                model.set_generation_params(duration=track.get("duration",30))

                # Generate audio
                wav = model.generate([prompt])
                torchaudio.save(raw_file, wav[0].cpu(), track.get("sample_rate",32000))

                # Post-processing
                audio = AudioSegment.from_wav(raw_file)
                if pp.get("normalize", False):
                    audio = audio.normalize()
                fade_out_ms = pp.get("fade_out", 0)
                if fade_out_ms > 0:
                    audio = audio.fade_out(fade_out_ms)
                overlay = pp.get("overlay")
                if overlay:
                    ambience = AudioSegment.from_wav(overlay["file"]) + overlay["volume_adjust"]
                    audio = audio.overlay(ambience)

                audio.export(final_file, format="wav")
                print(f"Saved {final_file}")

            except Exception as e:
                status = f"Error: {str(e)}"
                print(f"Track {track_id} failed: {status}")

            run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([
                run_id, run_date, track_id, prompt, album_prefix, meta["bpm"], meta["mood"], ";".join(meta["instruments"]),
                meta["release_tag"], raw_file, final_file,
                pp.get("normalize", False), pp.get("fade_out", 0),
                pp["overlay"]["file"] if pp.get("overlay") else "", 
                pp["overlay"]["volume_adjust"] if pp.get("overlay") else "",
                track.get("duration",30), track.get("sample_rate",32000),
                seed_value if seed_value is not None else "", status
            ])