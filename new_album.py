import replicate
import time
import requests
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename="album_runs.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

# Replace with your actual version hash
VERSION_HASH = "6164994fd416c4c8c2582272d8696dc15bb83fb547efd06d4fba54c9"

# Start prediction
prediction = replicate.predictions.create(
    version=VERSION_HASH,
    input={
        "generation_prompt": "lo-fi hip hop with rain ambience",
        "duration": 30,
        "sample_rate": 32000,
        "seeds": [1,2,3,4,5,6,7,8,9,10],
        "album_prefix": "MidnightArchives",
        "postprocess": True
    }
)

print("Prediction ID:", prediction.id)
logging.info(f"Run started | id={prediction.id} | version={VERSION_HASH}")

# Poll until finished
start_time = time.time()
while prediction.status not in ["succeeded", "failed", "canceled"]:
    elapsed = int(time.time() - start_time)
    print(f"Status: {prediction.status} | Elapsed: {elapsed}s")
    time.sleep(5)
    prediction = replicate.predictions.get(prediction.id)

# Final status
print("Final status:", prediction.status)
logging.info(f"Run finished | status={prediction.status} | gpu={prediction.hardware}")

if prediction.status == "succeeded":
    print("GPU type:", prediction.hardware)
    for i, track_url in enumerate(prediction.output, start=1):
        filename = f"MidnightArchives_seed{i}.wav"
        response = requests.get(track_url)
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Saved {filename}")
        logging.info(f"Track saved: {filename} | url={track_url}")