import replicate
import time
import requests
import logging

logging.basicConfig(
    filename="album_runs.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

VERSION_HASH = "63da25ea46b6099626653788cccdd498862f488b43449162e7c3d84ef927b279"  # update after push

prediction = replicate.predictions.create(
    version=VERSION_HASH,
    input={
        "prompt": "lo-fi hip hop with rain ambience",
        "duration": 8,  # start small
        "sample_rate": 32000,
        "seeds": [42],
        "album_prefix": "MidnightArchives",
        "postprocess": True,
        "output_format": "wav"
    },
    hardware="gpu-t4"  # force GPU tier
)

print("Prediction ID:", prediction.id)
logging.info(f"Run started | id={prediction.id} | version={VERSION_HASH}")

start_time = time.time()
while prediction.status not in ["succeeded", "failed", "canceled"]:
    elapsed = int(time.time() - start_time)
    print(f"Status: {prediction.status} | Elapsed: {elapsed}s")
    time.sleep(5)
    prediction = replicate.predictions.get(prediction.id)

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