import replicate

output = replicate.run(
    "andy-regulore/lofi_yt:latest",
    input={
        "generation_prompt": "lo-fi hip hop with rain ambience",
        "duration": 30,
        "sample_rate": 32000,
        "seeds": [1,2,3,4,5,6,7,8,9,10],
        "album_prefix": "MidnightArchives",
        "postprocess": True
    }
)

for track in output:
    print(track)