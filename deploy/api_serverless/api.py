"""AWS Lambda function serving remixer predictions."""
import json

import torch

from core.utils import PreprocessingConfig, serve_request, download_url, chain_functions, TrainingConfig

# model = VAE.load('mickjagger19/Remixer/vae:v1')
pipeline = VAE().to(device=torch.device("cuda"))


def handler(event, _context):
    """Provide main prediction API."""
    config = PreprocessingConfig()
    event = _from_string(event)
    event = _from_string(event.get("body", event))
    genre = _load_genre(event)

    print("INFO loading params complete")

    # if image is None:
    #     return {"statusCode": 400, "message": "neither image_url nor image found in event"}
    content = _load_audio(event, "content", config)
    style = _load_audio(event, "style", config)

    print("INFO loading audios complete")

    output, audio, img = serve_request(TrainingConfig(), genre, content, style, pipeline=pipeline)

    print("INFO inference complete")
    return {"audio": audio}


# event:
# genre: str
# content:  url
# original: url


def _load_genre(event):
    if "genre" not in event:
        return None
    return event["genre"]


def _load_audio(event, type: str, config: PreprocessingConfig):
    if type not in event:
        return None
    audio_url = event[type]
    path = download_url(audio_url)

    pipelines = [load, chroma]
    audio = chain_functions(pipelines, path, config)
    return audio


def _from_string(event):
    if isinstance(event, str):
        return json.loads(event)
    else:
        return event


if __name__ == "__main__":
    handler({"genre": "jazz"}, None)
