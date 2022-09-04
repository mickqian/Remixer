import logging
import os
from typing import Callable
from typing import Tuple

import gradio as gr
import numpy as np
from dotenv import load_dotenv

import constants
from constants import ROOT_DIR, GENRES
from core import utils
from core.models.VAE import build_pipeline, PipelineOrPaths
from core.utils import TrainingConfig

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # do not use GPU

logging.basicConfig(level=logging.INFO)

load_dotenv()  # load environment variables from a .env file if it exists
APP_DIR = ROOT_DIR / "frontend"
ASSETS_DIR = APP_DIR / "assets"
FAVICON = ASSETS_DIR / "logo.jpeg"
FAVICON_STR = FAVICON.__str__()
README = APP_DIR / "README.md"

DEFAULT_PORT = 11700

CALLBACK_TYPE = Callable[[str, Tuple[int, np.array]], Tuple[int, np.array]]


def make_frontend(fn: CALLBACK_TYPE, flagging: bool = False):
    """Creates a gradio.Interface frontend for an image + text to text function."""

    # examples = [[str(img_path), question] for img_path, question in zip(img_example_paths, questions)]
    examples = None

    allow_flagging = "never"
    if flagging:  # logging user feedback to a local CSV file
        allow_flagging = "manual"
        flagging_callback = gr.CSVLogger()
        flagging_dir = "flagged"
    else:
        flagging_callback, flagging_dir = None, None

    readme = _load_readme(with_logging=allow_flagging == "manual")

    slogan = get_heading_from_markdown(readme, "Slogan")

    frontend = gr.Interface(
        fn=fn,  # which Python function are we interacting with?
        outputs=gr.components.Audio(),
        inputs=[
            gr.components.Dropdown(choices=GENRES, value='blues', label='Genre', type="value"),
            gr.components.Audio(label="Content"),
            gr.components.Audio(label="Style"),
        ],
        title="Remixer",
        thumbnail=FAVICON_STR,
        description=slogan,
        article=readme,
        examples=examples,
        cache_examples=False,
        allow_flagging=allow_flagging,
        flagging_options=["low-quality", "genre mismatch", "inaudible"],
        flagging_callback=flagging_callback,
        flagging_dir=flagging_dir,
    )

    return frontend


def _log_inference(pred, metrics):
    for key, value in metrics.items():
        logging.info(f"METRIC {key} {value}")
    logging.info(f"PRED >begin\n{pred}\nPRED >end")


class PredictorBackend:
    """Interface to a backend that serves predictions.

    To communicate with a backend accessible via a URL, provide the url kwarg.

    Otherwise, runs a predictor locally.
    """

    def __init__(self, use_url=None, pip=PipelineOrPaths):
        if use_url:
            # URL of a backend to which to send image data
            self.url = os.getenv("BACKEND_URL")
        if not self.url:
            if isinstance(pip, Tuple):
                self.model = build_pipeline(pip[0], pip[1], utils.TrainingConfig())
            else:
                raise Exception("Please provide valid paths to model dicts.")

        self._predict = self._predict_from_endpoint

    def run(self, genre: str, content: Tuple[int, np.array], style: Tuple[int, np.array]):
        if self.url:
            return self._predict_from_endpoint(genre, content, style)

        # local inference
        output, audio, img = utils.serve_request(TrainingConfig(), genre, content, style)

        return audio

    def _predict_from_endpoint(self, genre: str, content: Tuple[int, np.array], style: Tuple[int, np.array]):
        """Send parameters to an endpoint that accepts JSON and return the audio.
        """
        import json, requests
        headers = {'Content-Type': "application/json"}
        payload = json.dumps({"genre": genre,
                              "content": content,
                              "style": style,
                              })
        response = requests.post(self.url, data=payload, headers=headers)
        print(f"{response=}")
        print(f"{response.content=}")
        response_json = response.json()
        audio = response_json["audio"]
        return audio


def get_heading_from_markdown(markdown_text, keyword=''):
    import re
    # This regex will match headings from level 1 to 6 that exactly contain 'AA'
    heading_regex = re.compile(fr'^(#{1, 6}) {keyword}$', re.MULTILINE)

    # Search for the pattern in the markdown text
    match = heading_regex.search(markdown_text)
    if match:
        # The full match is the heading we're looking for
        return match.group(0)
    else:
        return None


def _load_readme(with_logging=False):
    return open(README).read()


def deploy(pipeline: PipelineOrPaths):
    os.environ['BACKEND_URL'] = "http://localhost:9000/2015-03-31/functions/function/invocations"
    serverless_backend = PredictorBackend(use_url=True, pip=pipeline)
    frontend = make_frontend(serverless_backend.run, flagging=True)
    frontend.launch(
        share=True,
        ssl_verify=False,
        # share_api=True,
        server_name="0.0.0.0",
        favicon_path=FAVICON_STR,
    )


if __name__ == "__main__":
    constants.speedup()
    deploy(None)
