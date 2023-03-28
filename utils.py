import json
import logging
import os
import warnings
from PIL import Image

logging.disable(logging.WARNING)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

root_path = os.path.dirname(
    os.path.realpath(
        __file__
    )
)

outcome_path = os.path.join(
    root_path,
    "outcome"
)

checkpoint_path = os.path.join(
    outcome_path,
    "checkpoint.pt"
)

release_path = os.path.join(
    outcome_path,
    "release.pt"
)

archive_path = os.path.join(
    outcome_path,
    "archive.json"
)


def load_file(path):
    extension = os.path.splitext(path)[1]

    if extension == ".json":
        with open(
                path,
                "rt"
        ) as stream:
            return json.load(stream)

    elif extension == ".txt":
        with open(
                path,
                "rt"
        ) as stream:
            return stream.read()

    else:
        raise Exception("invalid extension.")


def dump_file(
        buffer,
        path
):
    extension = os.path.splitext(path)[1]

    if extension == ".json":
        with open(
                path,
                "wt"
        ) as stream:
            json.dump(
                buffer,
                stream
            )

    elif extension == ".txt":
        with open(
                path,
                "wt"
        ) as stream:
            stream.write(buffer)

    else:
        raise Exception("invalid extension.")


def get_image(path):
    with Image.open(path) as stream:
        image = stream.copy()
        image.path = path
        return image
