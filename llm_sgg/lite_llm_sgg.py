import base64
from io import BytesIO
from typing import Any, Dict

from PIL import Image

try:
    import litellm
except ImportError:
    litellm = None


def encode_image(image: Image.Image) -> str:
    """Encode PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def build_prompt(image_b64: str) -> str:
    """Return a prompt asking the LLM for a scene graph."""
    return (
        "Given the following image encoded in base64, "
        "describe the scene graph in JSON format.\n"
        f"image:\n{image_b64}"
    )


def query_scene_graph(image: Image.Image, model: str = "gpt-3.5-turbo") -> str:
    """Use LiteLLM to request a scene graph from the specified model."""
    if litellm is None:
        raise ImportError("litellm is required to use query_scene_graph")

    prompt = build_prompt(encode_image(image))
    response = litellm.completion(model=model, messages=[{"role": "user", "content": prompt}])
    return response["choices"][0]["message"]["content"]


def load_image(path: str) -> Image.Image:
    """Load an image from disk."""
    return Image.open(path).convert("RGB")


def run(path: str, model: str = "gpt-3.5-turbo") -> str:
    """High level helper to load an image and return its scene graph."""
    image = load_image(path)
    return query_scene_graph(image, model)


__all__ = [
    "encode_image",
    "build_prompt",
    "query_scene_graph",
    "load_image",
    "run",
]
