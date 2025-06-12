import base64
from typing import Optional

import litellm


class LLMSceneGraphGenerator:
    """Generate scene graphs using a language model."""

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        self.api_key = api_key
        self.model = model

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def generate(self, image_path: str, prompt: Optional[str] = None) -> str:
        """Return the scene graph description given an image."""
        if prompt is None:
            prompt = (
                "Generate a scene graph for this image as JSON with 'objects' and 'relations'."
            )
        b64 = self._encode_image(image_path)
        messages = [
            {"role": "system", "content": "You are a scene graph generator."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64}"},
                ],
            },
        ]
        response = litellm.completion(model=self.model, messages=messages, api_key=self.api_key)
        return response["choices"][0]["message"]["content"]
