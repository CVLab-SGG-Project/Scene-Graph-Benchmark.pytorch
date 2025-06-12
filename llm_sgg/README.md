# LLM-based Scene Graph Generation

This module provides a minimal interface for generating scene graphs using
[LiteLLM](https://github.com/BerriAI/litellm) and GPT models.

```
from llm_sgg.lite_llm_sgg import run

scene_graph = run("path/to/image.png")
print(scene_graph)
```

`run` loads the image, sends it to the specified GPT model, and returns the
model's response. The expected output is a JSON-formatted scene graph.

Note that you must install `litellm` and configure API credentials for the
desired LLM provider.
