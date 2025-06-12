import argparse

from maskrcnn_benchmark.llm import LLMSceneGraphGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scene graph using GPT")
    parser.add_argument("image", help="Path to an image file")
    parser.add_argument("--api-key", required=True, dest="api_key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o", help="Model name for generation")
    parser.add_argument("--prompt", default=None, help="Custom prompt text")
    args = parser.parse_args()

    generator = LLMSceneGraphGenerator(api_key=args.api_key, model=args.model)
    result = generator.generate(args.image, prompt=args.prompt)
    print(result)


if __name__ == "__main__":
    main()
