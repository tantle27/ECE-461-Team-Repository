import sys
from api.hf_client import HFClient
from url_router import UrlRouter, UrlType
from handlers import (
    build_model_context, 
    build_dataset_context, 
    build_code_context
)
from metric_eval import MetricEval


def main() -> int:
    arg = sys.argv[1]
    urls = read_urls(arg)

    hf_client = HFClient()
    url_router = UrlRouter()

    # Pass to api calls
    for url in urls:
        parsed = url_router.parse(url)

        if parsed.type == UrlType.MODEL:
            repo_ctx = build_model_context(url)
            print(repo_ctx)

        else:
            print(f"{url} is not a model URL, skipping.")

    return 0


# Helper function to read URLs from a file
def read_urls(arg: str) -> list[str]:
    try:
        with open(arg, "r", encoding="ascii") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: File '{arg}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{arg}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
