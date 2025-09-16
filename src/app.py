import sys
import api.hf_client as hf_client
import url_router


def main() -> int:
    arg = sys.argv[1]
    urls = read_urls(arg)

    hfClient = hf_client.HFClient()
    urlRouter = url_router.UrlRouter()

    # Pass to api calls
    for url in urls:
        modelId = urlRouter.parse(url).hf_id
        print(f"Processing model id: {modelId}")
        modelInfo = hfClient.get_model_info(modelId)

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
