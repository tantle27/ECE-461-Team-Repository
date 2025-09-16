import sys


def main() -> int:
    arg = sys.argv[1]
    urls = read_urls(arg)

    # Pass to api calls
    for url in urls:
        print(f"Processing URL: {url}")
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
