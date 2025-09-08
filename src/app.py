import sys


def main() -> int:
    #arg = sys.argv[1]
    #urls = read_urls(arg)
    urls = sys.argv[1] #get rid of later
    if check_sites(urls):
        print(urls)
    else:
        print("The string does NOT contain 'github.com' or 'huggingface.co'.")


def read_urls(arg: str) -> list[str]:
    try:
        with open(arg, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: File '{arg}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{arg}': {e}")
        sys.exit(1)


def check_sites(input_string: str) -> bool:
    return "github.com" in input_string or "huggingface.co" in input_string


if __name__ == "__main__":
    main()
