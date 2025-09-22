import sys


def main() -> int:
<<<<<<< HEAD
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
=======
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
>>>>>>> 758b31f7bc2741ef8b5d2837f23d1c110c7701bb
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: File '{arg}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{arg}': {e}")
        sys.exit(1)


<<<<<<< HEAD
def check_sites(input_string: str) -> bool:
    return "github.com" in input_string or "huggingface.co" in input_string


=======
>>>>>>> 758b31f7bc2741ef8b5d2837f23d1c110c7701bb
if __name__ == "__main__":
    main()
