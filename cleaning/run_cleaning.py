from pathlib import Path
from clean_text import clean_text


RAW_DIR = Path("data/sample_corpus")
CLEANED_DIR = Path("data/cleaned")


def main():
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    supported_extensions = {".txt", ".md", ".html"}

    for file in RAW_DIR.iterdir():
        if file.is_file() and file.suffix.lower() in supported_extensions:
            raw_text = file.read_text(encoding="utf-8")

            cleaned_text = clean_text(raw_text)

            output_file = CLEANED_DIR / file.name
            output_file.write_text(cleaned_text, encoding="utf-8")

            print(f"Cleaned: {file.name}")


if __name__ == "__main__":
    main()