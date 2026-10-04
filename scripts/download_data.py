"""Download the UCI Heart Disease archive and extract the Cleveland file."""
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ARCHIVE = RAW / "heart+disease.zip"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    with urlopen(URL, timeout=60) as response:
        ARCHIVE.write_bytes(response.read())
    with ZipFile(ARCHIVE) as archive:
        archive.extractall(RAW)
    print(f"Extracted UCI files to {RAW}")
    print(f"Cleveland dataset: {RAW / 'processed.cleveland.data'}")


if __name__ == "__main__":
    main()
