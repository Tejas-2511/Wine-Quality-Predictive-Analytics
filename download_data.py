"""
download_data.py - download the official UCI Wine Quality CSV files into data/
and verify them with SHA-256 checksums.

    python download_data.py

The project already ships with verified copies in data/, so you only need this
if those files are missing or you want to re-download them yourself.
"""
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
UCI_ZIP = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"

# SHA-256 of the official files (checked against archive.ics.uci.edu on 27 Oct 2026)
CHECKSUMS = {
    "winequality-red.csv": "4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e",
    "winequality-white.csv": "76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836",
}

# Optional time-series extension dataset (Hyndman TSDL `wineind`, via Rdatasets)
WINEIND_URL = ("https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/"
               "master/csv/forecast/wineind.csv")


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    DATA_DIR.mkdir(exist_ok=True)
    print("Downloading", UCI_ZIP)
    with urllib.request.urlopen(UCI_ZIP, timeout=60) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    for name, expected in CHECKSUMS.items():
        content = z.read(name)
        status = "OK" if sha256(content) == expected else "CHECKSUM MISMATCH"
        (DATA_DIR / name).write_bytes(content)
        print(f"  {name}: {len(content):,} bytes  [{status}]")

    ext = DATA_DIR / "extension"
    ext.mkdir(exist_ok=True)
    try:
        with urllib.request.urlopen(WINEIND_URL, timeout=60) as r:
            (ext / "australian_wine_sales_wineind.csv").write_bytes(r.read())
        print("  australian_wine_sales_wineind.csv: downloaded (optional extension)")
    except Exception as e:  # optional - the main project does not need it
        print("  Could not download optional time-series data:", e)


if __name__ == "__main__":
    main()
