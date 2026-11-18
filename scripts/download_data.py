#!/usr/bin/env python3
"""Download the six published FOF-CT core files used by this project."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path


PORTAL = "https://data.4dnucleome.org"
ACCESSIONS = (
    "4DNFIZWUQTXF",
    "4DNFIKVZTN9K",
    "4DNFID7HZ8LX",
    "4DNFIJYFR8FT",
    "4DNFIIOVHG1I",
    "4DNFI4VLY3WU",
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "hafner2023"
USER_AGENT = "interpretable-chromatin-gnn/0.1 (public research-data downloader)"


def request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def resolve_download_url(accession: str) -> str:
    metadata_url = f"{PORTAL}/{accession}/?format=json"
    with urllib.request.urlopen(request(metadata_url), timeout=60) as response:
        metadata = json.load(response)
    href = metadata.get("href")
    if not href:
        raise RuntimeError(f"4DN metadata for {accession} did not contain a download href")
    return urllib.parse.urljoin(PORTAL, href)


def download(accession: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        print(f"exists  {destination}")
        return
    url = resolve_download_url(accession)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"fetch   {accession}")
    with urllib.request.urlopen(request(url), timeout=120) as response:
        with temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
    temporary.replace(destination)
    print(f"saved   {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--show-accessions",
        action="store_true",
        help="Print the required accessions without making network requests.",
    )
    args = parser.parse_args()
    if args.show_accessions:
        print("\n".join(ACCESSIONS))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for accession in ACCESSIONS:
        download(accession, args.output_dir / f"{accession}.csv", args.force)


if __name__ == "__main__":
    main()
