"""Download the official DPR Wikipedia archive and extract a reservoir sample.

The upstream DPR downloader decompresses the complete corpus in memory. This
helper uses its resource identifier and URL, preserves the .gz archive, and
streams the TSV to keep memory use bounded.
"""

import argparse
import csv
import gzip
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from dpr.data.download_data import RESOURCES_MAP  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")

    resource = RESOURCES_MAP["data.wikipedia_split.psgs_w100"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "psgs_w100.tsv.gz"
    sample_path = args.output_dir / f"psgs_w100_sample_{args.count}.tsv"
    manifest_path = args.output_dir / f"psgs_w100_sample_{args.count}.manifest.json"

    if args.download:
        if archive.exists():
            raise FileExistsError(f"Refusing to overwrite existing archive: {archive}")
        import wget

        wget.download(resource["s3_url"], out=str(archive))
        print()

    if not archive.is_file():
        raise FileNotFoundError(f"Expected downloaded archive at {archive}")

    rng = random.Random(args.seed)
    reservoir = []
    total = 0
    with gzip.open(archive, "rt", encoding="utf-8", newline="") as source:
        rows = csv.reader(source, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        header = next(rows)
        if len(header) < 3 or header[:3] != ["id", "text", "title"]:
            raise ValueError(f"Unexpected DPR TSV header: {header!r}")
        for row in rows:
            if len(row) != len(header):
                raise ValueError(f"Malformed TSV row near passage {total + 1}: {len(row)} fields")
            total += 1
            if len(reservoir) < args.count:
                reservoir.append(row)
            else:
                j = rng.randrange(total)
                if j < args.count:
                    reservoir[j] = row

    if total < args.count:
        raise ValueError(f"Corpus has only {total} passages; requested {args.count}")

    with sample_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(reservoir)

    ids = [row[0] for row in reservoir]
    if len(set(ids)) != args.count:
        raise ValueError("Sample contains duplicate passage IDs")
    manifest = {
        "source_resource": "data.wikipedia_split.psgs_w100",
        "source_url": resource["s3_url"],
        "archive": str(archive.resolve()),
        "archive_bytes": archive.stat().st_size,
        "passages_in_source": total,
        "sample_file": str(sample_path.resolve()),
        "sample_count": len(reservoir),
        "sample_method": "reservoir sampling over the complete corpus",
        "seed": args.seed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
