"""Download the documented GitHub release assets and verify their SHA256."""
import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def verify(path, asset):
    return path.is_file() and path.stat().st_size == asset["bytes"] and sha256(path) == asset["sha256"]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["main", "single", "all"], default="main")
    parser.add_argument("--asset", nargs="+", help="Download named files only, overriding --group")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "weights/manifest.json").read_text(encoding="utf-8"))
    if args.asset:
        unknown = set(args.asset) - {a["name"] for a in manifest["assets"]}
        if unknown:
            parser.error("Unknown artifact name(s): " + ", ".join(sorted(unknown)))
        selected = [a for a in manifest["assets"] if a["name"] in args.asset]
    else:
        selected = [a for a in manifest["assets"] if args.group == "all" or a["group"] == args.group]
    args.output.mkdir(parents=True, exist_ok=True)
    print("Selected bytes:", sum(a["bytes"] for a in selected), flush=True)
    for asset in selected:
        target = args.output / asset["name"]
        if target.exists():
            if not verify(target, asset):
                raise RuntimeError("Existing artifact has wrong size/hash: " + str(target))
            print("Verified existing:", target.name, flush=True)
        else:
            url = "https://github.com/" + manifest["repository"] + "/releases/download/" + manifest["tag"] + "/" + asset["name"]
            partial = target.with_suffix(target.suffix + ".part")
            print("Downloading:", target.name, flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "Group5-DPR-Reproduction"})
            with urllib.request.urlopen(request, timeout=120) as source, partial.open("wb") as dest:
                done, logged = 0, 0
                while True:
                    block = source.read(4 * 1024 * 1024)
                    if not block:
                        break
                    dest.write(block)
                    done += len(block)
                    if done - logged >= 128 * 1024 * 1024:
                        print(target.name, round(100 * done / asset["bytes"], 1), "%", flush=True)
                        logged = done
            if not verify(partial, asset):
                raise RuntimeError("Downloaded artifact has wrong size/hash: " + str(partial))
            partial.replace(target)
            print("Verified download:", target.name, flush=True)
        if target.name == "qa-tests.zip":
            with zipfile.ZipFile(target) as archive:
                for member in archive.infolist():
                    destination = (args.output / member.filename).resolve()
                    if not destination.is_relative_to(args.output.resolve()):
                        raise RuntimeError("Unsafe archive entry: " + member.filename)
                archive.extractall(args.output)
            print("Extracted QA tests and source license files.", flush=True)

if __name__ == "__main__":
    main()
