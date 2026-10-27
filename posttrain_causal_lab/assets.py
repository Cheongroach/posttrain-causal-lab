"""Copy or download model files, requiring published byte sizes and hashes."""
import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_model(spec, destination, cache_root):
    root = destination / spec["model"].split("/")[-1]
    root.mkdir(parents=True, exist_ok=True)

    def prepare(record):
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe model path")
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        source = cache_root / root.name / relative
        if not (path.exists() and path.stat().st_size == record["size"]
                and digest(path) == record["sha256"]):
            temporary = path.with_name(path.name + ".part")
            if source.is_file():
                shutil.copyfile(source, temporary)
            else:
                subprocess.run([
                    "curl", "-fL", "--connect-timeout", "20", "--max-time", "1200",
                    "--retry", "1", "--speed-limit", "10000", "--speed-time", "60",
                    record["url"], "-o", str(temporary)], check=True)
            if temporary.stat().st_size != record["size"] or digest(temporary) != record["sha256"]:
                raise RuntimeError("Published model hash differs: " + str(relative))
            temporary.replace(path)
        print(json.dumps({"event": "model_file_verified", "model": spec["model"],
                          "path": str(relative), "bytes": record["size"],
                          "unix_seconds": time.time()}), flush=True)
        return record["path"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        verified = list(pool.map(prepare, spec["files"]))
    (root / "verified.json").write_text(json.dumps({
        "model": spec["model"], "hf_revision": spec["hf_revision"], "files": verified}, indent=2))
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--only")
    args = parser.parse_args()
    for spec in json.loads(args.manifest.read_text())["models"]:
        if args.only is None or spec["tag"] == args.only:
            prepare_model(spec, args.destination, args.cache_root)


if __name__ == "__main__":
    main()
