import argparse
import json
import os
import subprocess
import zipfile

DATASET = "olgaparfenova/daisee"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEST_ROOT = os.path.join(REPO_ROOT, "data", "raw", "daisee")
LISTING_CACHE = os.path.join(DEST_ROOT, "_file_listing.json")


def list_all_files(use_cache=True):
    """Paginate through the full kaggle dataset file listing (thousands of
    individual video files, not per-split archives). Cached locally since
    a full listing takes ~46 paginated API calls."""
    if use_cache and os.path.exists(LISTING_CACHE):
        with open(LISTING_CACHE) as f:
            return json.load(f)

    all_files = []
    token = None
    while True:
        cmd = ["kaggle", "datasets", "files", "-d", DATASET, "--format", "json", "--page-size", "200"]
        if token:
            cmd += ["--page-token", token]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
        out = result.stdout
        first_line, _, rest = out.partition("\n")
        if first_line.startswith("Next Page Token"):
            next_token = first_line.split("=", 1)[1].strip()
            json_str = rest
        else:
            next_token = None
            json_str = out
        batch = json.loads(json_str)
        all_files.extend(batch)
        if not next_token or not batch or next_token == token:
            break
        token = next_token

    os.makedirs(os.path.dirname(LISTING_CACHE), exist_ok=True)
    with open(LISTING_CACHE, "w") as f:
        json.dump(all_files, f)
    return all_files


def download_split(split, files, dest_dir, limit=None, skip_existing=True):
    """Download every file under DAiSEE/DataSet/<split>/ one at a time
    (the kaggle CLI has no per-split archive or wildcard download)."""
    prefix = f"DAiSEE/DataSet/{split}/"
    split_files = [f for f in files if f["name"].startswith(prefix)]
    if limit:
        split_files = split_files[:limit]

    os.makedirs(dest_dir, exist_ok=True)
    n_downloaded, n_skipped, n_failed = 0, 0, 0

    for i, f in enumerate(split_files):
        rel_path = f["name"][len(prefix):]
        local_path = os.path.join(dest_dir, rel_path)
        if skip_existing and os.path.exists(local_path) and os.path.getsize(local_path) == f["size"]:
            n_skipped += 1
            continue
        local_dir = os.path.dirname(local_path)
        os.makedirs(local_dir, exist_ok=True)
        cmd = ["kaggle", "datasets", "download", "-d", DATASET, "-f", f["name"], "-p", local_dir, "-q"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            n_failed += 1
            print(f"[{i+1}/{len(split_files)}] FAILED {f['name']}: {result.stderr.strip()}")
            continue

        # kaggle's -f download always wraps the file in a zip; unwrap it ourselves
        # (the CLI's --unzip flag only applies to whole-dataset downloads, not -f)
        zip_path = local_path + ".zip"
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(local_dir)
            os.remove(zip_path)

        if not os.path.exists(local_path):
            n_failed += 1
            print(f"[{i+1}/{len(split_files)}] FAILED {f['name']}: no file after unzip")
            continue
        n_downloaded += 1
        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(split_files)}] downloaded {n_downloaded}, skipped {n_skipped}, failed {n_failed}")

    print(f"done: downloaded {n_downloaded}, skipped {n_skipped}, failed {n_failed}, out of {len(split_files)}")
    return n_downloaded, n_skipped, n_failed


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list")

    dl = sub.add_parser("download-split")
    dl.add_argument("split", choices=["Train", "Validation", "Test"])
    dl.add_argument("--limit", type=int, default=None, help="cap number of files (for testing)")

    args = parser.parse_args()

    if args.command == "list":
        files = list_all_files()
        video_files = [f for f in files if f["name"].lower().endswith((".avi", ".mp4"))]
        label_files = [f for f in files if not f["name"].lower().endswith((".avi", ".mp4"))]
        total_size = sum(f["size"] for f in files)
        print(f"total files: {len(files)} ({len(video_files)} videos, {len(label_files)} other)")
        print(f"total size: {total_size / 1e9:.2f} GB")
        for f in label_files:
            print(f"  {f['name']} ({f['size']} bytes)")

    elif args.command == "download-split":
        files = list_all_files()
        dest_dir = os.path.join(DEST_ROOT, args.split.lower())
        download_split(args.split, files, dest_dir, limit=args.limit)


if __name__ == "__main__":
    main()
