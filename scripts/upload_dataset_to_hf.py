"""
Sync data/wikiart and the selected index CSV to a Hugging Face dataset repo via the HF CLI.
Assumes data/wikiart contains only selected images (run scripts/move_excluded_to_wikiart_excluded.py first).
Usage:
  python scripts/upload_dataset_to_hf.py --repo-id USERNAME/artyset [OPTIONS]
  python scripts/upload_dataset_to_hf.py --repo-id artyset --username USERNAME
Requires: HF_TOKEN or `hf auth login`. Uses `hf upload` (index) + `hf upload-large-folder` (images).
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WIKIART_ROOT = DATA_DIR / "wikiart"
INDEX_SELECTED = DATA_DIR / "wikiart_index_selected.csv"


def get_hf_username(token: str | None) -> str | None:
    """Return HF username via `hf auth whoami -q`."""
    try:
        out = subprocess.run(
            ["hf", "auth", "whoami", "-q"],
            capture_output=True,
            text=True,
            timeout=10,
            env={**__import__("os").environ, **({"HF_TOKEN": token} if token else {})},
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def ensure_repo(repo_id: str, repo_type: str, token: str | None) -> None:
    """Create the repo if it does not exist (hf repos create)."""
    cmd = ["hf", "repos", "create", repo_id, "--type", repo_type]
    if token:
        cmd += ["--token", token]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0 and "already exists" not in r.stderr.lower():
        print(r.stderr, file=sys.stderr)
        raise RuntimeError(f"Failed to create repo: {r.stderr}")


def hf_upload(
    repo_id: str,
    local_path: Path,
    path_in_repo: str,
    repo_type: str = "dataset",
    token: str | None = None,
    commit_message: str | None = None,
) -> None:
    """Run `hf upload REPO_ID LOCAL_PATH PATH_IN_REPO --repo-type=dataset`."""
    cmd = [
        "hf", "upload", repo_id, str(local_path), path_in_repo,
        "--repo-type", repo_type,
    ]
    if token:
        cmd += ["--token", token]
    if commit_message:
        cmd += ["--commit-message", commit_message]
    r = subprocess.run(cmd, timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(f"hf upload exited with {r.returncode}")


def hf_upload_large_folder(
    repo_id: str,
    folder_path: Path,
    repo_type: str = "dataset",
    token: str | None = None,
    num_workers: int | None = None,
    include: str | None = "**/*.jpg",
    exclude: str | None = None,
) -> None:
    """Run `hf upload-large-folder REPO_ID LOCAL_PATH --repo-type=dataset` (resumable)."""
    cmd = ["hf", "upload-large-folder", repo_id, str(folder_path), "--repo-type", repo_type]
    if token:
        cmd += ["--token", token]
    if num_workers is not None:
        cmd += ["--num-workers", str(num_workers)]
    if include:
        cmd += ["--include", include]
    if exclude:
        cmd += ["--exclude", exclude]
    r = subprocess.run(cmd, timeout=24 * 3600)
    if r.returncode != 0:
        raise RuntimeError(f"hf upload-large-folder exited with {r.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync data/wikiart and index CSV to a Hugging Face dataset repo (artyset).",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Dataset repo id (e.g. USERNAME/artyset or artyset to use current user)",
    )
    parser.add_argument(
        "--wikiart-root",
        type=Path,
        default=WIKIART_ROOT,
        help=f"WikiArt folder to sync (default: {WIKIART_ROOT})",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=INDEX_SELECTED,
        help=f"Index CSV to upload (default: {INDEX_SELECTED})",
    )
    parser.add_argument("--token", default=None, help="HF token (default: HF_TOKEN env)")
    parser.add_argument("--username", default=None, help="HF username (default: HF_USERNAME env or hf auth whoami). Use when repo-id is just the repo name (e.g. artyset)")
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Run `hf repos create` if the repo does not exist",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Workers for hf upload-large-folder (hash/upload/commit). Increase on fast networks.",
    )
    parser.add_argument(
        "--commit-message",
        default="Sync WikiArt selected subset (images + index)",
        help="Commit message for the upload",
    )
    args = parser.parse_args()

    token = args.token or __import__("os").environ.get("HF_TOKEN")
    if not token:
        print("Set HF_TOKEN or pass --token", file=sys.stderr)
        sys.exit(1)

    repo_id = args.repo_id.strip()
    if "/" not in repo_id:
        username = args.username or __import__("os").environ.get("HF_USERNAME") or get_hf_username(token)
        if not username:
            print("Could not infer username. Use --username USER or --repo-id USERNAME/artyset or set HF_USERNAME", file=sys.stderr)
            sys.exit(1)
        repo_id = f"{username}/{repo_id}"

    if not args.wikiart_root.exists():
        print(f"ERROR: {args.wikiart_root} not found.", file=sys.stderr)
        sys.exit(1)
    if not args.index.exists():
        print(f"ERROR: {args.index} not found.", file=sys.stderr)
        sys.exit(1)

    try:
        if args.create_repo:
            ensure_repo(repo_id, "dataset", token)

        # Upload index CSV then wikiart folder (same repo, two commits)
        hf_upload(
            repo_id=repo_id,
            local_path=args.index,
            path_in_repo="wikiart_index_selected.csv",
            repo_type="dataset",
            token=token,
            commit_message=args.commit_message,
        )
        hf_upload_large_folder(
            repo_id=repo_id,
            folder_path=args.wikiart_root,
            repo_type="dataset",
            token=token,
            commit_message="Sync WikiArt images",
            num_workers=args.num_workers,
        )
        print(f"Synced to https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
