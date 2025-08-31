import json, os, re, tempfile, argparse, sys
from pathlib import Path
from typing import Any


# ---------- Formatting Utilities ---------- #
def to_inline_list(lst: list[int], sep: str = ",") -> str:
    return "[" + sep.join(map(str, lst)) + "]"


def clean_string(s: str, mode: str = "space") -> str:
    """
    mode = "space" : replace real line breaks with a single space
    mode = "escape": replace line breaks with literal \\n
    """
    if mode == "space":
        return re.sub(r"\s*\n\s*", " ", s)
    else:  # "escape"
        return s.replace("\\", "\\\\").replace("\n", "\\n")


def transform(obj: Any) -> Any:
    """Recursively flatten lists and clean string newlines"""
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            if isinstance(v, list):
                new[k] = to_inline_list(v)
            elif isinstance(v, str):
                new[k] = clean_string(v, mode="space")
            else:
                new[k] = transform(v)
        return new
    elif isinstance(obj, list):
        return [transform(x) for x in obj]
    elif isinstance(obj, str):
        return clean_string(obj, mode="space")
    else:
        return obj


# ---------- Write JSONL (one object per line) ---------- #
def write_jsonl(objs, dst: Path):
    with dst.open("w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False, separators=(",", ": ")) + "\n")


# ---------- Conversion Logic ---------- #
def convert(src_path: Path):
    """Convert JSON file into JSONL format"""
    if not src_path.exists():
        print(f"❌ File does not exist: {src_path}")
        return False

    tmp_fd, tmp_name = tempfile.mkstemp(dir=src_path.parent, suffix=".tmp")
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)

    try:
        with src_path.open("r", encoding="utf-8") as f:
            first_non_ws = f.read(1).lstrip()

        if first_non_ws == "[":  # --- Input is JSON array ---
            print(f"📖 Detected JSON array format: {src_path}")
            data = json.loads(src_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError("File starts with [ but is not an array")
            objs = [transform(o) for o in data]
            print(f"📊 Converted {len(objs)} objects")
        else:  # --- Input is already JSONL ---
            print(f"📖 Detected JSONL format: {src_path}")
            objs = []
            with src_path.open("r", encoding="utf-8") as fin:
                for line_num, line in enumerate(fin, 1):
                    line = line.strip()
                    if line:
                        try:
                            objs.append(transform(json.loads(line)))
                        except json.JSONDecodeError as e:
                            print(f"⚠️  Line {line_num} parsing failed: {e}")
                            continue
            print(f"📊 Converted {len(objs)} objects")

        write_jsonl(objs, tmp_path)
        tmp_path.replace(src_path)  # atomic replace
        print(f"✅ Successfully converted to one-object-per-line → {src_path}")
        return True
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        tmp_path.unlink(missing_ok=True)
        return False


def main():
    """Main function: parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Convert JSON files to JSONL format (one object per line)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transform.py input.json
  python transform.py /path/to/results.json
  python transform.py *.json  # batch process multiple files
        """
    )

    parser.add_argument(
        "files",
        nargs="+",
        help="Path(s) to JSON file(s) (supports multiple)"
    )

    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup file before conversion (.bak suffix)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Specify output directory (default: overwrite original)"
    )

    args = parser.parse_args()

    # Collect file paths
    file_paths = []
    for file_pattern in args.files:
        path = Path(file_pattern)
        if path.is_file():
            file_paths.append(path)
        elif "*" in file_pattern or "?" in file_pattern:
            # Support wildcards
            import glob
            matched_files = glob.glob(file_pattern)
            file_paths.extend([Path(f) for f in matched_files])
        else:
            print(f"❌ File not found: {file_pattern}")
            continue

    if not file_paths:
        print("❌ No files found to process")
        sys.exit(1)

    print(f"📁 Found {len(file_paths)} file(s) to process")

    success_count = 0
    for file_path in file_paths:
        print(f"\n🔄 Processing file: {file_path}")

        # Create backup
        if args.backup:
            backup_path = file_path.with_suffix(file_path.suffix + ".bak")
            try:
                backup_path.write_bytes(file_path.read_bytes())
                print(f"💾 Backup created: {backup_path}")
            except Exception as e:
                print(f"⚠️  Backup failed: {e}")

        # Handle output path
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / file_path.name

            try:
                output_path.write_bytes(file_path.read_bytes())
                if convert(output_path):
                    success_count += 1
                    print(f"📁 Output file: {output_path}")
            except Exception as e:
                print(f"❌ Failed to copy to output dir: {e}")
        else:
            if convert(file_path):
                success_count += 1

    print(f"\n🎉 Done: {success_count}/{len(file_paths)} files converted successfully")


if __name__ == "__main__":
    main()
