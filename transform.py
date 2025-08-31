import json, os, re, tempfile, argparse, sys
from pathlib import Path
from typing import Any


# ---------- 格式化工具 ---------- #
def to_inline_list(lst: list[int], sep: str = ",") -> str:
    return "[" + sep.join(map(str, lst)) + "]"


def clean_string(s: str, mode: str = "space") -> str:
    """
    mode = "space" : 把真正的换行替换为单个空格
    mode = "escape": 把换行替换为字面量 \\n
    """
    if mode == "space":
        return re.sub(r"\s*\n\s*", " ", s)
    else:  # "escape"
        return s.replace("\\", "\\\\").replace("\n", "\\n")


def transform(obj: Any) -> Any:
    """递归：压平 list，清理字符串换行"""
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


# ---------- 写出一行一个对象 ---------- #
def write_jsonl(objs, dst: Path):
    with dst.open("w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False, separators=(",", ": ")) + "\n")


# ---------- 主逻辑 ---------- #
def convert(src_path: Path):
    """转换JSON文件为JSONL格式"""
    if not src_path.exists():
        print(f"❌ 文件不存在: {src_path}")
        return False

    tmp_fd, tmp_name = tempfile.mkstemp(dir=src_path.parent, suffix=".tmp")
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)

    try:
        with src_path.open("r", encoding="utf-8") as f:
            first_non_ws = f.read(1).lstrip()

        if first_non_ws == "[":  # --- 输入是 JSON 数组 ---
            print(f"📖 检测到JSON数组格式: {src_path}")
            data = json.loads(src_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError("文件以 [ 开头但不是数组")
            objs = [transform(o) for o in data]
            print(f"📊 转换了 {len(objs)} 个对象")
        else:  # --- 输入已是 JSONL ---
            print(f"📖 检测到JSONL格式: {src_path}")
            objs = []
            with src_path.open("r", encoding="utf-8") as fin:
                for line_num, line in enumerate(fin, 1):
                    line = line.strip()
                    if line:
                        try:
                            objs.append(transform(json.loads(line)))
                        except json.JSONDecodeError as e:
                            print(f"⚠️  第 {line_num} 行解析失败: {e}")
                            continue
            print(f"📊 转换了 {len(objs)} 个对象")

        write_jsonl(objs, tmp_path)
        tmp_path.replace(src_path)  # 原子覆盖
        print(f"✅ 已转为一行一个对象 → {src_path}")
        return True
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        tmp_path.unlink(missing_ok=True)
        return False


def main():
    """主函数：处理命令行参数"""
    parser = argparse.ArgumentParser(
        description="将JSON文件转换为JSONL格式（每行一个JSON对象）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python convert_jsonl.py input.json
  python convert_jsonl.py /path/to/results_3b_opt1_05_msg12_bit2_seg6.json
  python convert_jsonl.py *.json  # 批量处理多个文件
        """
    )

    parser.add_argument(
        "files",
        nargs="+",
        help="要转换的JSON文件路径（支持多个文件）"
    )

    parser.add_argument(
        "--backup",
        action="store_true",
        help="转换前创建备份文件（添加.bak后缀）"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="指定输出目录（默认覆盖原文件）"
    )

    args = parser.parse_args()

    # 处理文件路径
    file_paths = []
    for file_pattern in args.files:
        path = Path(file_pattern)
        if path.is_file():
            file_paths.append(path)
        elif "*" in file_pattern or "?" in file_pattern:
            # 支持通配符
            import glob
            matched_files = glob.glob(file_pattern)
            file_paths.extend([Path(f) for f in matched_files])
        else:
            print(f"❌ 文件不存在: {file_pattern}")
            continue

    if not file_paths:
        print("❌ 没有找到要处理的文件")
        sys.exit(1)

    print(f"📁 找到 {len(file_paths)} 个文件待处理")

    success_count = 0
    for file_path in file_paths:
        print(f"\n🔄 处理文件: {file_path}")

        # 创建备份
        if args.backup:
            backup_path = file_path.with_suffix(file_path.suffix + ".bak")
            try:
                backup_path.write_bytes(file_path.read_bytes())
                print(f"💾 备份已创建: {backup_path}")
            except Exception as e:
                print(f"⚠️  备份失败: {e}")

        # 处理输出路径
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / file_path.name

            # 先复制到输出目录，再转换
            try:
                output_path.write_bytes(file_path.read_bytes())
                if convert(output_path):
                    success_count += 1
                    print(f"📁 输出文件: {output_path}")
            except Exception as e:
                print(f"❌ 复制到输出目录失败: {e}")
        else:
            # 直接转换原文件
            if convert(file_path):
                success_count += 1

    print(f"\n🎉 处理完成: {success_count}/{len(file_paths)} 个文件转换成功")


if __name__ == "__main__":
    main()