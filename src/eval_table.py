"""평가 결과를 리타게팅 품질 옆에 놓고 표로 만든다.

    python src/eval_table.py

리타게팅 단계의 발 오차와 정책의 완주율·추적 오차를 한 줄에 놓는다.
발 오차가 작은 시퀀스가 정말 학습도 잘 되는지 보려는 표다.
"""

import argparse
import csv
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent


def load_quality(path):
    """시퀀스별 발 추적 오차(cm). quality_metrics.py가 만든 csv에서 읽는다."""
    out = {}
    if not path.exists():
        return out
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["seq"]] = float(row["feet_mean"])
    return out


def main(eval_dir, quality_csv, out_path):
    quality = load_quality(quality_csv)
    rows = []
    for p in sorted(pathlib.Path(eval_dir).glob("*.json")):
        d = json.load(open(p))
        rows.append(d)

    order = [s.strip() for s in open(REPO / "configs" / "train_order.txt") if s.strip()]
    rank = {s: i for i, s in enumerate(order)}
    rows.sort(key=lambda d: (rank.get(d["motion"], -1), d["motion"]))

    head = ("| 시퀀스 | 발 오차 (cm) | 완주율 | E_g-mpbpe (mm) | E_mpbpe (mm) |"
            " E_mpjpe (rad) | 롤아웃 |")
    sep = "| --- | --- | --- | --- | --- | --- | --- |"
    lines = [head, sep]
    for d in rows:
        feet = quality.get(d["motion"])
        feet_s = f"{feet:.2f}" if feet is not None else "-"
        lines.append(
            f"| {d['motion']} | {feet_s} | {d['success_rate'] * 100:.0f}% |"
            f" {d['e_g_mpbpe_mm']:.0f} | {d['e_mpbpe_mm']:.0f} |"
            f" {d['e_mpjpe_rad']:.3f} | {d['num_envs']}{'·dr' if d['randomized'] else ''} |"
        )
    table = "\n".join(lines)
    print(table)
    if out_path:
        pathlib.Path(out_path).write_text(table + "\n")
        print(f"\n저장: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", default=str(REPO / "outputs" / "eval"))
    p.add_argument("--quality_csv", default=str(REPO / "outputs" / "metrics" / "quality.csv"))
    p.add_argument("--out", default=str(REPO / "outputs" / "eval" / "table.md"))
    a = p.parse_args()
    main(a.eval_dir, pathlib.Path(a.quality_csv), a.out)
