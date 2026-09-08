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
    notes = []
    for d in rows:
        feet = quality.get(d["motion"])
        feet_s = f"{feet:.2f}" if feet is not None else "-"
        # 세 지표는 완주한 롤아웃에서만 정의된다. 완주가 없으면 NaN이라 그대로
        # 쓰면 "nan"이 찍힌다. 칸은 비우고 실제로 잰 값은 표 아래에 적는다.
        finished = d["success_rate"] > 0 and d["e_g_mpbpe_mm"] == d["e_g_mpbpe_mm"]
        if finished:
            errs = (f" {d['e_g_mpbpe_mm']:.0f} | {d['e_mpbpe_mm']:.0f} |"
                    f" {d['e_mpjpe_rad']:.3f} |")
        else:
            errs = " - | - | - |"
            alive = d.get("mean_alive_frames", 0.0)
            total = d.get("motion_frames", 0) or 1
            notes.append(
                f"- {d['motion']}: 완주한 롤아웃이 없어 세 지표가 정의되지 않는다."
                f" 평균 추적 길이 {alive / total * 100:.0f}%"
                f" ({alive:,.0f}/{total:,} 프레임),"
                f" 전체 롤아웃 E_g-mpbpe {d.get('e_g_mpbpe_mm_all', float('nan')):.0f} mm"
            )
        lines.append(
            f"| {d['motion']} | {feet_s} | {d['success_rate'] * 100:.0f}% |"
            f"{errs} {d['num_envs']}{'·dr' if d['randomized'] else ''} |"
        )
    if notes:
        lines += [""] + notes
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
