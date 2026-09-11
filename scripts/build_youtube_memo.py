"""각 시퀀스 폴더의 *_youtube.txt 넷을 모아 유튜브_메모.md 를 다시 만든다.

    python scripts/build_youtube_memo.py [시퀀스 ...]

txt 가 원본이고 md 는 사람이 보기 위한 사본이다. txt 를 고쳤으면 이걸 다시 돌린다.
"""
import pathlib
import sys

BASE = pathlib.Path("/home/sehoon/Desktop/참고/할일/26/09/g1_motion_tracking")
SECTIONS = [
    ("compare", "정책과 레퍼런스 비교"),
    ("progression", "발전 과정"),
    ("randomization", "도메인 랜덤화"),
    ("sim2sim", "Sim-to-Sim (MuJoCo)"),
]


def build(seq):
    d = BASE / seq
    out = [f"# {seq} 유튜브 메모", ""]
    n = 0
    for sub, ko in SECTIONS:
        f = d / sub / f"{seq}_{sub}_youtube.txt"
        if not f.exists():
            continue
        title, body = f.read_text().split("\n", 1)
        out += [f"## {ko}", "", "제목", "", "```", title, "```", "",
                "설명", "", "```", body.strip(), "```", ""]
        n += 1
    (d / "유튜브_메모.md").write_text("\n".join(out))
    return n


if __name__ == "__main__":
    seqs = sys.argv[1:] or sorted(
        d.name for d in BASE.iterdir() if (d / "sim2sim").is_dir())
    for s in seqs:
        print(f"{s}  {build(s)}절")
