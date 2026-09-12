#!/usr/bin/env python3
"""공식 런치가 띄운 MuJoCo simulate 창을 찾아 추적 카메라로 바꾸고 창 위치를 돌려준다.

MuJoCo 의 simulate 뷰어는 `[` `]` 로 카메라를 순환한다 (Free → 모델에 정의된 카메라).
오버레이 MJCF 에 `<camera name="track" mode="trackcom">` 을 넣어 뒀으므로 한 번 누르면
로봇을 따라가는 시점이 된다. Isaac 쪽이 `viewer.origin_type = "asset_root"` 로
로봇을 따라가는 것과 같은 개념이다 (tracking_env_cfg.py:319-322).

물리나 관측은 건드리지 않는다. 창에 키 하나를 보낼 뿐이다.

    python3 scripts/mujoco_track_cam.py          # 카메라 전환 + 창 위치 출력
    python3 scripts/mujoco_track_cam.py --geom   # 위치만 출력
"""
import argparse
import time

from Xlib import X, XK, display
from Xlib.ext import xtest


def find_window(dsp, root, depth=0):
    """MuJoCo simulate 창을 이름으로 찾는다."""
    found = []
    for w in root.query_tree().children:
        try:
            name = w.get_wm_name() or ""
        except Exception:
            name = ""
        if name and ("MuJoCo" in name or "mujoco" in name or "Simulate" in name):
            found.append((w, name))
        if depth < 3:
            found += find_window(dsp, w, depth + 1)
    return found


def send_key(dsp, win, keysym_name):
    """GLFW 는 send_event 로 만든 합성 이벤트를 무시한다. XTEST 로 실제 입력을 넣는다."""
    keysym = XK.string_to_keysym(keysym_name)
    keycode = dsp.keysym_to_keycode(keysym)
    xtest.fake_input(dsp, X.KeyPress, keycode)
    dsp.sync()
    time.sleep(0.05)
    xtest.fake_input(dsp, X.KeyRelease, keycode)
    dsp.sync()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", action="store_true", help="카메라 전환 없이 창 위치만 출력")
    ap.add_argument("--presses", type=int, default=1, help="`]` 를 몇 번 누를지")
    a = ap.parse_args()

    dsp = display.Display()
    wins = find_window(dsp, dsp.screen().root)
    if not wins:
        raise SystemExit("MuJoCo 창을 못 찾았다")
    win, name = wins[0]
    g = win.get_geometry()
    abs_pos = win.translate_coords(dsp.screen().root, 0, 0)
    x, y = -abs_pos.x, -abs_pos.y
    print(f"창 '{name}'  {g.width}x{g.height}+{x}+{y}")

    # x11grab 은 화면 영역을 뜨므로 다른 창이 위에 있으면 그게 녹화된다. 앞으로 올린다.
    win.configure(stack_mode=X.Above)
    dsp.sync()
    time.sleep(0.3)

    if not a.geom:
        win.set_input_focus(X.RevertToParent, X.CurrentTime)
        dsp.sync()
        for _ in range(a.presses):
            send_key(dsp, win, "bracketright")
            time.sleep(0.2)
        print(f"']' {a.presses}회 전송 (Free → track 카메라)")
