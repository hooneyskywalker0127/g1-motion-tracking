# BeyondMimic 쪽에 두는 스크립트

이 폴더의 파일은 이 저장소에서 실행하지 않습니다. BeyondMimic
(`whole_body_tracking`) 체크아웃의 `scripts/rsl_rl/` 아래에 복사해서 씁니다.
IsaacLab 환경을 띄워야 하기 때문입니다.

    cp scripts/beyondmimic/eval_sym.py <whole_body_tracking>/scripts/rsl_rl/

## eval_sym.py

Isaac 쪽 평가입니다. MuJoCo 쪽 `src/score_standard.py` 와 같은 정의로 다섯
지표를 내고, 두 성공 판정을 서로 독립으로 계산합니다. `scripts/sym_all.sh` 가
이것을 호출합니다.

기본 `play.py` · `eval.py` 와 다른 점 셋입니다.

    --init_randomize  초기 상태 교란만 켠다. 물성·질량중심·밀치기는 끈 채로 둔다.
                      MuJoCo 에서 똑같이 재현할 수 있는 것만 켜야 대칭이 된다
    종료조건 끔       끝까지 굴리고 판정은 사후에 계산한다. 켜 두면 넘어진 뒤의
                      전역 오차가 기록되지 않아 PolySim 기준이 비대칭이 된다
    시점 정렬         레퍼런스와 로봇을 같은 순간에 읽는다

`<out>_env0.npz` 를 같이 남깁니다. MuJoCo 채점기로 다시 채점해 온라인 값과
맞대는 대조용이고, 키 이름이 MuJoCo 쪽 롤아웃 npz 와 같습니다.
