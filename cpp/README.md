# 실시간 추론 루프

`src/sim2sim_student.py` 의 제어 루프를 C++ 로 옮긴 것이다. 물리 1kHz, 제어 50Hz,
PD 를 직접 계산해 `qfrc_applied` 에 넣는 것까지 같다.

옮긴 이유는 성능이 아니라 **지연을 재려고**다. 한 스텝(20ms) 지연이 완주율을
99.9% 에서 29.6% 로 떨어뜨리므로, 추론이 그 예산 안에 드는지가 실제 배포의 조건이 된다.

| 파일 | 하는 일 |
| --- | --- |
| `dump.py` | ONNX 메타데이터와 npz 레퍼런스를 평평한 바이너리로 내보낸다 |
| `rt_loop.cpp` | C++ 루프. 지연을 재고 `lat_cpp.csv` 를 남긴다 |
| `py_loop.py` | 같은 파일을 읽는 파이썬 루프. 언어만 다르게 둔 대조군 |

## 빌드

pip 로 깔린 라이브러리에 직접 링크한다. ONNX Runtime 은 C++ 헤더가 pip 패키지에
없으므로 릴리스 tarball 에서 `include/` 만 받는다.

```bash
P=$(python -c "import site;print(site.getsitepackages()[0])")
g++ -O2 -std=c++17 rt_loop.cpp -o rt_loop \
    -I$P/mujoco/include -Ionnxruntime-linux-x64-<ver>/include \
    -L. -lmujoco -lonnxruntime
```

## 확인한 것

두 구현은 1000 제어 스텝 뒤 루트 위치·자세·관절각이 완전히 같다. 지연만 재고
동작이 틀리면 뜻이 없으므로 이걸 먼저 맞췄다.

측정값과 해석은 `logs/distill/aiming_분석계획.md` 결과 361 에 있다. 요약하면
추론이 0.4ms, 제어 주기 전체가 0.9ms 로 20ms 예산에 열 배 가까운 여유가 있고,
C++ 로 얻은 것은 주기당 약 0.2ms 다. **최악값을 정하는 것은 언어가 아니라 OS 이므로,
"C++ 이라서 실시간이 보장된다" 는 주장은 이 측정으로 할 수 없다.**
