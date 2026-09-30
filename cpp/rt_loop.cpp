// 증류한 학생 정책을 MuJoCo 에서 돌리는 실시간 루프.
//
// src/sim2sim_student.py 를 그대로 옮겼다. 물리 1kHz, 제어 50Hz, PD 를 직접
// 계산해 qfrc_applied 에 넣는 것까지 같다. 목적은 동작이 아니라 **지연**이다 —
// 제어 한 주기의 예산이 20ms 인데 그 안에 드는지를 재려고 옮겼다. 20ms 라는
// 값은 임의가 아니라 측정에서 나왔다(결과 360: 한 스텝 지연에 완주율 99.9% ->
// 29.6%).
//
// 파이썬으로는 이 질문에 답할 수 없다. 평균은 낮아도 GC 가 언제 멈춰 세울지
// 모르므로 최악을 보장하지 못한다. 여기서 보는 것도 평균이 아니라 꼬리다.

#include <mujoco/mujoco.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr double kSimDt = 0.001;
constexpr int kDecimation = 20;  // 제어 50Hz
constexpr int kNumObs = 260;
constexpr double kVelScale = 0.05;

// meta.json 을 통째로 파싱하지 않는다. 필요한 건 수 배열 몇 개와 문자열 배열
// 둘뿐이라 그것만 긁는다. 의존성을 하나라도 덜 들이려는 선택이다.
std::string Slurp(const std::string& path) {
  std::ifstream f(path);
  return std::string((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
}

std::vector<double> Numbers(const std::string& js, const std::string& key) {
  auto p = js.find("\"" + key + "\"");
  if (p == std::string::npos) return {};
  p = js.find('[', p);
  auto q = js.find(']', p);
  std::vector<double> out;
  size_t i = p + 1;
  while (i < q) {
    char* end = nullptr;
    double v = std::strtod(js.c_str() + i, &end);
    if (end == js.c_str() + i) break;
    out.push_back(v);
    i = (end - js.c_str());
    while (i < q && (js[i] == ',' || js[i] == ' ' || js[i] == '\n')) ++i;
  }
  return out;
}

std::vector<std::string> Strings(const std::string& js, const std::string& key) {
  auto p = js.find("\"" + key + "\"");
  p = js.find('[', p);
  auto q = js.find(']', p);
  std::vector<std::string> out;
  for (size_t i = p; i < q;) {
    auto a = js.find('"', i);
    if (a == std::string::npos || a > q) break;
    auto b = js.find('"', a + 1);
    out.push_back(js.substr(a + 1, b - a - 1));
    i = b + 1;
  }
  return out;
}

double Scalar(const std::string& js, const std::string& key) {
  auto p = js.find("\"" + key + "\"");
  p = js.find(':', p);
  return std::strtod(js.c_str() + p + 1, nullptr);
}

std::vector<double> ReadBin(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  auto n = f.tellg();
  f.seekg(0);
  std::vector<double> v(n / sizeof(double));
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

double Pct(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  size_t i = static_cast<size_t>(p * (v.size() - 1));
  return v[i];
}

}  // namespace

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : ".";
  std::string onnx = argc > 2 ? argv[2] : "student.onnx";
  std::string mjcf = argc > 3 ? argv[3] : "g1.xml";
  int steps = argc > 4 ? std::atoi(argv[4]) : 1000;

  const std::string js = Slurp(dir + "/meta.json");
  const auto joint_names = Strings(js, "joint_names");
  const auto body_names = Strings(js, "body_names");
  const auto qdef = Numbers(js, "default_joint_pos");
  const auto kp = Numbers(js, "stiffness");
  const auto kd = Numbers(js, "damping");
  const auto ascale = Numbers(js, "action_scale");
  const auto tlim = Numbers(js, "torque_limit");
  const int anchor = static_cast<int>(Scalar(js, "anchor"));
  const int nq = static_cast<int>(joint_names.size());
  const int nb = static_cast<int>(body_names.size());
  const int frames = static_cast<int>(Scalar(js, "frames"));

  const auto ref_jp = ReadBin(dir + "/ref_jp.bin");
  const auto ref_jv = ReadBin(dir + "/ref_jv.bin");
  const auto ref_bp = ReadBin(dir + "/ref_bp.bin");
  const auto ref_bq = ReadBin(dir + "/ref_bq.bin");

  char err[1000];
  mjModel* m = mj_loadXML(mjcf.c_str(), nullptr, err, sizeof(err));
  if (!m) {
    std::fprintf(stderr, "MJCF 를 못 읽었다: %s\n", err);
    return 1;
  }
  m->opt.timestep = kSimDt;
  mjData* d = mj_makeData(m);

  // ONNX 관절 순서 -> MuJoCo 주소
  std::vector<int> qadr(nq), vadr(nq), bid(nb);
  for (int i = 0; i < nq; ++i) {
    int j = mj_name2id(m, mjOBJ_JOINT, joint_names[i].c_str());
    if (j < 0) {
      std::fprintf(stderr, "관절이 없다: %s\n", joint_names[i].c_str());
      return 1;
    }
    qadr[i] = m->jnt_qposadr[j];
    vadr[i] = m->jnt_dofadr[j];
  }
  for (int i = 0; i < nb; ++i) {
    bid[i] = mj_name2id(m, mjOBJ_BODY, body_names[i].c_str());
  }

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "rt");
  Ort::SessionOptions so;
  so.SetIntraOpNumThreads(1);   // 실시간 루프에서 스레드 풀은 지터의 원인이다
  so.SetInterOpNumThreads(1);
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  Ort::Session sess(env, onnx.c_str(), so);
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  const char* in_names[] = {"obs"};
  const char* out_names[] = {"actions"};
  std::array<int64_t, 2> shape{1, kNumObs};

  // 관측의 속도 항에 곱하는 벡터. 자리는 env_adapter 와 같다 —
  // command 의 뒤 절반(29:58), base_lin_vel(58+9..), joint_vel, motion_joint_vel_diff.
  std::vector<float> vscale(kNumObs, 1.0f);
  for (int i = 29; i < 58; ++i) vscale[i] = kVelScale;          // 레퍼런스 관절속도
  for (int i = 67; i < 73; ++i) vscale[i] = kVelScale;          // base lin/ang vel
  for (int i = 102; i < 131; ++i) vscale[i] = kVelScale;        // joint_vel
  for (int i = 189; i < 218; ++i) vscale[i] = kVelScale;        // motion_joint_vel_diff

  std::vector<float> obs(kNumObs, 0.0f);
  std::vector<double> action(nq, 0.0), last_action(nq, 0.0), pd(nq);
  for (int i = 0; i < nq; ++i) pd[i] = qdef[i];

  // 레퍼런스 첫 프레임 자세로 놓는다
  for (int i = 0; i < nq; ++i) d->qpos[qadr[i]] = ref_jp[i];
  d->qpos[0] = ref_bp[0]; d->qpos[1] = ref_bp[1]; d->qpos[2] = ref_bp[2];
  d->qpos[3] = ref_bq[0]; d->qpos[4] = ref_bq[1];
  d->qpos[5] = ref_bq[2]; d->qpos[6] = ref_bq[3];
  mj_forward(m, d);

  std::vector<double> t_infer, t_cycle, t_phys;
  steps = std::min(steps, frames);

  for (int k = 0; k < steps; ++k) {
    auto c0 = Clock::now();

    // ---- 관측 ----
    const double* jp = &ref_jp[static_cast<size_t>(k) * nq];
    const double* jv = &ref_jv[static_cast<size_t>(k) * nq];
    const double* bp = &ref_bp[static_cast<size_t>(k) * nb * 3];
    const double* bq = &ref_bq[static_cast<size_t>(k) * nb * 4];

    const double* rob_pos = &d->xpos[3 * bid[anchor]];
    const double* R = &d->xmat[9 * bid[anchor]];        // 행 우선 3x3
    const double* ap = &bp[3 * anchor];
    const double* aq = &bq[4 * anchor];

    double w = aq[0], x = aq[1], y = aq[2], z = aq[3];
    double Rr[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)};

    double dp[3] = {ap[0] - rob_pos[0], ap[1] - rob_pos[1], ap[2] - rob_pos[2]};
    double anchor_pos_b[3];
    for (int i = 0; i < 3; ++i)
      anchor_pos_b[i] = R[i] * dp[0] + R[3 + i] * dp[1] + R[6 + i] * dp[2];  // R^T dp

    double rel[9];   // R^T Rr
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        rel[3 * i + j] = R[i] * Rr[j] + R[3 + i] * Rr[3 + j] + R[6 + i] * Rr[6 + j];

    const double* Rb = &d->xmat[9 * bid[0]];
    double blin[3];
    for (int i = 0; i < 3; ++i)
      blin[i] = Rb[i] * d->qvel[0] + Rb[3 + i] * d->qvel[1] + Rb[6 + i] * d->qvel[2];

    int o = 0;
    for (int i = 0; i < nq; ++i) obs[o++] = static_cast<float>(jp[i]);
    for (int i = 0; i < nq; ++i) obs[o++] = static_cast<float>(jv[i]);
    for (int i = 0; i < 3; ++i) obs[o++] = static_cast<float>(anchor_pos_b[i]);
    for (int i = 0; i < 3; ++i) {                     // 각 행의 앞 두 성분
      obs[o++] = static_cast<float>(rel[3 * i]);
      obs[o++] = static_cast<float>(rel[3 * i + 1]);
    }
    for (int i = 0; i < 3; ++i) obs[o++] = static_cast<float>(blin[i]);
    for (int i = 0; i < 3; ++i) obs[o++] = static_cast<float>(d->qvel[3 + i]);
    for (int i = 0; i < nq; ++i)
      obs[o++] = static_cast<float>(d->qpos[qadr[i]] - qdef[i]);
    for (int i = 0; i < nq; ++i) obs[o++] = static_cast<float>(d->qvel[vadr[i]]);
    for (int i = 0; i < nq; ++i) obs[o++] = static_cast<float>(last_action[i]);
    // 학생만 보는 뒤 100
    for (int i = 0; i < nq; ++i)
      obs[o++] = static_cast<float>(jp[i] - d->qpos[qadr[i]]);
    for (int i = 0; i < nq; ++i)
      obs[o++] = static_cast<float>(jv[i] - d->qvel[vadr[i]]);
    for (int b = 0; b < nb; ++b) {
      double e[3] = {bp[3 * b] - d->xpos[3 * bid[b]],
                     bp[3 * b + 1] - d->xpos[3 * bid[b] + 1],
                     bp[3 * b + 2] - d->xpos[3 * bid[b] + 2]};
      for (int i = 0; i < 3; ++i)
        obs[o++] = static_cast<float>(R[i] * e[0] + R[3 + i] * e[1] + R[6 + i] * e[2]);
    }
    for (int i = 0; i < kNumObs; ++i) obs[i] *= vscale[i];

    // ---- 추론 ----
    auto i0 = Clock::now();
    Ort::Value in = Ort::Value::CreateTensor<float>(mem, obs.data(), obs.size(),
                                                    shape.data(), shape.size());
    auto out = sess.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
    const float* a = out[0].GetTensorData<float>();
    auto i1 = Clock::now();
    t_infer.push_back(std::chrono::duration<double, std::milli>(i1 - i0).count());

    for (int i = 0; i < nq; ++i) {
      action[i] = a[i];
      last_action[i] = a[i];
      pd[i] = qdef[i] + ascale[i] * a[i];
    }

    // ---- 물리 20 스텝 ----
    auto p0 = Clock::now();
    for (int s = 0; s < kDecimation; ++s) {
      for (int i = 0; i < nq; ++i) {
        double tq = (pd[i] - d->qpos[qadr[i]]) * kp[i] - d->qvel[vadr[i]] * kd[i];
        tq = std::max(-tlim[i], std::min(tlim[i], tq));
        d->qfrc_applied[vadr[i]] = tq;
      }
      mj_step(m, d);
    }
    auto p1 = Clock::now();
    t_phys.push_back(std::chrono::duration<double, std::milli>(p1 - p0).count());
    t_cycle.push_back(std::chrono::duration<double, std::milli>(p1 - c0).count());
  }

  auto report = [](const char* name, const std::vector<double>& v, double budget) {
    std::printf("%-22s p50 %6.3f  p99 %6.3f  p99.9 %6.3f  max %6.3f ms", name,
                Pct(v, 0.50), Pct(v, 0.99), Pct(v, 0.999),
                *std::max_element(v.begin(), v.end()));
    if (budget > 0) {
      int over = 0;
      for (double x : v) if (x > budget) ++over;
      std::printf("   예산 %.0fms 초과 %d/%zu", budget, over, v.size());
    }
    std::printf("\n");
  };

  std::printf("스텝 %d, 제어 50Hz, 물리 1kHz\n", steps);
  report("추론", t_infer, 0);
  report("물리 20스텝", t_phys, 0);
  report("제어 주기 전체", t_cycle, 20.0);

  // 원자료를 남긴다. 히스토그램은 따로 그린다.
  std::ofstream f(dir + "/lat_cpp.csv");
  f << "infer_ms,phys_ms,cycle_ms\n";
  for (size_t i = 0; i < t_infer.size(); ++i)
    f << t_infer[i] << "," << t_phys[i] << "," << t_cycle[i] << "\n";

  // 포팅이 맞는지 보려고 마지막 상태를 남긴다. 같은 루프를 파이썬으로도 돌려
  // 이 값을 대조한다. 지연만 재고 동작이 틀리면 의미가 없다.
  std::ofstream g(dir + "/state_cpp.txt");
  g.precision(12);
  for (int i = 0; i < 7; ++i) g << d->qpos[i] << "\n";
  for (int i = 0; i < nq; ++i) g << d->qpos[qadr[i]] << "\n";

  mj_deleteData(d);
  mj_deleteModel(m);
  return 0;
}
