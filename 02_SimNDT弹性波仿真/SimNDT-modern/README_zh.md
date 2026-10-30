# SimNDT Modern 2.0

SimNDT Modern 2.0 是二维平面应变 P–SV 超声无损检测模拟器。本版加入按 k-Wave 阵列工作流设计、按弹性固体方程重新推导的相控阵：物理阵元、分数延迟、收发加窗、逐阵元 RF、DAS、FMC 和均匀介质 P 波 TFM。

物理模型、公式、验收结果和二维限制见 `PHASED_ARRAY_PHYSICS_zh.md`。

## 启动

双击 `run_modern.cmd`。工程使用 `D:\SimNDTModern` 纯英文目录联接，避免 CUDA/NVRTC 无法处理中文安装路径。命令行示例：

```powershell
.\.venv\Scripts\simndt-modern.exe `
  --project .\examples\steel_phased_array_focus.json `
  --backend cupy --output phased_array_result.npz
```

FMC 和 TFM：

```powershell
.\.venv\Scripts\simndt-modern.exe `
  --project .\examples\steel_phased_array_focus.json `
  --backend cupy --output pulse_echo.npz `
  --fmc-output fmc.npz --tfm-output tfm.npz --tfm-pixels 128
```

## 已实现

- 二阶交错网格速度–应力 P–SV 求解器；
- 法向 P 主导和切向 SV 主导表面牵引源，单位 Pa；
- 严格离散零牵引顶面；
- 整数/半网格分离的非分裂 CPML；
- 物理阵元宽度、pitch/kerf、子网格覆盖权重；
- 聚焦、平面波偏转和无整数采样误差的解析 tone-burst 时移；
- rectangular、Hann、Hamming、Tukey 收发加窗；
- 独立收发有效阵元掩码；
- 每阵元 RF、固定焦点 DAS、FMC 和均匀介质 P 波 TFM；
- 圆形低阻抗空洞；
- JSON 工程、NPZ 波场/通道/FMC/TFM 数据；
- 真实步号/时间的 PNG、GIF、MP4 过程导出；
- NumPy 严格参考内核、CuPy CUDA 数组内核；
- Numba、Cython、OpenCL 工程值通过严格参考兼容路径运行，避免调用旧同位网格方程。

## 数值设置

- `Points / shortest wavelength`：按最高建模频率和最短有效波长建网，建议 10–15；
- `CFL`：二维显式交错格式默认 0.45；
- `Focus depth = 0`：平面波偏转；大于零：有限距离聚焦；
- pitch 超过当前偏转角的无栅瓣限制时，验证摘要会标记风险；
- 已验证 8 mm 底部 CPML 的平面波反射低于 −50 dB；薄层只适合预览。

## 二维限制

阵元在二维模型中是无限延伸线源。结果不包含阵元高度、elevation focus、三维扩散或三维栅瓣。若要和真实探头绝对声压逐点对比，需要三维模型或实验标定。

均匀声速 TFM 只适合均匀材料。异质材料、楔块或曲面中的严格 TFM 需要单独计算传播时间场。

## 输出字段

- `receiver_channels_pa`：`Ne × Nt` 逐阵元 RF；
- `receiver_pa`：静态接收聚焦 A 扫；
- `element_centers_x_m`、`transmit_delays_s`、收发加窗；
- `field_magnitude` 和时间历程：粒子速度模，m/s；
- FMC：`Ntx × Nrx × Nt`。

## 验证

双击 `run_tests.cmd`。2026-08-24 验证为 28 项全部通过，覆盖 k-Wave 延迟公式等价、三分辨率网格收敛、P/SV 波速、相控延迟、阵列因子、CPML、FMC/TFM、GPU 一致性、GUI、文件格式和动画导出。

## 目录

```text
src/simndt_modern/
  models.py       单位化材料、阵列、边界和数值模型
  signals.py      tone burst、分数延迟、加窗和 DAS
  grid.py         网格、CPML、空洞和物理阵元映射
  validation.py   阵列因子、焦点到达时间和质量摘要
  solver.py       后端、FMC、TFM 和统一结果接口
  app.py          PySide6 GUI
  io.py           JSON/NPZ
tests/            数值、物理、后端和 GUI 门禁
```
