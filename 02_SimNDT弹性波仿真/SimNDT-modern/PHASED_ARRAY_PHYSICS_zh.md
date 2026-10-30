# SimNDT Modern 2.0 二维 P–SV 相控阵物理说明

## 模型范围

本实现求解二维平面应变、各向同性线弹性介质中的 P–SV 波。阵元代表垂直于计算平面无限延伸的线阵截面，因此能够严格描述成像平面内的偏转、聚焦、P/SV 模态和二维几何扩散，但不能替代真实三维探头的阵元高度、elevation focus、三维栅瓣或绝对三维声压标定。

## 控制方程与离散

采用一阶速度–应力方程：

\[
\rho \dot{\mathbf v}=\nabla\cdot\boldsymbol\sigma,
\qquad
\dot{\boldsymbol\sigma}=\lambda\,\mathrm{tr}(\nabla\mathbf v)\mathbf I
+2\mu\,\mathrm{sym}(\nabla\mathbf v).
\]

`σxx`、`σyy` 位于单元中心，`vx`、`vy` 位于对应面，`σxy` 位于单元角点；时间采用 leapfrog 等价更新。密度在速度位置按物理密度平均，剪切模量在角点使用调和平均。圆形空洞用低阻抗、零剪切模量声学填充正则化，保留钢–空气反射系数并避免把空洞误设为固定边界。

顶面满足零牵引条件。阵元发射时，仅在阵元覆盖范围用规定法向或切向牵引替换相应的零牵引分量。输入幅值单位为 Pa，不再把压力波形作为无量纲应力增量逐步累加。

外边界使用非分裂卷积 PML（CPML），整数和半网格位置分别计算递推系数。平面波基准中，8 mm 底部 CPML 的回波低于 −50 dB；较薄 PML 可以用于快速预览，但不得宣称达到同一反射指标。

## 阵元几何

阵元在 SI 物理坐标中定义。第 `i` 个阵元中心为

\[
x_i=x_c+\left(i-\frac{N-1}{2}\right)(w+k),
\]

其中 `w` 是阵元宽度，`k` 是 kerf，`w+k` 是 pitch。阵元与表面网格节点的耦合权重由节点控制体和物理阵元区间的重叠长度计算，因此满足

\[
\sum_j W_{ij}\,\Delta x=w_i.
\]

这与有限差分离散保持守恒；没有直接复制 k-Wave 面向 Fourier collocation 的带限插值权重。

## 发射延迟与加窗

对焦点 `rf`，阵元传播距离和非负发射延迟为

\[
R_i=\|\mathbf r_f-\mathbf r_i\|,
\qquad
\tau_i=\frac{\max_jR_j-R_i}{c_P}.
\]

平面波偏转使用

\[
\tau_i=\frac{x_i\sin\theta-\min_j(x_j\sin\theta)}{c_P}.
\]

每个阵元的牵引为

\[
q_i(t)=A_i w_i q_0(t-\tau_i).
\]

Gaussian tone burst 直接在 `t-τi` 上解析求值，因此延迟没有整数采样量化。支持 rectangular、Hann、Hamming、Tukey 加窗及独立收发有效阵元掩码。

## 接收、DAS、FMC 与 TFM

每个阵元先按其物理覆盖权重积分，生成 `Ne × Nt` 的原始 RF 矩阵。固定焦点接收使用分数延迟插值后加权求和，原始 RF 始终保留。

`run_fmc` 依次单阵元发射并记录所有接收阵元，返回 `Ntx × Nrx × Nt`。`tfm_image` 根据均匀介质中的

\[
t_{ij}(\mathbf r)=\frac{\|\mathbf r-\mathbf r_i\|+\|\mathbf r-\mathbf r_j\|}{c_P}
\]

进行 P 波 TFM 重建。异质介质的严格 TFM 需要另行提供走时场，不能使用此均匀声速近似。

## 验证门禁

- P 波速误差：约 0.1%；
- SV 波速误差：约 0.3%；
- 聚焦几何到达时间差：机器精度；
- 17° 平面波阵列因子主瓣：17° ± 0.05°；
- 8 mm CPML 平面波反射：低于 −50 dB；
- NumPy、Numba/Cython/OpenCL 严格兼容路径和 CuPy CUDA 路径：统一回归测试通过；
- 工程 JSON 向后兼容：1.1 单孔径工程迁移为一个物理阵元。

验证命令：`run_tests.cmd`。基准命令：`run_benchmark.cmd`。

## 结果单位

- `receiver_channels_pa`：逐阵元近表面应力，Pa；
- `receiver_pa`：静态接收聚焦后的应力 A 扫，Pa；
- `field_magnitude`、`snapshots_magnitude`：粒子速度模，m/s；
- `transmit_delays_s`：发射延迟，s；
- `element_centers_x_m`：阵元中心，m。
