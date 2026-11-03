# CrI3 frozen magnon：当前入口

> 2026-09-29 路径入口已接入三胞 J：`benchmarks/path60/run_path60.py` 新增 `--config` 与 `--temperature`，复用通用参数读取与来源哈希；`prepare/merge/plot` 必须使用相同配置、温度、路径/积分网格。配置可选固定或自洽色散，命令行 `--sc-mesh`、`--xi-mev` 显式覆盖；省略配置保持历史 Luo 默认。已启动三胞 ansatz J 与修正 Luo 对照的 6.7→20→30→50 K、120点、固定xi=0.029377批次。远端结果 `../runs/ansatz_path120_20260929/REPORT.md`；[本地报告](../deployment/frozen_magnon_current_20260926/ansatz_path120_20260929/REPORT.md)。Fortran公式未改。

> 2026-09-27 四阶顶点已修正：两个 Fortran 入口统一为 `HP_COMPLEX_BONDS_V1`，保留 J1/J3 的完整复数壳层和，并修复 V2 的 J2/A 符号。独立逐键 HP 的四通道复振幅、玻色交换、厄米共轭及归一化回归通过。旧散射率仍属于旧核，不能作为修正结果使用；直接响应矩阵不受影响。见[修正记录](../deployment/frozen_magnon_current_20260926/vertex_fix_20260927/REPORT.md)，远端记录位于 `../records/vertex_fix_20260927/`。


> 同日散射计数与参数绑定修正：有序末态权重为 `(1/2,1/2,1,1)`，标记 `ORDERED_FINAL_HALF_IDENTICAL_V1`。每个工作目录首次使用时创建 `calculation_identity.json`，后续模型与数值条件必须一致。见[本次修复记录](../deployment/frozen_magnon_current_20260926/scattering_fix_20260927/REPORT.md)，远端证据在 `../records/scattering_fix_20260927/`。

位置：`25_li:/home2/25_li/CrI3/frozen_magnon/`。本地源码镜像：项目根目录 `frozen_magnon/`。
仅保留已有 Luo full-HP / Lorentzian 核、最新基准和 native 三项力矩接口。
旧 Gaussian / uinv 程序、配置和入口已从此工作目录移除。

## 当前 full14 的 J1、J2、J3

**s31+s13+SD 联合提取已完成并接入。** 旧 s13 端点所用候选与当前 full14 不同，因此仅补了两个 s13 固定势端点；入口 `../ansatz/torque_s13.py` 复用已有 native 三项力矩代码。62550 准备、62551/62552 正负端点、62553 联合分析均已生成完成结果。提交参数见 `../records/FULL14_S13_SUBMISSION_20260926.json`，实际日志见 `../records/full14_s13_logs/`。

当前输入为 `../data/current14/three_cell_response/J123.json`：`J1=-4.74137552781, J2=-0.551065848062, J3=+0.0924718915723 meV`，联合rank3、条件数2.3701。总响应拟合残差0.483863%，s31/s13/SD分别0.404441%/0.403198%/0.683078%，均小于2%，因此 `full14_luo_A.json` 已自动切换。十项输入哈希、六份力矩、联合拟合重读及Luo模型身份复核通过。这个一致性条件不是独立DFT验收；无新SCF、无重拟合、无s31/SD重算。

`../data/current14/s31_response/J123.json` 保留为先前的 s31 单胞读数与复用证据。
它复用已完成的 s31 实际方向正负端点，不新增 SCF 或电子态对角化。
`FULL14_EQUIVALENCE.json` 记录当前 full14 的势核、密度核折叠到 s31 后与完成端点所用核的逐点比较，
同时检查两个四分量 FM 场；若完成端点的大体积输入场仍在本机，也复查其一致性。
未回传到 25_li 的端点场明确记为不可复查，不伪造一次新的电子态核查。
两核及 FM 一致时，s31 续算就是当前 full14 在该超胞中的响应。

约定 `H=+sum_<ij> J_ij S_i.S_j`，每键一次、`S=1.5`，负 J 为铁磁。
读入 `TORQUES.json` 的 Ry/rad 三项总力矩，以实际单位自旋和旋转轴构造交换设计：
`tau_i = -S^2 sum_j J_ij axis_i · (e_i × e_j)`。
中央差分之后拟合三个独立 J，转换 `1 Ry = 13605.693122994 meV`。
完整六站点、有限数值、三项求和、单位方向、占据态与文件身份都在读取时检查。
不再使用旧 `direct_torque_*` 的最近 EF 行，也不把 band-only torque 当作完整力矩。

```bash
cd /home2/25_li/CrI3
python3 frozen_magnon/fit_native_torque.py fit \
  data/current14/three_cell_response/torque_input.json --output /tmp/full14_J123_check.json
```

结果为 **J1--J3 截断模型下的条件交换参数，physical_release=false**。
s31 的设计 rank=3；SD 单独仍为 rank=2，只识别 J1+J3、J2。
SD 参考精度续算不改变该几何事实。更远邻、非 Heisenberg 项、完整 Hessian、步长/尺寸收敛及高阶顶点尚未由本次接口工作验收。
本接口不从谐波 J 推断独立 Phi4，不从此无 SOC 交换读出自动附加 Luo 的各向异性 A。

## 已有最新 Luo 基准

`run_luo_full_hp.py` 和 `fortran/cal_luo_full_hp_scattering.f90` 复用 2026-08-04 最新通用实现。
保留原有四通道、排除 f=i/Dirac 简并基底转换、固定零温色散和有限温度自洽。
2026-09-27 修复了四阶顶点复相位/符号及有序末态计数，继续使用既有自洽方程和明确保留的Bose近似。读取入口：明确 Hamiltonian 符号，H=-sum 的历史 J 自动换号，
无约定/重复/非有限 J 拒绝，rank<3 的 JSON 拒绝，spin_S 必须匹配；A_meV 必须显式给出。

- `benchmarks/config_fixed.json`：固定零温色散；
- `benchmarks/config_self_consistent.json`：自洽有限温度，独立 24×24 自洽网格；
- `benchmarks/path60/`：已有 60×60、6.7 K、Γ–K–M–Γ 专用计算入口与逐 q 续算实现。

三者使用修正后的 Luo 参数 `J=(-2.9377,-0.6205,+0.1562) meV`、`A=0.3627 meV`。
2025 正文的 J2 正号为漏写负号。本次不重新运行大网格散射率批次。

```bash
# 25_li 使用已安装 ifort；本地可用 --compiler gfortran。
python3 frozen_magnon/run_luo_full_hp.py band \
  frozen_magnon/benchmarks/config_fixed.json --work-dir /tmp/luo_fixed --compiler ifort
python3 frozen_magnon/run_luo_full_hp.py band \
  frozen_magnon/benchmarks/config_self_consistent.json --work-dir /tmp/luo_finite_t --compiler ifort
```

散射使用同一命令的 `run` 子命令；大网格需计算节点，不在登录节点启动。
path60 的既有入口：先 `python3 benchmarks/path60/run_path60.py prepare --work-dir WORK`，
再用 `ifort -O3 -qopenmp benchmarks/path60/cal_luo_path60.f90 -o WORK/cal_luo_path60`，
在 WORK 内运行可执行文件，最后调用同脚本的 `merge`、`plot`。
这里未复制旧机器专用调度脚本。

要让当前 full14 的 J 进入下游，配置的 `j_source_file` 指向上述 `J123.json`，
`spin_S=1.5`，并显式声明所采用的 `A_meV` 和温度色散模式。
使用文献 A 时，结果只能称为“full14 交换 + 文献各向异性”的条件模型。
`full14_luo_A.json` 已提供这一显式条件配置；`prepare` 可读取模型而不启动散射。

```bash
python3 frozen_magnon/run_luo_full_hp.py prepare \
  frozen_magnon/full14_luo_A.json --mesh 6 --work-dir /tmp/full14_luo_model
```

## 最小回归检查

```bash
PYTHONDONTWRITEBYTECODE=1 python3 frozen_magnon/tests/check_interfaces.py
OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s frozen_magnon/tests -p 'test_*.py' -v
```

第一项检查符号、Ry 三项读取、已知 J 的 rank3 恢复、SD rank2 拒绝和异常输入；
第二项对两个 Fortran 核逐键独立展开 HP 四阶项，比较复振幅并检验交换/共轭/归一化，随后用归一化升降算符与热平均检验末态计数，并检查既有 FGR 排除规则的数值回归。需要 NumPy 和 gfortran，不依赖 pytest。检查通过只说明实现接口和所测数值行为，不提升物理发布状态。

新核计算使用新的工作目录。首次 `prepare`、`run` 或 `band` 冻结 J来源/源码身份、J/S/A、温度、展宽、色散、自洽设置、网格及H2/H4/FGR约定，后续不重写该身份。`prepare --mesh` 必须与后续计算一致；显式 `compile` 需要先 `prepare`。旧目录不会自动补造新版身份。不同网格的色散和散射率放在不同目录，例如 `band24` 与 `rates6`。

配置可显式设置 `xi_meV`（默认仍为 `abs(J1)/100`），以及恰好四个互不相同的正温度 `temperatures_K`；实际输入和元数据读取同一组值。通用分片必须在同一目录依次运行；同时写同一目录会被互斥检查拒绝。合并还核对冻结温度、mesh与实际q坐标。路径入口通过 `--temperature` 从配置四温中明确选择一个温度；模型或数值条件改变需新目录。通用 `run` 遍历全BZ外q，大网格高对称路径计算应使用路径入口。

通用入口按源码 SHA-256 决定是否重编译。直接手动调用 Fortran 可执行文件不经过Python身份/互斥检查，因此使用这种低层方式时必须保持冻结输入并使用新目录，不能复用旧 `path_points`。本轮保护正常入口，未增加抵抗外部手动篡改的系统。

## 固定积分网格上的任意外部 q（2026-09-29）

path60 Fortran 核的 `luo_scatt.in` 新增可选 `external_q=qx qy`（倒格基矢分数坐标）。使用时设置 `path_only=.false.`、空 `checkpoint_dir`；每次只计算这个外部 q，内部积分仍由 `nqx/nqy` 决定。结构因子、能带、自洽和 HP 顶点继续使用现有公式，非整格动量通过一套平移网格精确满足动量守恒，没有插值。省略该项时行为不变。

这一低层选项用于 K 邻域极限诊断；常规 `run_path60.py` 的整格路径 CLI 保持不变。可复现运行入口为本地 `deployment/frozen_magnon_current_20260926/luo_K_limit_20260929/run_limit.py`，25_li 对应 `runs/luo_K_limit_20260929/`。它冻结源、二进制、模型输入和每个外 q 的结果。旧数据继续绑定旧源快照，无须因采样接口扩展而重算。

`tests/test_offgrid_path.py` 检查新旧整格结果、任意外 q 与独立实空间 HP 完整小网格 FGR、周期性和非法输入拒绝。K 极限的积分收敛与物理解释另见本批报告，接口测试不替代这些结论。

## 完整分支 sunset 与谱线展宽（2026-10-09）

配置显式设置 `"scattering_definition": "full_sunset"` 可使用新谱线定义；省略时仍为 `legacy_four_channel`，旧四通道输出用于原有回归。两个 Fortran 核及通用/路径 Python 入口均支持此配置。新模式保留现有 J1–J3+A 的正规序领先 HP 四阶顶点，枚举外支、热浴支及两条末态支的全部 16 个有序组合，统一使用末态计数因子 1/2 和完整 Bose 因子 `n_bath*(1+n_final1+n_final2)-n_final1*n_final2`。同一 Fock 态及简并旁观粒子转换沿用排除规则。

定义参考 Eto 等的 [PRX 16, 021053 (2026)](https://journals.aps.org/prx/abstract/10.1103/tbh2-jq9r) 式 (12)、(15)、(29)，完整 Bose 因子也可核对 Banerjee/Humeniuk 的 [PRB 112, 094407 (2025)](https://journals.aps.org/prb/abstract/10.1103/llz4-5k6g) 式 (8)、(9) 与附录 B。这里补齐的是给定 H4 的分支求和与 on-shell 虚部，未实现 PRX 的 T-matrix、频率依赖实部或所有有限 S 修正。

输出 `full_sunset_spectral.tsv` 含 31 列：`T, iq, qx, qy, qz, E_alpha, E_beta, R_alpha, R_beta, HWHM_alpha, HWHM_beta, FWHM_alpha, FWHM_beta, tau_population_alpha, tau_population_beta`，随后为 `c0000` 到 `c1111` 的 16 个部分净率。组合顺序为 `(created1, created2, bath, external)`，0=beta、1=alpha。能量/宽度单位 meV，率为 s⁻¹，寿命为 s。`R_net=2*(-Im Sigma_R)/hbar`，`HWHM=hbar*R_net/2`，`FWHM=hbar*R_net`，`tau_population=1/R_net`；相干振幅寿命为其两倍。这里的寿命是该自能近似下的单模 population 等效寿命，不是完整碰撞算符的输运或集体弛豫时间。

路径模式生成 `full_sunset_spectral_pathN_raw.tsv`，`merge` 检查完整行、温度、坐标和寿命并添加路径位置列；不使用旧四通道的逐点检查点。原四通道绘图入口会拒绝新 schema，避免误读 HWHM/FWHM。有限数值 `xi_meV` 是能量守恒 delta 函数的积分展宽，不能当作物理 HWHM；少量 off-shell 部分净率可能为负，应检查总率及其随 xi 缩小的残余。

CrBr3 的少量点生产与实验对照见 `../deployment/crbr3_full_sunset_20261009_r01/REPORT.md`。该批在新目录保存源快照、模型身份、输出哈希、原参数/有效 gap 参数的区别及网格/xi 收敛证据。

```bash
python3 frozen_magnon/tests/test_full_sunset.py
python3 deployment/crbr3_full_sunset_20261009_r01/run_production.py --mesh 180 --width 0.0037125
python3 deployment/crbr3_full_sunset_20261009_r01/analyze.py
```

独立实空间展开检验两个核的全部 16 个复顶点，小网格参考检验部分率、分支总率及 HWHM/FWHM/lifetime 的单位和因子。生产验收仅覆盖所列几个 q/温度及条件 J+A 模型，仍为 `physical_release=false`，不验证独立 ansatz Phi4 或 CrI3 的实际寿命。
