# Ansatz 代码入口

## 当前任务：s31+s13+SD 联合 J

用户已授权补齐当前 full14 的两个 s13 固定势端点。入口 `torque_s13.py` 复用下述 native 三项力矩核心，只适配 s13 网格、48个k点和路径；没有重写物理核。62550 准备已完成，62551/62552 为正负端点，62553 自动联合现有 s31、SD。提交记录在 `../records/FULL14_S13_SUBMISSION_20260926.json`。

输出已完成：`../data/current14/three_cell_response/J123.json`，`J=(-4.74137552781,-0.551065848062,+0.0924718915723) meV`，H=+sum、S=1.5、rank3。各胞响应残差均≤2%，`../frozen_magnon/full14_luo_A.json` 已切换到联合结果。仅复用旧 s13 的自旋几何和FM种子，不复用其不同候选的力矩。无新 SCF、无重拟合、无 s31/SD 重算，physical_release=false。

## 当前 14 核拟合

入口：`fit14.py` → `native_pw/physical_validation_20260924_r01/fit14.py`。
它使用 XML 实际自旋方向和配对训练场；当前输出在 `../data/current14/`。

同目录的 `coordinates.py` 负责方向与站点映射；`projection_consistency.py` 负责相同 G 表示的比较。
可以用 `python3 ansatz/fit14.py check` 检查既有设计读取。`fit` 会创建新输出，当前已存核受拒绝覆盖保护。

## 当前完整三项力矩

入口：`torque_sd.py` → `native_pw/physical_validation_20260924_r01/torque_sd.py`。
它复用 `coordinate_layer1_issp_20260923_r01/run.py` 的核心计算：

1. `density_ansatz_harris_r02/pp_variational.x`：由预测密度生成原生非线性 HXC。
2. `frozen_pw_projection_scan_r01/pw_frozen_wfc.x`：固定势电子态对角化。
3. `full_k_density_readout_r01/pp_variational.x`：读出占据态密度。
4. 合成 `-∫rho_out dV + ∫rho_in dV + ∫(V−V_native) drho_in`。

对应 Fortran 源码、build.sh 和共享修改版 libpw 一并保存。运行依赖本机已有 Intel MPI/MKL；重编译仍使用 `/home/chen/software/qe-7.1`，这些系统软件不重复复制。
`native_pw/` 中若干日期目录是被 import 的原模块；不要分别运行它们的旧测试 main()。
已有端点结果是只读参考。为了精简新工作树，原生 PW 端点复算所需的 49 个 SD FM 波函数仍在旧目录，路径见 `../records/OPTIONAL_INPUTS.json`；这里保存代码、种子和已完成的读数。当前获授权的 s13 补齐使用另一组48个FM波函数，复制到 `../runs/full14_s13_20260926/` 各端点私有目录，其源路径和核身份在该目录 INPUTS.json 固定。

## 已有 Wannier ansatz 主流程

`wannier/` 保留原有分阶段代码：

```text
1_scf → 2_band / 2_band_no_mag → 4_potential_2x2_test3 → 5_nscf
      → 6_THE_5x5_torque/1_make_RVR
      → 6_THE_5x5_torque/9_spin/H_sk
```

主要文件：`V_split.f90`/拟合代码、Wannier 生成与网格转换代码、`cal_RVR_split.f90`、
`cal_Hsk.f90`、`combine.f90`、`cal_torque_hf.f90`。
Wannier 数据来自同一原始 `5_nscf`，通过内部相对链接提供。保留的是已有主流程源代码，而不是一个已经用当前 14 核重新验收的计算结果。

**当前完整 14 核尚未在这里生成匹配的 V_func_tmp/B_general。** `THE.in` 中旧 n_func 等模型参数属于原程序配置，本次只调整目录路径，没有擅自修改物理参数。不要直接提交旧 run.sh 来“补齐”接口。

## Frozen-magnon

当前入口为 `../frozen_magnon/README.md`，使用已有最新 Luo full-HP 核和 native 三项力矩读取。
当前 full14 的势核、密度核折叠到 s31 后与已完成的实际方向续算核一致到机器精度，FM 场完全相同；可复用该续算的正负力矩，无需重跑。
当前联合 J1/J2/J3 在 `../data/current14/three_cell_response/J123.json`（rank=3）；原 s31 单胞读数和全网格等价证据保留在 `../data/current14/s31_response/`。
SD 自身仍只识别两组合；不要把 SD 限制泛化到完整 ansatz。当前 J 为有限角度、J1--J3 截断模型下的条件读数，physical_release=false。
旧 Gaussian/uinv 路线已移除；当前只补两个 s13 固定势电子态/三项力矩，没有新 SCF、Phi4 或散射率。
