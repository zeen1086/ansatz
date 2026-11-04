# CrI3 ansatz

仅保留 ansatz 势拟合、Wannier/RVR、自旋哈密顿量和原生 PW 三项力矩代码。
仓库不包含计算数据；`physical_release = false`。

```text
parameter                 全局参数，TOML 格式
parameters.py             Python 读取及 Fortran 参数模块生成
configure.py              生成输入文件，不执行计算
build.py                  编译入口，不提交作业
1_scf/                    磁性 SCF、势导出模板和原子结构
2_band/                   磁性能带模板
3_band_no_mag/            无磁 SCF/能带模板
4_potential/              势分割、对称化、R_vec
  fit/                    六通道线性/调制势拟合及检查
5_wannier/                Wannier90、pw2wannier、wfck2r 模板
  make_wan/               Wannier 函数及权重生成
6_RVR/                    RVR 生成及合并
  speedup/                分块 ZGEMM 候选实现
7_spin/                   H_sk、自旋能带、占据/能量和力矩
  native_pw/              原生 PW 三项力矩及 full14 依赖代码
  torque_s13.py           s13 端点及三胞条件 J 拟合
  response_matrix.py      已有模型的三胞响应诊断
  fit_native_torque.py    三项力矩读取与条件 J 拟合
```

旧目录与 frozen-magnon 模型已移除。原先放在该模型目录中的
`fit_native_torque.py` 是 ansatz 力矩依赖，已移至 `7_spin/`。
分析程序输出 J 和响应判据，不再改写下游磁振子模型。

参数只在根目录的 `parameter` 中设置。Python 入口运行时读取它；Fortran 使用
`configure.py` 生成的 `.build/parameter_mod.f90`。**修改参数后应重新生成输入并重编译
受影响的 Fortran 程序**，已有二进制不会自动改变。可用 `ANSATZ_PARAMETER` 指定另一个
完整参数文件；相对路径相对于所选参数文件所在目录解析。

`parameter` 包含路径、Wannier 轨道数/截断范围、实空间与 k 网格、晶格常数、费米能、
展宽、势拟合正则化/收敛设置、自旋通道控制、角步长，以及原生 PW 数据契约。
数学常数、二进制格式标识、固定站点/通道几何和自检用例仍保留在源代码中。
原子坐标保存在 `1_scf/structure.in`，Wannier 投影几何保存在对应 `.template`；
这些是输入结构数据，改变它们需核对固定 CrI3 站点映射。

当前公共 Wannier 自旋组装支持 `n_func=18/36`，势拟合使用六通道几何；
原生 PW full14 是独立的 14 核路线。两者没有因共享参数文件而建立数据或物理等价性。
默认值取自原代码及输入示例，并非一套已匹配验证的数据。原仓库未提供展宽列表，
示例 `broadening=[0.002]` 使用旧控制参数的 eta；运行时应按实际数据设置。

使用 Python 3.11+（标准库自带 TOML），或 Python 3.10 加已安装的 `tomli`。
原生 PW 和测试使用 NumPy。无需新增环境。

```bash
python3 configure.py --check
python3 configure.py
python3 build.py 4_potential/make_R_vec.f90 4_potential/fit/fit_com_ortho_cold.f90
python3 build.py 7_spin/cal_Hsk.f90 7_spin/combine.f90 7_spin/cal_torque_hf.f90
```

GNU 默认使用 `-fopenmp -llapack -lblas`。Intel MPI/MKL 示例：

```bash
python3 build.py --fc mpiifort --flags='-O2 -qopenmp -mkl' --libs='-mkl' \
  5_wannier/make_wan/make_wan_fun_mpi.f90 6_RVR/cal_RVR_split.f90 \
  6_RVR/speedup/cal_RVR_zgemm.f90 7_spin/band.f90
```

Wannier/RVR 的行号现在是运行参数，无需替换源码再编译：

```bash
cd 6_RVR
mkdir -p tmp
mpirun -np 2 ./cal_RVR_split.x 1
# ZGEMM 候选也从 6_RVR 目录运行，保持相同 tmp/ 和输出位置：
mpirun -np 2 ./speedup/cal_RVR_zgemm.x 1
```

`make_wan_fun_mpi` 和参考 `cal_RVR_split` 使用主进程/工作进程布局，需要至少两个
MPI 进程。每个 RVR 行写 `tmp/B_general_<行号>`，之后由 `6_RVR/combine.x` 合并。
`5_wannier/make_wan/submit.sh` 在已有资源分配中运行各轨道，不调用调度器。
`7_spin/run_spin.sh` 编译并运行自旋程序，也不提交作业。

生成的 `*.in`、`*.win`、`.build/`、二进制和数据均不入 Git。
`configure.py` 只覆盖带生成标记的输入文件，遇到用户自建的同名输入会报错。
运行前需提供赝势、各阶段匹配的保存态及训练场；2_band 使用磁性 SCF 保存态，
3_band_no_mag 使用独立无磁保存态，5_wannier 的示例是无 SOC 的标量轨道路线。
配置生成不会自动复制密度/波函数、拟合势或提交任何计算。

RVR 还需同一 Wannier 分支的 `wan_data`、`cal_TF`、`V_HYC_dp`、`R_vec` 和
`V_func_tmp`。自旋合并读取 `parameter` 的 `super_h_dir`（默认
`6_RVR/super_H`），其中需要匹配的 `hr_HYC_*`、`ir_vec_HYC` 等产物；
本代码快照没有包含其生成器或数据。不要混用不同分支的同名文件。
`spin.in` 是每次计算的自旋纹理输入，文件/数据格式不因全局配置迁移而改变。

原生 PW 的三个 `build.sh` 读取同一个 `parameter` 中的 QE、MPI、MKL 和修改版
`libpw_cr_only.a` 路径，并保留拒绝覆盖已有二进制的保护。
该修改版 QE 库和原生数据集由外部提供；保留的日期模块名用于内部依赖与数据追溯，
不表示授权重启历史计算。原生模块还依赖原工作树的相对数据结构。

离线检查：

```bash
python3 -B test_configuration.py
python3 -B 7_spin/torque_s13.py check
python3 -B 7_spin/response_matrix.py check
```

测试包含参数传播与错误拒绝、生成文件保护、生成模块编译，以及小型合成 Fortran
记录的裁剪、势分割和 Wannier 权重检查；后者需先编译对应的三个程序。
这些检查验证代码接口与局部数值行为，不替代 QE→Wannier→RVR→力矩的端到端验收。
