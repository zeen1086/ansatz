# Ansatz 自旋与力矩

公共参数入口是根目录 `parameter`；目录和编译说明见 [根 README](../README.md)。

Fortran 主链为 `cal_Hsk` → `combine` → `cal_torque_hf`。`cal_Hsk_trace` 记录指定
R 向量和矩阵元的通道贡献；目标、通道掩码/来源和调制参数都由 `parameter` 设置。
`band`、`cal_totE_occu` 读取相同模型与网格参数。

`torque_s13.py` 与 `response_matrix.py` 保留原生 PW 的 ansatz 诊断入口。
`fit_native_torque.py` 校验三项力矩数据和输入哈希后提取条件 J。
full14 核的拟合与三项力矩分别在 `native_pw/physical_validation_20260924_r01/`
的 `fit14.py`、`torque_sd.py`，其余模块是现有依赖。

原生三项力矩为 `-∫rho_out dV + ∫rho_in dV + ∫(V−V_native) drho_in`。
源码中的数据路径对应外部原生工作树；本仓库没有训练场、种子、波函数或已完成端点。
`check` 为离线接口检查，`prepare`/`endpoint`/`fit` 会创建计算产物，应在准备匹配输入
并取得对应计算授权后运行。历史提交和旧 README 的任务授权不随代码迁移继承。

J 拟合输出保留 `physical_release=false`，响应残差通过也不代表独立 DFT 验收。
