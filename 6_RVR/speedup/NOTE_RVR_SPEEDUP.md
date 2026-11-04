# 分块 ZGEMM RVR 候选

`cal_RVR_zgemm.f90` 保留既有 packing 和分块 ZGEMM 实现。
它与 `../cal_RVR_split.f90` 共享根目录 `parameter` 的模型、路径与截断设置，
通过命令行第一个参数选择行号。块大小由 `rvr_chunk` 和 `rvr_channel_block` 设置。
两种实现应从 `6_RVR/` 运行，以共享同一组输入路径和 `tmp/` 输出约定。

这仍是候选实现：求和次序与标量参考不同，迁移后的实际输入数值等价性和性能
需要在同一 Wannier 数据上另行验证。原始性能报告保存在仓库迁移前的 Git 历史中；
其旧路径、硬件和时间不能直接视为本次重排后的测试结果。
