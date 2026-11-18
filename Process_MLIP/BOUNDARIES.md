# 项目边界

- Process_AL_MC/relax.py 的单结构 MACE 入口已兼容导出本项目实现；旧路径保留。
- Process_AL_MC 的 MC 排序、采样池、委员会和检查点仍属于采样项目，不并入弛豫工具。
- Process_Chgnet 的模型弛豫可成为独立 calculator 适配器；训练数据转换仍保留原项目。旧导入需先修复。
- Process_face 负责生成原子约束，本项目负责在弛豫中保留并执行约束，不重复实现固定层算法。
- 不为统一接口改变模型能量基准、默认 head、dtype 或晶胞优化方法。不同返回结构需经过兼容性设计后再整合。

保留公开接口、模型基准和各入口的默认参数。内部共享优化实现与校验，不强行统一不同返回结构。

## MACE 合并已完成

MACE/SevenNet 的模型包装分别保留在 mace_relax.py / sevennet_relax.py，
单结构结果适配集中在 _relaxation.py，ASE 优化由 MLIPRelaxer 统一执行。
Process_AL_MC/relax.py 仍兼容导出同一 MACE 函数；MC 类的委员会弛豫保持独立。

## NEP/GPUMD 接入边界

NEP calculator、数据转换、训练/微调和原生 GPUMD MD 均归 `Process_MLIP`。`Process_MD` 只委托 NEP calculator；不复制 `nep.in`、GPUMD 轨迹或训练算法。CPU/GPU 后端必须显式选择，长时间 GPU MD 与 ASE `run_md` 的轨迹/报告语义分开。

## 主动学习采样边界

`al_sampling.py` 只做「给定非负权重后的选点」：不生成权重、不归一化特征或权重、不去重、不判断结构是否有效。QBC/能量/应力等指标的融合由调用者完成。

- 距离是特征空间欧氏距离，候选与已算构型必须来自同一编码器和同一尺度；维度不一致、NaN/inf、权重缺失或负数在校验期直接报错。
- kmeans++/farthest-first 在候选不足时选满候选数；NMS 可能少于 `n_select`，不补齐。重复特征距离为 0，不会被距离规则选中，这是数据问题，采样器不负责去重。
- 全零权重、重复特征、距离全零时走确定性回退（最高权重，并列取候选顺序靠前者），不会死循环，但说明权重或特征没有区分度。
- 复杂度：初始最近距离 `O(n·m·d)`（按候选分块），每次选点增量更新 `O(n·d)`，不预先构建 `n×n` 距离矩阵。10,000 候选 × 1,000 已算 × 128–256 维、选 100 个实测 0.7–1.5 s（py1，CPU）。
