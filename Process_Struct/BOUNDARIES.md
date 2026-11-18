# 功能合并与拆分建议

| 当前功能 | 建议 | 状态 |
|---|---|---|
| Process_Vasp.structure 的 deduplicate/deduplicate_dict/deduplicate_df | 通用匹配去重迁入 Process_Struct，VASP 保留兼容导出 | 候选，未移动 |
| Na1_to_Nax、层序变换 | 比较 Process_LayeredOxide 的现有实现，体系相关逻辑归入该项目 | 候选，未合并 |
| Get_Oct_distortion 与 Get_structfeatures_all 的八面体逻辑 | 比较配体、周期边界、指标定义及缺失值，再抽公共内核 | 候选，不能因名称相近直接合并 |
| 理论计量容量与 Chemical_Capacity_Constraints | 保留两个入口，前者计量公式，后者化学约束估计 | 不直接合并 |
| percolation 与 channel_codes_optimized | 保留通用分析函数，批处理项目将来复用它；先比较输出和探针定义 | 待核对 |
| octadist | 保持第三方独立，仅调用其科学计算接口 | 保留 |

本轮只处理 Process_Struct 的说明与功能模板，不改变科学实现或其他项目接口。

## 通用去重合并已完成

实现统一到 Process_Struct/deduplication.py；Process_Vasp 保留同名兼容入口，层状氧化物与表面直接复用公共实现。deduplicate 保留首次出现的结构副本，deduplicate_dict 跨所有键去重并省略空组，deduplicate_df 按能量升序保留指定数量；采用默认 StructureMatcher，可能匹配平移/缩放及等价晶胞，不能理解为逐坐标完全相等。

列表模板：[deduplicate_structures.py](examples/deduplicate_structures.py)。DataFrame 原接口使用标签索引，建议输入索引唯一；n_keep 使用正整数。本轮保留这些旧行为，没有改匹配容差或能量基准。

## 统一结构特征模块（features）

统一入口只做聚合与命名，不复制科学实现：`geometry` 委托 `StructureFeatureExtractor`，`mlip` 包装 `Process_MLIP.direct_sampling.build_encoder`，两者既有签名不变。明确不做的事：不输出 MLIP 弛豫标量（每原子能量/最大力/体积变化留在 Process_MLIP），不纳入 Process_LayeredOxide 的层间距/结构因子，不把 `cnn`/`cnn_fp`/`mlip` 设为默认（默认只有毫秒级的 `geometry`）。

已知限制与坑：

- CrystalNN 缺氧化态或半径时会静默退化为共价/原子半径（warning 原文：`CrystalNN: cannot locate an appropriate radius, covalent or atomic radii will be used, this can lead to non-optimal results.`）。该提示与 `No oxidation states specified on sites! ...` 一并原文写入 `FeatureResult.metadata["warnings"]`，不要丢弃；要严格可比就传带氧化态的 `Species` 结构。
- 配位数异常先查测试结构是否合理：岩盐型 NaCl 必须用 `Fm-3m` + 8 位点（Cl 在 (0,0,0) 与 (0.5,0,0)）；把 Cl 放在 (0.5,0.5,0.5) 是 CsCl 型（Na–Cl 4.884 Å 超出 cutoff，CN=0），2 原子 (0.5,0,0) 胞是 1D 链（CN=2）。这不是 pymatgen 缺陷，不要改库。
- `octahedra` 组依赖 octadist；ccnb 环境 `_OC_OK = False`，该组 14 项全为 NaN，属既有行为而非本模块丢数据。
- `cnn_fp` 依赖 matminer，仅 py1（0.10.1）已安装，其他环境构造期抛 ImportError（消息含当前解释器路径）。仓库不新增 matminer 安装依赖。
- `mlip` 的编码器是 callable（`__call__(atoms)`），不是 `get_embedding`；模型在 `FeatureExtractor` 构造期加载一次，批量请复用对象。
- MACE 逐层描述符不是官方独立接口：`get_descriptors(num_layers=n)` 只给累积 stack，第二层由 `node_feats` 的 l=0 分量切片得到（layer1 = 前 128，layer2 = `[512:640]`）。已在 `mace 0.3.14` + `mace-omat-0-medium.model`（`num_interactions=2`、`dim_per_layer=128`）上与官方 `num_layers=1/2` 做过 `np.allclose` 交叉验证；换 MACE 大版本或换模型结构后必须重跑该交叉验证，切片位置可能变。
- `layers` 只对 MACE 有意义；M3GNet 没有逐层描述符，传非 `"all"` 的 layers 会在构造期抛 ValueError。`aggregate="element_mean"` 同理仅 MACE 支持，列名为 `<元素>.d000…`（统一入口再加 `mlip.` 前缀）。
- MACE 需要 `mace` + `torch`（本机只有 `py-mace` 环境：mace 0.3.14 / torch 2.9.1+cpu）；M3GNet 需要 `m3gnet` + `maml` + TensorFlow（本机只有 `m3gnet_tf`：tf 2.10.0 / m3gnet 0.2.4 / maml 2025.4.3）。两个编码器都已用真实 checkpoint 跑通（MACE：`MLIP_model/mace/mace-omat-0-medium.model`；M3GNet：`MLIP_model/M3gnet`，64 维、走 `mean_pooled_node_features` 回退路径）。py1 的 Keras 3 读不了老 M3GNet checkpoint，会抛 `RuntimeError`（maml 的 “File format not supported” 被包装），别当成本模块的 bug。ccnb 连 `Process_MLIP` 都 import 不了（`Process_MLIP/__init__.py:22` 的 `from .neb import ...` 需要 `ase.mep`，ccnb 的 ase 3.15 没有），属既有环境问题，与本模块无关。
- 输入 POSCAR/CONTCAR/`.vasp`/`.poscar` 走 `Poscar.from_file`，氧化态会丢失（CrystalNN 因此可能给半径退化 warning）；CIF 走 `Structure.from_file`，能否保留氧化态取决于文件内容。
- 位点表只保留数值型字段；`geom.O_neighbors` 是邻居索引列表，会被丢弃并在 `warnings` 记录 `dropped non-numeric site fields: geom.O_neighbors`，需要邻居明细请直接调 `StructureFeatureExtractor`。
- `Na_ratio` 在缺 Na 或 O 时为 NaN（`Get_structfeatures_all.py` 安全取值），非层状氧化物不再 KeyError，但该比值对非 Na 体系没有科学含义。
