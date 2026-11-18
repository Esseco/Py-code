# Process_Struct

<!-- Manual project documentation -->

结构分析工具，采用延迟导入；单独使用某项功能不要求安装所有可选依赖。默认 py1，通道分析使用已有 CCNB/CAVD 环境。

## 按功能选择

| 功能 | 公开入口 | 输入 → 输出 | 独立模板 |
|---|---|---|---|
| 扩胞矩阵 | get_supercell_matrices | 结构、扩胞倍数、a/b/c 方向 → 不等价矩阵及逐项短基矢约化矩阵 | 见下方调用示例 |
| 理论容量 | theoretical_specific_capacity | 化学式、迁移离子 → mAh/g | [capacity.py](examples/capacity.py) |
| Wyckoff 分组 | get_wyckoff_sites | Structure、容差 → 分组列表 | [wyckoff_sites.py](examples/wyckoff_sites.py) |
| 八面体畸变 | analyze_octahedra_distortion / Octahedron | 有序氧化物结构 → 局域畸变记录 | [octahedra.py](examples/octahedra.py) |
| 结构特征 | StructureFeatureExtractor | 结构 → global/local 字典（Na/O 比值缺失时为 NaN） | [structure_features.py](examples/structure_features.py) |
| 统一结构特征 | extract_features / FeatureExtractor | 结构（Structure/Atoms/文件）+ 特征集选择 → FeatureResult（扁平 values、位点表、metadata） | [feature_sets.py](examples/feature_sets.py) |
| 几何通道连通 | analyze_percolation | 结构、迁移元素、探针半径 → 通道结果 | [percolation.py](examples/percolation.py) |
| 通道位点导出示例 | analyze_percolation + 示例导出 | 结构、迁移元素、探针半径 → 标记 CIF、间隙与瓶颈 CSV | [percolation_sites.py](examples/percolation_sites.py) |
| 化学容量约束 | estimate_redox_capacity / estimate_mobile_ion_window | 化学组成及约束 → 容量与离子窗口 | [子项目说明](Chemical_Capacity_Constraints/README.md) |
| CCNB 位点容量 | estimate_spatial_capacity | 结构、离子、空隙半径、位点间距 → 可行位点数与 mAh/g | [位点容量说明](SPATIAL_CAPACITY.md) |
| 轨迹转换 | save_traj_xyz / traj_to_info | ASE 轨迹 → 文件/信息 | [参数索引](API.md) |
| 形成能 | calc_formation_energy | 组成和能量 DataFrame → 派生结果 | [参数索引](API.md) |

## 使用方式

模板顶部集中设置参数；从仓库根目录独立运行：

```text
python -m Process_Struct.examples.capacity
python -m Process_Struct.examples.wyckoff_sites
python -m Process_Struct.examples.octahedra
python -m Process_Struct.examples.structure_features
python -m Process_Struct.examples.feature_sets
python -m Process_Struct.examples.percolation
python -m Process_Struct.examples.percolation_sites
```

除容量模板外，需要先修改输入路径。模板导入不会读取结构或启动分析。现有函数和参数保持兼容；每项可单独调用，不要求统一 CLI。

## 科学语义与限制

### 扩胞矩阵

```python
from pymatgen.core import Structure
from Process_Struct import get_supercell_matrices

structure = Structure.from_file("POSCAR")
matrices = get_supercell_matrices(structure, size=8, dimensions="ab")
all_matrices = matrices["all"]
recommended = matrices["recommended"]

expanded = structure.copy()
expanded.make_supercell(recommended[0])
```

`size` 是相对输入晶胞的体积/原子数倍数；`dimensions` 是晶格方向，支持 `a/b/c/ab/ac/bc/abc`，与书写顺序无关，未选方向保持原样。返回值仅有 `all` 和 `recommended` 两组整数 3×3 NumPy 矩阵，按索引一一对应；约定 `new_cell = matrix @ old_cell`，可直接传入 pymatgen 或 ASE 的扩胞接口。也支持 ASE Atoms 输入。

`all` 由 icet `enumerate_supercells(..., niggli_reduce=False)` 枚举，按输入结构的元素、坐标和对称性去重；建议使用原胞，不自动转换原胞，不考虑磁矩或无序占位。`recommended[i]` 用 ASE Minkowski reduction 仅在所选方向内约化 `all[i]`，保持同一个超晶胞，可能含负整数；它不是所有候选中唯一最优的矩阵，也不保证接近正方形/立方体。原始 PBC 标志由 `dimensions` 在输入副本上覆盖，仅用于本次枚举和约化。

默认 py1，依赖 icet、ASE、NumPy；输入 Structure 时另需 pymatgen。`symprec=1e-5` Å 可调。无随机性、不修改输入、不写文件，重复调用返回新结果；枚举数会随倍数增长，三维大倍数需注意耗时和内存。已用小型合成立方/三斜晶胞核对数量、矩阵等价性和方向限制，未验证大规模枚举性能。

参考：[icet 枚举接口](https://icet.materialsmodeling.org/_modules/icet/tools/structure_enumeration.html)、[ASE Minkowski reduction](https://ase-lib.org/ase/geometry.html#ase.geometry.minkowski_reduce)。

通道位点导出示例的参数、单位、输出、重复运行策略和验证记录见 [PERCOLATION_EXAMPLE.md](PERCOLATION_EXAMPLE.md)。在已有 ccnb 环境执行；标记 CIF 的 He/Ne 表示几何点位，完整周期连接保留在 NET/CSV 中。

- 理论容量按化学式中全部迁移离子可转移的计量计算，不是实验可逆容量；摩尔质量包含完整化学式。原函数的 pymatgen 单位对象附带 amu 标签，模板用 float 展示计算公式对应的 mAh/g。
- Wyckoff 分组依赖 symprec（Å）和 angle_tolerance（度）；返回包含 PeriodicSite，不能直接用普通 json.dump 写出整个对象。
- 八面体分析默认 O 配体及 Fe/Mn 中心；局部失败可能打印错误并返回缺少畸变字段的记录，需要检查完整性。
- StructureFeatureExtractor 支持非 Na/O 体系：缺少 Na 或 O 时 Na_ratio 记为 NaN，不再抛 KeyError；octahedra 组依赖 octadist，缺失时该组全为 NaN（ccnb 环境即如此）；特征组使用列表，如 ["lattice"]，避免传单个组名字符串。
- 通道维数是几何连通性，不代表迁移能垒。CCNB/CAVD 依赖、氧化态和探针模型需要单独确认。
- 形成能/电压必须明确每原子、每化学式或离子归一化，不能混合不同能量基准。

### 统一结构特征

```python
from Process_Struct import extract_features, FeatureExtractor

r = extract_features(structure, sets=["geometry", "cnn"])
r.values      # {"geom.a": 5.64, "cnn.cn_mean": 6.0, ...}
r.names       # 与 values 同序的列名
r.to_frame()  # 单行 DataFrame
r.warnings    # CrystalNN/pymatgen 的原始提示

both = FeatureExtractor(sets=["geometry", "cnn"], level="both")(structure)
both.sites    # 逐位点记录，可直接转 to_site_frame()

model = "E:/Py-Code/MLIP_model/mace/mace-omat-0-medium.model"
mlip = FeatureExtractor(sets=["mlip"], encoder="mace", model_path=model)
vector = mlip(atoms)  # 模型只在构造期加载一次；默认两层描述符拼接（2×128）后逐原子平均

layer1 = FeatureExtractor(
    sets=["mlip"], encoder="mace", model_path=model, mlip_layers=1
)(atoms)  # 只取第一层（128 维）
by_element = FeatureExtractor(
    sets=["mlip"], encoder="mace", model_path=model, mlip_aggregate="element_mean"
)(atoms)  # 每个元素一段：mlip.Na.d000…、mlip.Cl.d000…
```

四类特征集用前缀区分：`geometry`（`geom.`，默认，委托 StructureFeatureExtractor）、`cnn`（`cnn.`，pymatgen CrystalNN 配位数统计与逐位点 CN）、`cnn_fp`（`cnn_fp.`，matminer CrystalNN 指纹，可选依赖）、`mlip`（`mlip.`，MACE 逐层描述符或 M3GNet 平均池化 embedding）。`mlip` 额外支持 `mlip_layers`（`1`/`2`/`"all"`，默认 `"all"`）与 `mlip_aggregate`（`"atom_mean"`/`"element_mean"`，默认 `"atom_mean"`）；M3GNet 没有逐层描述符，只接受默认值。`sets` 是粗选，`include`/`exclude` 用 glob（如 `"geom.lattice.*"`）细化到叶子；输入接受 pymatgen Structure、ASE Atoms 或结构文件路径。默认只开 `geometry`，毫秒级；`cnn`/`cnn_fp`/`mlip` 必须显式点选。

`extract_many` 逐结构隔离失败：坏结构进入 `status="error"` 与 `error` 列，不中断整批；`return_sites=True` 时返回结构表与位点表。非数值位点字段（如 `O_neighbors` 的邻居索引列表）不进入位点表，并在 `warnings` 中记录。`metadata` 保存特征集、结构摘要、模型 sha256、matminer 版本与是否带氧化态。缺依赖在构造期抛明确 ImportError，不静默降级。

## 功能边界

见 [BOUNDARIES.md](BOUNDARIES.md)。通用去重是与 Process_Vasp 的迁移候选；完整算法比较前不移动代码。octadist 保持第三方项目独立。

## 验证

容量模板已实运行；Wyckoff 使用合成结构核对分组。percolation_sites 示例在 ccnb 环境用 NaFeF3 核对了 0.5 Å 连通、1.0 Å 无通道情况，以及 CIF 回读和周期连接 CSV。其他示例的语法与导入检查不代表真实材料或模型计算已验证。统一结构特征用 `python -m unittest Process_Struct.test_features` 实测：ccnb 20 项通过（6 项模型用例按设计跳过），py-mace 20 项通过（MACE 4 项真跑、M3GNet 2 项跳过），m3gnet_tf 20 项通过（M3GNet 2 项真跑、MACE 4 项跳过）；NaCl（Fm-3m，8 位点）配位数为 6，LiCoO2 的 `geom.Na_ratio` 为 NaN，`cnn_fp` 在 py1 返回 48 个结构列。MACE 部分用真实 checkpoint `MLIP_model/mace/mace-omat-0-medium.model` 跑了第一层（128 列）、第二层（128 列且与第一层不同）、两层拼接（256 列）与逐元素平均（2 元素 × 256 列）；单独一层是用 `node_feats` 切片实现的，已与官方 `get_descriptors(num_layers=1/2)` 做 `np.allclose` 交叉验证。M3GNet 用真实 checkpoint `MLIP_model/M3gnet` 在 m3gnet_tf 环境跑通：64 列 `mlip.d000…`、两次调用逐位一致，走的是 m3gnet 0.2.4 下的均值池化节点特征回退路径（`metadata["providers"]["mlip"]["descriptor"] == "mean_pooled_node_features"`），并拒绝 `mlip_layers`/`mlip_aggregate`。`Process_MLIP/tests/test_mace_descriptors.py` 的 11 项切片/归一化单元测试在 py-mace 通过，用假张量避免加载权重。真实签名见 [API.md](API.md)，具体任务记录见 [TASK_TEMPLATE.md](TASK_TEMPLATE.md)。

## 通用去重合并已完成

实现统一到 Process_Struct/deduplication.py；Process_Vasp 保留同名兼容入口，层状氧化物与表面直接复用公共实现。deduplicate 保留首次出现的结构副本，deduplicate_dict 跨所有键去重并省略空组，deduplicate_df 按能量升序保留指定数量；采用默认 StructureMatcher，可能匹配平移/缩放及等价晶胞，不能理解为逐坐标完全相等。

列表模板：[deduplicate_structures.py](examples/deduplicate_structures.py)。DataFrame 原接口使用标签索引，建议输入索引唯一；n_keep 使用正整数。本轮保留这些旧行为，没有改匹配容差或能量基准。
