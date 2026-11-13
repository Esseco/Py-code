# MLIP NEB：从结构和父位点生成迁移势垒

`generate_neb_pairs(structure, parent_structure, ...)` 复用 `Process_Vasp.generate_neb_endpoints` 生成离子–空位端点；`run_mlip_neb(...)` 继续用现有 MLIP calculator 进行固定晶胞端点弛豫、IDPP 插值、普通 NEB 和 CI-NEB。输入接受 ASE Atoms、pymatgen Structure 或 ASE 可读结构文件；需要 pymatgen 与 scipy。

父结构是**同一晶胞中的候选移动离子位点目录**，必须包含目标结构全部移动离子位点，骨架元素、数量和位置必须匹配。对 P2 Na1/Na2，候选位点可以互斥，不应将父位点目录直接解释为物理上同时满占的结构。`match_tol` 和 `overlap_tol` 单位 Å；后者仅按现有端点函数排除近重叠，不保证路径合理，Na1/Na2 的相互排斥应另外检查。随机选择由 `seed` 控制；有限路径采样不等于穷尽所有迁移环境。

```python
from Process_MLIP import model_config, run_mlip_neb

# 只启用一个模型；路径和环境按实际模型填写。
models = {
    "mace": model_config("mace", model_path="/models/mace-omat-0-medium.model",
                         head=None, default_dtype="float32", device="cuda",
                         enable_symmetrix=True, symmetrix_use_kokkos=True),
    "sevennet": model_config("sevennet", model_path="/models/checkpoint_7net_nano_6.0.pth",
                             device="cuda", enable_flash=True),
    "orb": model_config("orb", model_path="/models/orb-v3-conservative.ckpt",
                        device="cuda", model_name="orb-v3-conservative-inf-omat"),
}
result = run_mlip_neb(
    "deintercalated.vasp", "parent_site_catalog.vasp", "neb_mace",
    model=models["mace"], mobile_ion="Na", random_num=3, seed=42,
    max_distance=3.5, intermediate_images=3,
    endpoint_fmax=0.1, endpoint_steps=2000, endpoint_optimizer="LBFGS",
    neb_fmax=0.1, neb_steps=500, cineb_fmax=0.05, cineb_steps=500,
)
print(result["summary_csv"])
```

默认端点在固定晶胞中弛豫并要求收敛；若端点坍缩到同一移动离子位点，该路径记为失败。NEB 使用一份共享 calculator、串行图像；MACE/ORB/7Net 的 head、精度、modal、加速开关由既有 `model_config` 和 calculator 工厂处理，加载失败不会切换模型。已有 calculator 可通过 `calculator=` 传入。

输出目录必须不存在：`run.json` 记录参数，`path_XXX/` 含端点、弛豫和 NEB/CI-NEB 轨迹及 `result.json`，`summary.csv` 汇总状态和势垒。`forward_barrier_eV = max(E_i)-E_initial`、`reverse_barrier_eV = max(E_i)-E_final`；非收敛路径保留数值和 `not_converged`，不能当作收敛势垒。单条路径失败会记录原因并继续处理其他路径，不提供断点续算。

跨模型比较时可先调用 `generate_neb_pairs`，筛选并保存端点，再向每次 `run_mlip_neb` 传入 `endpoint_pairs=[(initial, final), ...]`；端点可为结构对象或文件路径。运行前会检查初态对应目标结构、原子顺序与晶胞一致、每对仅移动一个指定离子且终点对应父位点，避免不同模型随机选到不同迁移对。

这些是固定构型、固定晶胞下的单路径势垒，不等同于多温度扩散拟合的有效 Arrhenius Ea；路径覆盖、端点局域极小值、图像数量和势模型精度仍需独立验证。
