# electerm HPC 配置

`tools/electerm_hpc_profiles.json` 记录 electerm 连接档案。每个档案把 HPC 主机和必须遵守的专属 skill 绑定在一起；默认档案是 `bj-hpc`。

任务文件统一分开保存：本地使用 `E:\\0-electerm_mcp\\tasks\\<task-name>`，BJ-HPC 使用 `/data/home/lichaoyue/0-Codex_task/<task-name>`。默认任务目录保持为 `README.md + submit.sh`，只有需要时才增加 `config/`、`input/`、`logs/`、`results/`。`E:\\Py-Code` 与 `/data/home/lichaoyue/Process_code` 只作为项目代码仓库，不作为任务工作目录。

任务交互结束后只清理已确认无用途的压缩包、重复上传包、测试副本、空目录、临时日志和失败传输残留。现有任务默认暂缓，不移动、不覆盖、不重命名、不清理其输入、日志、结果或检查点。删除前必须列出精确路径、大小和原因；远端删除仍需按 electerm 审批执行。用户明确指定本地或远端路径时，以用户指定为准。

在任何 electerm MCP 操作前，先用预检入口解析档案并读取规则：

```powershell
python tools/electerm_hpc.py --profile bj-hpc --host hpc3.chinahpc.com
```

预检会检查：

- 选择的档案存在；
- 绑定的 skill 文件存在、可读且非空；
- 提供 `--host` 时，实际连接主机与档案主机完全匹配。

没有对应档案或 skill 时应停止，不得猜测规则或直接调用 electerm。切换到另一台 HPC 后重新运行预检。通用的 `electerm-operations/SKILL.md` 仍然必须先读取，档案中的 skill 是该 HPC 的附加规则。

新增 HPC 时，在 `profiles` 中增加一个唯一名称，并填写 `display_name`、`host`、`skill`；同时补充该 HPC 的专属 skill 文件。`repository`、`task_root` 和 `environment` 只记录该服务器的运行约束，不会自动上传文件、安装依赖或提交作业。

## Agent 层强制模式

`tools/electerm_hpc.py` 现在提供短期预检凭证：

```python
from tools.electerm_hpc import preflight_hpc, require_preflight

receipt = preflight_hpc(
    "bj-hpc",
    connected_host="hpc3.chinahpc.com",
    tab_id="<tab-id>",
)
# 调用每个 electerm MCP 工具前都执行：
context = require_preflight(
    receipt,
    profile="bj-hpc",
    connected_host="hpc3.chinahpc.com",
    tab_id="<tab-id>",
)
```

凭证默认 30 分钟有效，并绑定 skill 的 SHA-256。主机、tab、skill 或档案变化会使凭证失效；缺少凭证、凭证过期或校验失败时，Agent 必须拒绝调用 electerm MCP。连接新 tab 时，先用预检覆盖连接目标；拿到 tab ID 后重新生成绑定该 tab 的凭证，再执行后续操作。
