# AGENTS.md

## 工作方式
- 整个 `E:\Py-Code` 都可以改，但每次提权（danger-full-access / 越过沙箱权限写入）前，先用一两句话说明要写哪些文件、为什么，再动手。
- 动手新写之前先搜一遍仓库里有没有现成实现（尤其被要求"检查有没有某类函数"时，通常已经有了），能复用就别重写，先说清结论再改。
- 改完源码后重跑 `python tools/project_catalog.py`，刷新各项目 `API.md`、`TASK_TEMPLATE.md` 和 `docs/project_catalog.json`；直接跑若报 Non-UTF-8 是假错误，改用 `runpy` 执行。
- 往 `Process_MLIP` 公开 API 加名字时，必须同步 `tests/public_api.json`，否则 `tests/test_calculators.py` 严格比对 `__all__` 会直接红。
- 解释权限、环境这类问题时用大白话：先说现在什么状况、卡在哪、有哪几个选择，别堆术语。
