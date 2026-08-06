# 文档索引

| 文档 | 语言 | 适合读者 | 内容 |
|---|---|---|---|
| [DSL E2E 环境](../test/dsl_e2e/README.md) | 中文 | DSL/E2E 开发者 | Triton Ascend 发行包、venv、离线安装与验证 |
| [使用指南](Usage_zh.md) | 中文 | 工具使用者 | 构建、参数、调度模式、诊断、CI 接入与排障 |
| [运行与可视化调试](Debugger_zh.md) | 中文 | Kernel 开发者 | Web 运行验证、数值确认、逐步回放 pipe、同步等待和内存值 |
| [架构与实现](Architecture_zh.md) | 中文 | 开发者 | 延迟提交、流水线、同步、内存、检查器、数值精度与代码结构 |
| [Architecture and implementation](Architecture.md) | English | Developers | Runtime model, checks, memory, numerics, coverage and testing |
| [精度扫描](../test/Precision/README.md) | English | 数值路径开发者 | f16、bf16、f8 与整数路径的逐位验证方法 |

第一次使用建议先阅读[使用指南](Usage_zh.md)；准备修改调度、内存、同步或 op handler
时，再阅读[架构与实现](Architecture_zh.md)。
