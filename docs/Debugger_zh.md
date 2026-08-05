# NPUIR 可视化调试器

`npuir-interp-debug` 把解释器的一次执行记录为可回放的 JSON Lines 会话。它与
`npuir-interp` 使用完全相同的调度器、pipe 延迟提交和同步模型；区别只是每个解释
步骤结束后会额外采集状态，因此调试器看到的就是实际执行状态，而不是二次推导的
近似结果。

## 生成调试会话

```bash
build/bin/npuir-interp-debug kernel.mlir \
  --sched=lazy \
  --args=a.npy,b.npy,zeros \
  --debug-output=npuir-debug.jsonl
```

debug 二进制未显式指定 `--debug-output` 时默认写入当前目录的
`npuir-debug.jsonl`。普通的 `npuir-interp` 不会默认记录；需要时也可以显式传入
同一个参数。

调试器记录：

- 本步刚执行的 op、每个 core 的下一条 op、状态和 vector clock；
- 每条 S/V/M/MTE/FIX pipe 上尚未退休的 effect、token 和 resident access；
- effect 的读写 arena 及半开地址区间；
- core 的阻塞原因、等待的 flag/barrier/lock、已发布 flag 和 barrier 到达状态；
- 每个 arena 的 allocation、生命周期、高水位和实际字节值。

内存使用“初始内容 + 每步写入补丁”保存。记录器只检查本步实际提交的写区间以及
新分配的区间，不会在每条指令后扫描整个 GM。因而前端可以重建任意地址在任意步骤
的内容，同时避免为每一步复制整块 tensor。

## 打开界面

直接用浏览器打开 [`tools/debug-ui/index.html`](../tools/debug-ui/index.html)，点击
“加载调试会话”并选择 JSONL 文件即可。文件由浏览器本地读取，不会上传。

也可以启动一个本地静态服务器：

```bash
cd tools/debug-ui
python3 -m http.server 8000
```

然后访问 `http://localhost:8000`。界面支持：

- 上一步、下一步、首尾跳转、自动播放和时间线拖动；
- 查看本步完整 operation 文本、执行前输入 SSA 值和执行后输出 SSA 值；
- 在完整 HIVM IR 窗口中自动定位并高亮当前执行行；
- 选择 core，观察每条 pipe 中任务的入队和退休；
- 查看当前同步等待、intra-core flag、cross-core flag 和 barrier；
- 选择 arena 和 allocation，按地址浏览字节；本步发生变化的字节会高亮；
- 键盘左右方向键单步，空格播放或暂停。

界面中的核拓扑固定为**一个物理核内的切片**：纯 Vector kernel 显示一个 AIV，
MIX kernel 显示一个 AIC 和一个 AIV；当 `--sub-block-num=2` 时，Vector 侧显示为
`AIV.0`、`AIV.1` 两个 sub-vector 执行单元。`--block-dim` 表示 Triton launch 的
program instance 数量，并不表示界面中存在同样多个物理 AIV。解释器仍执行全部
program 以保证最终 tensor 的数学验证完整，但记录文件和回放页只包含
`--debug-core` 所在 program 的核状态、向量时钟和片上内存；GM 等共享地址空间仍
展示完整结果。界面始终提供 AIC 和 AIV 观察卡；纯 Vector kernel 没有实际 AIC
entry，此时 AIC 明确显示为 `inactive` 且 pipe 均为 idle，不会伪造 AIC 任务。MIX
kernel 中该卡自动显示真实 AIC 的指令、pipe、同步和时钟状态。

目标硬件 pipe 只展示 `S`、`V`、`M`、`MTE1`、`MTE2`、`MTE3` 和 `FIX`，并按
AIV/AIC 实际拥有的执行 pipe 进一步裁剪。HIVM 枚举中的 `MTE4`、`MTE5`、`V2`
是兼容/扩展枚举，不作为本调试目标上的独立硬件 pipe 展示或执行。同步面板会同时
显示本步刚执行的 `set_flag`/`wait_flag`、仍排在生产 pipe 上的 pending token，以及
已经发布但尚未消费的 semaphore。

## 从 DSL 用例一键运行

每个 `test/dsl_e2e/cases/` 下的 case 文件都可以直接执行。先按
[`test/dsl_e2e/README.md`](../test/dsl_e2e/README.md) 安装 Triton Ascend 发行包并
激活 venv，然后运行 Flash Attention：

```bash
python test/dsl_e2e/cases/flash_attention.py --debug --serve
```

该命令会依次完成：

1. Triton DSL → TTAdapter；
2. TTAdapter → post-GraphSyncSolver HIVM；
3. 用 `npuir-interp-debug` 执行 lazy 调度并记录 session；
4. 与 inorder 输出做逐字节比较，并与 case 的数学参考值比较；
5. 执行一次 seed 固定的 fuzz 调度；
6. 启动本地服务器并自动打开已经加载 session 的调试页面。

产物默认保存在 `build/debug/flash_attention/`。Flash Attention 会启动 32 个 Triton
program instance，而不是在调试模型中创建 32 个物理 AIV。默认记录 program 0 的
`AIV#0.0` 逐步事件，同时保留全局状态和最终内存，以把日志控制在可交互的大小。
使用 `--debug-core=AIV#3.0` 切换到 program 3，或使用
`--debug-core=all` 记录所有 core。无图形环境时可添加 `--no-open`，只打印页面 URL。

## JSONL 协议

首行是 `meta`，协议号当前为 `npuir-interp-debug/v1`，硬件模型标识为
`ascend-single-core/v2`。随后是按 `sequence` 排序的 `state`，末行是 `finish`。
失败和死锁也会正常写出 `finish`，因此失败现场可以完整回放。前端会拒绝缺少硬件
模型标识的旧会话，E2E 脚本也会提示重新构建 `npuir-interp-debug`。

每个 `state` 都包含完整控制状态，但内存只包含 `memory_patches`：

```json
{"event":"state","sequence":4,"executed":{"name":"hivm.hir.wait_flag"},
 "cores":[...],"cross_flags":[],"barriers":[],"arenas":[...],
 "memory_patches":[{"arena":3,"offset":0,"bytes":"000001000200"}]}
```

`bytes` 是按地址递增的十六进制原始字节。消费者从 initial state 开始依次应用补丁，
即可重建某个 state 的内存。协议使用 JSON Lines 而不是单个大 JSON 数组，使脚本可以
边执行边读取，也使异常终止前已落盘的状态仍然有效。

## 当前边界

首版是确定性的离线回放调试器，不会暂停正在运行的解释器进程。它适合定位 pipe
提交、同步和数值变化发生在哪一步。后续若需要交互式断点，可在同一状态模型之上增加
进程控制协议，不需要更改前端对 session state 的理解。
