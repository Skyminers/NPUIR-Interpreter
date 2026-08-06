# NPUIR 可视化调试器

`npuir-interp-debug` 把解释器的一次执行记录为可回放的 JSON Lines 会话。它与
`npuir-interp` 使用完全相同的调度器、pipe 延迟提交和同步模型；区别只是每个解释
步骤结束后会额外采集状态，因此调试器看到的就是实际执行状态，而不是二次推导的
近似结果。

## 统一 Web 入口

在仓库根目录运行：

```bash
python3 tools/interpreter_web.py
```

浏览器会打开 `http://127.0.0.1:8000/tools/debug-ui/index.html`。这个入口包含两个功能：

- **运行验证**：读取或粘贴一份 TTAdapter/HIVM MLIR，配置参数、调度和 AIC/AIV
  拓扑后执行；页面展示下降状态、退出状态、耗时、输出 NPY 的 dtype/shape/数值预览，
  并保留本次日志；
- **日志重放**：加载任意 `debug.jsonl`，或在运行结束后点击“重放本次运行”，进入逐步
  pipe、同步、SSA、IR 和内存调试工作台。

“数学期望值”用于验证真实公式，而不是只比较两种调度结果。它按扁平顺序比较指定
输出 arg 的前 N 个元素，并使用页面填写的 `atol`/`rtol`。留空时，页面只以
Interpreter 自身的同步、竞争、死锁、越界检查和进程退出码判断成功。所有请求只发送
到本机 `127.0.0.1`，后端不经过 shell，也不接受任意命令行参数或输出路径。

参数既可以继续使用 `zeros`、`arange`、标量和本机 NPY 路径，也可以点击“选择参数
NPY”。页面会把所选文件映射为 `@input0`、`@input1` 等本次运行专用 token，并自动
放入参数列表；可以继续编辑列表，为输出 buffer 添加 `zeros`。上传文件只写入当前
`build/web-runs/<run-id>/inputs/`，不会作为任意服务器路径使用。

页面会在编辑器上方主动标识 IR 阶段。输入已经带有 `hacc.entry`、
`hivm.func_core_type` 和 `hivm.module_core_type` 时直接执行；检测到 TTAdapter、tensor、
Linalg、GPU 等未完全下降形式时，会自动调用
[`test/lower_ttadapter_for_interp.py`](../test/lower_ttadapter_for_interp.py) 和
`bishengir-compile`，生成 post-GraphSyncSolver HIVM 后再交给 Interpreter。结果区会
单独展示下降状态和耗时，并允许下载生成的 `kernel.lowered.mlir`，或把它重新载入左侧
编辑器继续检查和运行。

自定义端口、二进制和运行超时时间：

```bash
python3 tools/interpreter_web.py \
  --port 8127 \
  --interpreter build/bin/npuir-interp-debug \
  --compiler build/bin/bishengir-compile \
  --timeout 120
```

每次运行的源码、输出 NPY 和 JSONL 默认保存在 `build/web-runs/<run-id>/`。

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

直接用浏览器打开 [`tools/debug-ui/index.html`](../tools/debug-ui/index.html)，切换到
“日志重放”并选择 JSONL 文件也可以离线使用。文件由浏览器本地读取，不会上传；此时
“运行验证”会明确显示后端未连接。

也可以启动一个本地静态服务器：

```bash
cd tools/debug-ui
python3 -m http.server 8000
```

然后访问 `http://localhost:8000`。界面支持：

- 上一步、下一步、首尾跳转、自动播放和时间线拖动；
- 查看本步完整 operation 文本、执行前输入 SSA 值和执行后输出 SSA 值；
- 在 AIC/AIV 两个完整 HIVM IR 窗口中分别定位并高亮各自执行行；
- 选择 core，观察每条 pipe 中任务的入队和退休；
- 查看当前同步等待、intra-core flag、cross-core flag 和 barrier；
- 选择 arena 和 allocation，按地址浏览字节；本步发生变化的字节会高亮；
- 键盘左右方向键单步，空格播放或暂停。

界面中的核拓扑固定为**一个物理核内的切片**：纯 Vector kernel 显示一个 AIV，
MIX kernel 显示一个 AIC 和一个代表性 AIV。`--sub-block-num=2` 时两个 Vector lane
仍在解释器中参与调度、同步和数值计算，但前端只展示 lane 0，避免重复状态占用空间。
`--block-dim` 表示 Triton launch 的 program instance 数量，并不表示界面中存在同样
多个物理 AIV。解释器仍执行全部 program 以保证最终 tensor 的数学验证完整，但记录文件和回放页只包含
`--debug-core` 所在 program 的核状态、向量时钟和片上内存；GM 等共享地址空间仍
展示完整结果。界面始终提供 AIC 和 AIV 观察卡；纯 Vector kernel 没有实际 AIC
entry，此时 AIC 明确显示为 `inactive` 且 pipe 均为 idle，不会伪造 AIC 任务。MIX
kernel 中该卡自动显示真实 AIC 的指令、pipe、同步和时钟状态。

目标硬件 pipe 只展示 `S`、`V`、`M`、`MTE1`、`MTE2`、`MTE3` 和 `FIX`，并按
AIV/AIC 实际拥有的执行 pipe 进一步裁剪。HIVM 枚举中的 `MTE4`、`MTE5`、`V2`
是兼容/扩展枚举，不作为本调试目标上的独立硬件 pipe 展示或执行。同步面板会同时
显示本步刚执行的 `set_flag`/`wait_flag`、仍排在生产 pipe 上的 pending token，以及
已经发布但尚未消费的 semaphore。

## 最小 AIC/AIV 逐步示例

[`test/debug/mix-walkthrough.mlir`](../test/debug/mix-walkthrough.mlir) 只做一件事：AIC
向共享 GM 写入 `42` 并发布跨核 Flag；AIV 等待该 Flag，读取 `42`，计算
`42 + 1 = 43`，再写入输出。先在仓库根目录生成 session：

```bash
mkdir -p build/debug/mix-walkthrough
build/bin/npuir-interp-debug test/debug/mix-walkthrough.mlir \
  --sched=lazy \
  --args=zeros,zeros \
  --debug-core=AIV#0.0 \
  --debug-output=build/debug/mix-walkthrough/mix-walkthrough.debug.jsonl \
  --out=build/debug/mix-walkthrough/mix-walkthrough.
python3 -m http.server 8000
```

打开以下地址：

```text
http://127.0.0.1:8000/tools/debug-ui/index.html?session=/build/debug/mix-walkthrough/mix-walkthrough.debug.jsonl
```

按事件号逐项核对：

| 事件 | 应看到的内容 | 真实语义 |
|---:|---|---|
| 2 | AIV 为“等待 Flag”，AIV IR 高亮 `sync_block_wait` | AIV 在 AIC 发布前不能读取 mailbox |
| 4 | 选择 AIC 后，`PIPE_S` 有一个 `memref.store`；GM `%arg0[0]` 为 `42` | AIC 已写入 mailbox，但跨核 Flag 尚未发布 |
| 5 | 本步为 `sync_block_set`，AIV 恢复“可执行”，同步面板显示 AIC↔AIV Flag 代次 1 | AIC 正式发布数据可见性 |
| 11 | 本步 SSA 输入为 `42`、`1`，输出为 `43` | AIV 完成唯一的算术运算 |
| 12 | GM `%arg1[0]` 为 `43` | AIV 将最终结果写回 |

该用例的 lit 检查还会直接读取 JSONL，验证阻塞对象、Pipe 任务、Flag 代次、SSA 值和
内存补丁；最终 `.npy` 也必须为 `[43, 0, 0, 0]`。因此页面显示与解释器真实状态由
同一组断言交叉验证。

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

当前运行入口以一次完整执行为单位，调试工作台随后重放持久化日志；它不会暂停正在
运行的解释器进程。该模式适合先确认数值与运行期检查，再定位 pipe 提交、同步和数值
变化发生在哪一步。后续若需要交互式断点，可在同一状态模型之上增加进程控制协议，
不需要更改前端对 session state 的理解。
