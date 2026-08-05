# `npuir-interp` —— NPUIR (HIVM) CPU 解释器

[返回文档索引](README.md) · [使用指南](Usage_zh.md) · [English](Architecture.md)

> 在没有昇腾硬件的情况下，把编译后的 memref 形态 HIVM IR 跑在主机上。

---

## 1. 这个工具解决什么问题

`npuir-interp` 有两个用途，**按优先级排列**：

1. **同步正确性检查（不可替代的价值）**
   验证编译器插入的 flag、barrier、lock 到底够不够。这件事没有别的工具能做：
   - lit 用例只能检查 IR **文本**，检查不了运行期语义；
   - triton-ascend 的 `ascend_interpreter.py` 工作在 Triton Python 层，它的
     `sync_block_set` / `sync_block_wait` 全是 no-op，**结构上**看不见 HIVM 同步；
   - 真机上漏同步表现为"偶发结果不对"，极难定位。

2. **数值正确性参考**
   无硬件跑通 kernel，与 numpy/torch golden 比对。

`lazy` 同步检查要求输入位于 HIVM 本核 `GraphSyncSolver` / `InjectSync` **之后**。
更早阶段的 dump 可能已经有跨核 `sync_block_*`，但本核 `set_flag` / `wait_flag` 尚未
物化；这类 IR 应使用 `--sched=inorder`，或先运行本核同步 pass 再使用 `lazy`。

```bash
npuir-interp kernel.mlir --sched=lazy --args=a.npy,b.npy,zeros --out=out.
```

---

## 2. 核心设计判断：延迟提交模型

**这一节决定这个工具是调试工具还是玩具。**

真机语义：`hivm.hir.load` 发射到 MTE2 流水就返回了。数据什么时候真正落到 UB，
由 `set_flag(MTE2→V)` / `wait_flag(MTE2→V)` 这一对指令决定。

如果解释器按程序序同步执行每条 op，就人为引入了一个**比真机强得多的顺序保证**。
在那个模型下，**无论同步插没插、插对没插，结果都是对的** —— 工具退化成一个跑得
很慢的 numpy。

所以：**带流水线（pipe）的 op 执行时不立即生效**，而是把一个延迟的 `Effect`
压进对应 pipe 的队列，只在硬件会排空队列的地方才真正提交。

漏同步于是会以三种方式暴露：

| 现象 | 说明 |
|---|---|
| `MISSING SYNC` 诊断 | 直接点名两条 op、两条流水线，以及该插的 `set_flag` 对 |
| 毒值（NaN / `0xCD`） | 生产者从未提交，消费者读到毒值 |
| `DATA RACE` 报告 | 两侧在不同核上时 |

### 2.1 三种调度模式

| `--sched=` | 行为 | 用途 |
|---|---|---|
| `inorder` | effect 立即提交；每执行一条 op 后轮换可运行核 | 确定性的纯数值 golden。**按构造原理，发现不了任何同步问题** |
| `lazy`（默认） | effect 尽可能晚提交；每执行一条 op 后轮换可运行核 | 确定性地模拟多核并行推进的检查模式 |
| `fuzz --seed=N` | lazy + 随机选核 + 1～8 条 op 的随机时间片 | 找脆弱的同步 |

**对 pass 后 IR，差分测试是最强的自动判据**：同一份 IR 分别用 `inorder` 和
`lazy` 跑，输出一致说明已经物化的同步是充分的，不一致说明有同步缺失。多个用例的
RUN 行直接用 `cmp` 做这件事。

### 2.2 冲刷（flush）规则

| IR | 语义 |
|---|---|
| 带 pipe `P` 的 op | effect 入队 `P`；同 pipe 内 FIFO 退休 |
| 带 `MacroOpPipeTrait<P1, P2>` 的 op（`mmadL1`） | 拆成两段，分别入队 `P1` / `P2`，见 §2.4 |
| `memref.load` / `store`、`llvm.load` / `store` | PIPE_S：数据立即生效，但留一个**驻留标记**在 `S` 队列上，见 §2.3 |
| `set_flag[src, dst, id]` | 在 `src` 队尾打一个 token；**队列排到 token 时** flag 才真正置起 |
| `wait_flag[src, dst, id]` | 排空 `src` 直到那个 token，合并发布的向量时钟；没有 token 则阻塞 |
| `pipe_barrier[P]` | 排空 `P`（`PIPE_ALL` 排空全部） |
| `sync_block_set[core, tpipe, pipe] flag=N` | **只**排空 `tpipe`，跨核信号量 +1，发布时钟 |
| `sync_block_wait[...] flag=N` | 信号量 −1（为 0 则阻塞），合并发布的时钟 |
| `sync_block[<MODE>, id]` | 按 mode 的参与者集合 rendezvous；每个到达的核排空全部 pipe |
| `sync_block_lock` / `unlock` | 基于 `lock_var` 的互斥；`lock_var == blockIdx` 才放行 |
| kernel `return` | 排空全部 —— 否则最后一批写会凭空消失 |

> `sync_block_set` **只**排空 `tpipe` 是刻意的。在别的 pipe 上写数据的生产者对
> 消费者仍然不可见 —— 这正是我们要抓的 bug。

### 2.3 PIPE_S：标量单元也是一条流水线

标量单元不是"流水线之外"的东西，它就是 `PIPE_S`。InjectSync 对
`test_mem_memref_load_store` 这类 IR 插的正是这三对 flag：

```mlir
hivm.hir.load ins(%gm) outs(%ub)              // MTE2 写 UB
hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]   // ← 标量才能读
hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
scf.for ... { memref.load %ub[%i] ; ... ; memref.store %v, %ub[%i] }
hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]   // ← MTE3 才能读
hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
hivm.hir.store ins(%ub) outs(%gm)
```

但 `PIPE_S` 和别的 pipe 有一点本质区别：**标量单元是发射单元**，它顺序执行自己的
指令，不可能自己排在自己的队列里等。`memref.load` 的结果下一条 `arith` op 就要用，
延迟提交在语义上是错的。

所以 PIPE_S 走的是**驻留标记（resident marker）**模型：

1. 数据搬运和影子内存记录**立即完成**（`completesNow`）；
2. 但访问的字节范围作为一个**没有 commit、不再二次记录**的 `Effect` 留在 `S`
   队列上；
3. 别的 pipe 上的 op 发射时照常扫描 `S` 队列 —— 于是"标量写了 UB 却没有
   `set_flag[PIPE_S, PIPE_MTE3]` 就交给 MTE3"会被点名；
4. `wait_flag[S, dst]` / `pipe_barrier` / barrier / `return` 把标记退休掉。

这个模型是对称的：`set_flag[MTE2, S]`（RAW，DMA→标量）、`set_flag[S, MTE3]`
（RAW，标量→DMA）、`set_flag[S, MTE2]`（WAR，标量读完 DMA 才能重填）三个方向都能
检出，分别对应 `sync/missing-scalar-flags.mlir` 和 `sync/missing-scalar-war.mlir`。

**标记合并**：一个标量循环逐元素写 buffer，会一个元素压一个标记。所以相邻/重叠、
同一个 op、同一方向、同一 arena 的标记会合并成一个区间（只向队尾回看 8 项，且不跨
token —— 跨过去会让 `wait_flag` 提前把后面的访问退休掉）。合并规则由
`unittests/PipeEngineTest.cpp` 直接覆盖。合并不掉的散乱访问有一个
4096 条的上限，超过就丢弃并**打一次 warning**：宁可漏报也不能凭空加宽区间去误报。

### 2.4 宏 op 的两段建模

`mmadL1` 带 `MacroOpPipeTrait<PIPE_MTE1, PIPE_M>` —— 它是**两条硬件指令**：MTE1 把
L1 上的 A/B tile 搬进 L0A/L0B，然后 cube 在 L0 上做乘加写 L0C。

把它压成一条 `PIPE_M` effect 会同时错两次：

- `set_flag[PIPE_MTE1, PIPE_MTE2]`（"tile 已经搬完，L1 可以重填了"）匹配不到任何
  在飞的东西，于是一个**合法**的双缓冲 kernel 被报成漏同步；
- 反过来，真的漏了这对 flag 时，报告会指向 `PIPE_M`，给出的修复建议是错的。

所以两段分别入队：

| 段 | pipe | reads | writes | commit |
|---|---|---|---|---|
| staging | `getInPipe()` = MTE1 | A、B | — | 把 A/B 按 `real_m × real_k × real_n` 打包快照 |
| compute | `getOutPipe()` = M | C（累加时） | C | 用快照做乘加 |

快照是必须的，不是优化：MTE1 段退休之后 L1 上的 A 允许被重填，而 cube 看到的必须
仍然是它当初搬进 L0A 的那份。`functional/mmad-macro-pipes.mlir` 就是这个形状 ——
少了快照，第一次 mmad 会算到第二块 tile 上去。快照按原始字节存，一块 256×256 的
f16 tile 是 128 KB。

`stage()` 是幂等的：某条冲刷规则可能先排空 `M` 而没有先排空 `MTE1`，这时 compute
段自己补做 staging。两段属于同一个 `Operation *`，硬件本来就保证它们有序，所以
`checkPipeHazards` 跳过 `pending.op == op` 的自我冲突。

GraphSyncSolver 还可能把这两段周围的 flag 降到 `mmadL1` 的 7 个
`sync_related_args` 里：槽 0/1 在搬 A/B 前执行 MTE2→MTE1 wait，槽 2/3 在搬完后
发布 MTE1→MTE2 set，槽 4 控制内部 K-loop 双缓冲选择，槽 5/6 是提升到一组
mmad 外围的 M→MTE1 ping-pong event。解释器在 staging / compute 对应边界消费、发布
这些 event；若把它们当成无语义的普通操作数，就会把正确 IR 误报成漏同步，并在外围
最终 wait 处死锁。

### 2.5 跨核 flag 建模为 MIX 组 generation

跨核 flag 的 key 是 `(scope, tpipe, pipe, flag_id)`。`scope` 对 intra-block 和
inter-subblock 是 block 号，对 `INTER_BLOCK_SYNCHRONIZATION` 是全局。在 MIX kernel
里，方向由生产者和消费者的核类型决定：

- AIC set 发布一个 generation，同 scope 的每个 AIV sub-core 都能各消费一次；
- AIV set 提交该 sub-core 的一个信用，AIC wait 要等同 scope 的所有 AIV sub-core
  都提交后才放行，并一次消费每个 sub-core 的一个信用；
- 只有一种核类型的 kernel 仍使用普通计数信号量语义。

汇聚 wait 会合并所有 AIV 生产者的向量时钟，广播 generation 则把 AIC 时钟带给每个
消费者。这与 SIMD MIX kernel 降低后的组握手及循环复用同一 flag id 的形状一致；
但尚未用硬件规格独立核对 FFTS 的精确契约。

### 2.6 barrier 的 rearm 与唤醒

barrier 站点的 key 是 `(mode, flag id)`，**不含 op 身份** —— SplitMixKernel 之后
AIC/AIV 两半在不同函数里，需要 rendezvous 的是两个不同的 `Operation *`。
因此"只有部分核执行的 barrier"自然表现为到达数不一致，这正是
`AddControlFlowCondition` / `CloneOps` 这一族 pass 的漏改症状。

每个站点带一个 generation 计数器，循环里的 barrier 可以正确 rearm：一个核只有在
站点的 generation 超过它自己的计数时才放行，这样快核不会连过两轮而慢核一轮没过。

阻塞在 flag 上的核不会自己重新检测条件，所以**生产者会显式唤醒消费者**。
另外调度器有一层兜底：只要还有核在上一次重试之后产生过全局可见的进展，就不允许
判定死锁 —— 先把所有阻塞核唤醒再试一次。少一次唤醒就会把正确的 kernel 报成死锁，
而这是检查器最糟糕的失败方式：它会训练使用者忽略报告。

---

## 3. 检查能力

默认全开，用 `--check=sync,race,deadlock,oob` 选择子集。

### 3.1 `sync` —— 核内漏 flag

某条 op 触碰的字节还在**同一个核的另一条流水线**队列里没提交。同 pipe 重叠没问题
（FIFO 退休），跨 pipe 重叠而中间没有 flag 就是问题。

```
MISSING SYNC on AIV#0.0: PIPE_V op touches data still in flight on PIPE_MTE2
  in flight  PIPE_MTE2  hivm.hir.load @ add.mlir:4:5
  consumes   PIPE_V  hivm.hir.vadd @ add.mlir:8:5
  the two are unordered without a set_flag[PIPE_MTE2, PIPE_V, <id>] / wait_flag pair between them
```

### 3.2 `race` —— 跨核数据竞争

用**向量时钟**，不是 lockset。happens-before 边来自：

1. 同核程序序；
2. `set_flag` / `wait_flag` 配对；
3. `sync_block_set` / `sync_block_wait` 配对；
4. `sync_block` barrier 的参与者集合两两合并；
5. `sync_block_lock` 的前后持有者。

两个不同核对重叠字节的访问，若时钟不可比（既非 ≤ 也非 ≥）且至少一方是写，就报竞争。

```
DATA RACE on gm %arg1 +0 [0x40, 0x60)
  W  AIC#0    missing-c2v-flag.mlir:28:5  hivm.hir.store  vc=<4,0>
  R  AIV#0.0  missing-c2v-flag.mlir:40:5  hivm.hir.load   vc=<0,1>
  no happens-before edge between these two accesses
  nearest sync op: hivm.hir.sync_block @ kernel.mlir:180
                   (barrier only - carries no data-dependency flag)
```

最后两行"最近的同步点 + 它为什么不够"是让人立刻定位到哪个 pass 出问题的关键。

**UB 也检查**：一个 AIV 的多个 sub-vector core 共享 UB，MIX kernel 里 AIC 也共享。
不能因为"UB 是核内私有"就跳过。

有两类并发访问**刻意不报**：

- **裸指针 flag 便签本**：两侧都是 `inttoptr` 地址上的 `llvm.load`/`llvm.store`。
  那是编译器自己的 flag 区，并发访问是机制而不是 bug。`--check-raw-pointer-races`
  可以打开。
- **原子对原子**：`hivm.hir.store ... atomic = <add>` 之类是硬件读改写，彼此之间
  串行化。原子 vs 普通访问仍然会报。

同一条豁免规则也用在核内漏 flag 上：两侧都是裸指针时不报 `MISSING SYNC`，同样受
`--check-raw-pointer-races` 控制。标量访问本身照常参与跨核竞争检测（影子内存记录
是在发射时立即做的，见 §2.3）。

### 3.3 `deadlock`

没有核可运行且至少有一个核阻塞时报告。区分两种形态：

**(a) flag 等待环** —— 谁阻塞在哪个 flag 上，哪些核已经结束：

```
DEADLOCK: circular flag wait
  AIV#0.0 blocked at :21  sync_block_wait[PIPE_MTE3->PIPE_MTE2 flag=7 block=0]
  AIC#0 is Done (returned at :16)
```

**(b) barrier 到达数不一致** —— 期望的参与者 vs 实际到达的：

```
DEADLOCK: barrier ALL:1 arrival mismatch
  expected participants: AIC#0 AIV#0.0 (2 cores)
  arrived:  AIC#0 @ :20
  AIV#0.0 is Done (returned at :32) and never reached ALL:1
  hint: a barrier that only some cores execute is the classic symptom of a
        barrier cloned into a conditional region
```

### 3.4 `oob` —— 片上内存越界 / 超容量

片上内存池是**固定容量的 arena**，容量从 module 的 `dlti.target_system_spec` 读取
（可用 `--ub-size` 等覆盖）。放不下的分配、走出池子的地址都是错误，而不是悄悄踩坏
邻居。PlanMemory 烘焙好的偏移**照搬**，所以算错的偏移会撞到 arena 边界，而不会被
解释器自己的分配器悄悄"修正"掉。

> 越界访问**总是**致命的：那条访问根本没法执行，把诊断关掉只会变成静默失败。

### 3.5 layout 标签检查

`convert_layout` / `copy` 会校验生产者留下的 layout 标签和消费者声明的是否一致：

```
error: layout mismatch: convert_layout declares its source is ND
       but the producer left it as nZ
```

数据在解释器里始终按**逻辑 ND** 存放，只有标签在流动。这是两阶段 layout 策略的
便宜的那一半：不做分形寻址，但生产者/消费者不一致免费抓到。字节精确的分形寻址
（`--exact-layout`）尚未实现。

---

## 4. 内存模型

| 池 | 归属 |
|---|---|
| GM、SSBUF | 全局一份，所有核共享 |
| UB、L1、L0A、L0B、L0C | **每个 block 一份** —— MIX kernel 里 AIC 的 fixpipe 写的就是它的 AIV 要读的那块 UB |
| Host | 解释器私有的临时区（手写用例里非 GM 的入参） |

核用完整的 `(blockIdx, coreKind, subBlockIdx)` 三元组索引，即使只模拟一个 block
也是如此，这样开多 block 不需要改数据结构。

### 4.1 毒值填充

新分配的 buffer 一律填毒值（`--poison`，默认开）：

- 浮点：该格式的 quiet NaN；
- 整型：`0xCD` 重复填充。

配合 lazy 提交，**漏同步会直接读出 NaN**，比"数值差了 1e-3"好定位一个数量级。

### 4.2 分配模型

每个 `memref.alloc` **站点**在每个核上拥有一块 buffer，站点再次执行时复用（并重新
填毒值）。循环体不能每轮都消耗新的片上存储：PlanMemory 也是给每个站点分配一个地址，
按执行次数 bump 分配会耗尽 UB，把本来放得下的 kernel 报成超容量。

代价是同一个 alloc op 的两块 buffer 不能同时存活 —— 单核内的 SSA 本来也不允许。

---

## 5. 数值精度

所有算术走 `APInt` / `APFloat`，包括 f16（`IEEEhalf`）、bf16（`BFloat`）和 f8 格式。

**为什么不能用 `float` 凑合**：用 `float` 模拟 f16 的 tie 行为会错，f32→bf16 的舍入
也会错，round-to-odd 更是根本表达不出来。"功能正确性参考"最常见的失败方式就是死在
这里 —— 数值差 1 ulp，你会先怀疑编译器，最后才发现是解释器自己的锅。

`hivm.hir.vcast` 实现了全部六种舍入模式：`rint`、`round`、`floor`、`ceil`、`trunc`、
`odd`（Von Neumann 舍入），外加 `truncwithoverflow`。

**例外**：超越函数（`vexp`、`verf`、`vsqrt` 等）在 double 里算完再舍回目标格式 ——
硬件自己的近似实现本来就不一样，追求逐位一致没有意义。

### 5.1 逐位精度扫描

lit 用例只能告诉你一个 op 接上线了；只有把成千上万个位模式跑一遍、逐位对独立参考
实现比对，才能确认结果对到最后一位。`test/Precision/` 下的四个扫描脚本干这
件事（`lit.local.cfg` 让 lit 完全忽略这个目录，它们是开发工具而不是回归用例）：

| 脚本 | 覆盖 |
|---|---|
| `sweep_binary.py` | `+ - * / max min` 与 `abs relu sqrt rec`，f16 / f32，边界值两两组合 |
| `sweep_exhaustive.py` | 全部 65536 个 f16 位模式的一元运算；`vcast` 的每种舍入模式 |
| `sweep_integer.py` | 回绕、`INT_MIN`、移位量等于/超过位宽、除零取模零 |
| `sweep_lowprec.py` | bf16 与两种 f8：全部可达的 `vcast` 组合、bf16 前缀扫描、bf16 进 `mmadL1`、各格式的毒值 |

参考实现 `fp.py` 用 Python 的 double 算完再一次性舍入到目标格式。这对 f16 / bf16 /
f32 的 `+ - * /` 和 `sqrt` 是**正确舍入**的：安全的二次舍入要求中间格式至少
`2p+2` 位，f16 要 24 位、bf16 要 18 位、f32 要 50 位，而 double 有 53 位。
`Fmt.round` 本身由"把每个 f16 / bf16 / f8e5m2 位模式往返一遍"校验。
遇到定义式而非算术式的语义（round-to-odd、五种取整、IEEE `maximum`/`minimum`），
参考实现直接照定义写。

扫描抓出并已修复的三个 bug，都是"看起来完全合理的数值"：

| bug | 现象 | 依据 |
|---|---|---|
| `vmax`/`vmin`/`arith.maximumf`/`minimumf`/`vreduce` 的 ±0 | `a > b ? a : b` 在 `max(+0, -0)` 上两个方向都答错 —— 两个零谁都不大于谁，于是它无条件返回第二个操作数 | `LowerToLoops` 把浮点 `vmax` 降成 `arith.maximumf`，即 IEEE 754-2019 `maximum`，要求 `-0 < +0` |
| `vrelu(NaN)` | 用"负数就取 0"实现时，符号位碰巧置起的 NaN 返回 `+0`，没置起的返回 NaN —— 答案取决于一个不携带任何含义的位 | `HIVMToArith` 把 `vrelu` 定义成 `maximumf(0, x)` |
| round-to-odd 下溢 | 小于最小 f16 次正规数的输入被舍成 0 | round-to-odd 存在的全部意义就是"经它中转再舍到最近"等价于一次正确舍入；舍成 0 恰好丢掉它要保住的那点信息，结果应当是最小次正规数 |

这三条分别由 `functional/float-edge-cases.mlir`、`functional/round-to-odd.mlir`
钉住，不必重跑扫描也不会退化。cube 累加器的宽度由
`functional/mmad-accumulator-width.mlir` 钉住：A 行是 `[4096, 1, 1, 1]`、B 全 1，
f32 累加得 4099，而 f16 累加器（4096 处 ulp 是 4）会答 4096 —— 一个完全说得通的
错数。

### 5.2 低精度格式

bf16 和 f8 在 HIVM 里几乎没有自己的算术 op —— `vadd` 那一族的 verifier 拒绝 bf16，
f8 更是一个都没有 —— 所以它们的正确性几乎全部落在 `vcast` 上。真实 kernel 里每一个
bf16 / f8 值都是从某个转换来的，因此**转换表本身就是被验证的对象**，不是细节。

哪些转换真的可达是**探测 verifier** 得出的，不是照抄 ODS 里那张不完整的表：

```
f32  -> bf16      rint round floor ceil trunc     bf16 -> f32   rint round
f32  -> f8e4m3fn  rint                            f8e4m3fn -> f32   rint
f32  -> f8e5m2    rint                            f8e5m2   -> f32   rint
bf16 -> i32       rint round floor ceil trunc     i8   -> bf16  rint
```

加宽必须精确，所以**穷举**：全部 65536 个 bf16 位模式、两种 f8 各 256 个，任何不一致
都毫无歧义。窄化对着 `fp.py` 的格式模型比。此外还覆盖了在 bf16 自身上累加的四种前缀
扫描、bf16 操作数进 `mmadL1` 累加到 f32 L0C，以及每种格式的毒值。

**两种 f8 不是同一个形状**，把任何一个当成"位数更少的 f16"都会错：

- **f8e5m2** 是 IEEE 形状：1-5-2，有无穷，最大有限值 57344。
- **f8e4m3fn 没有无穷**。它的 NaN 编码只有 `0x7f` 和 `0xff`，于是全 1 指数除最后一个
  模式外是一个**普通的 binade** —— 最大有限值是 **448**，而不是按 IEEE 形状读指数域
  会得到的 240；溢出产生 NaN，因为没有别的可产生。

`functional/bf16-cast.mlir` 与 `functional/f8-cast.mlir` 把两张表都钉住，包括**毒值
必须是目标格式自己的 NaN** —— 否则"漏同步会读出 NaN"这条性质对这些类型就悄悄不成立了。

> 溢出约定值得标注为**未验证**：解释器跟随 APFloat，APFloat 跟随 OCP 论文取 NaN。
> 如果硬件是饱和到最大有限值，二者会不一致，这一点没有在真机上确认过。

---

## 6. 执行模型

核是**协作式绿色线程**。每个核一个显式的帧栈，而不是 C++ 递归 —— 这样才能在阻塞
的 op 上挂起：

- `CallFrame`：一次函数激活，持有 `DenseMap<Value, RuntimeValue>` 环境；
- `RegionFrame`：区域激活，持有 block 和指令指针，外加 `scf.for` / `scf.while`
  的循环状态。

`inorder` 和 `lazy` 使用逐指令轮询：每个可运行核执行一条 op 后，调度器就按 `CoreId`
字典序选择下一个核；已阻塞或结束的核会被跳过。这样不依赖宿主线程也能稳定模拟多个
核的并行推进，并且调试会话可以完全复现。`fuzz` 则随机选核，并给它 1～8 条 op 的
随机时间片。阻塞时**不推进指令指针**，下次调度重新执行同一条 op。

---

## 7. 支持范围

**输入必须是 memref 形态。** tensor 类型的操作数会得到明确的报错，指向 bufferization
而不是一句莫名其妙的 "unbound operand"。

支持：

| 类别 | 内容 |
|---|---|
| 社区 dialect | `func`、`scf`（for / if / while / execute_region / index_switch）、`cf`、`arith`、`math`、`index`、`memref`、`annotation`、`scope` |
| `vector` | `transfer_read` / `transfer_write`（含 permutation_map 与越界补齐）、`broadcast`、`splat`、`shape_cast`、`multi_reduction`、`extract` |
| `llvm` 子集 | `inttoptr` / `ptrtoint` / `load` / `store` / `getelementptr` 及整数运算（lowering 之后残留的裸指针标量代码） |
| HIVM elementwise | ~40 个 op，加 `vcast` / `vcmp` / `vsel` |
| HIVM reduce / 重排 | `vreduce`、`vbrc`、`vtranspose`、`vflip`、`vconcat`、`vpad`、`vgather`、`vinterleave`、`vdeinterleave`、`vsort` |
| HIVM 前缀扫描 | `vcumsum`、`vcumprod`、`vcummax`、`vcummin`（含 `reverse`） |
| HIVM copy 族 | `load` / `store` / `copy` / `fixpipe` / `nd2nz` / `nz2nd` / `l12ub` |
| HIVM 稀疏 / 跨步 / 原子 | `indirect_load` / `indirect_store`（含 mask 与 `other`）、`stride_load` / `stride_store`、`atomic_cas`、`atomic_xchg`、`load_scalar` |
| HIVM cube | `mmadL1` / `batchMmadL1`（含 `real_m/k/n` 裁剪，MTE1/M 两段建模） |
| HIVM 其他族 | view 族、query 族、完整的 sync 族 |

索引来自数据的访问（`vgather` / `indirect_*`）在发射时无法知道会碰到哪些字节，
所以它们向竞争检测器声明**整块源（或目的）buffer**。这是保守的方向：可能过报共享，
但不会漏；另一种做法是提前提交 effect 去读索引，那等于放弃延迟提交模型。
索引越界**总是**报错，而不是折回某个相邻元素 —— 安静地读错元素正是这个工具该替你
抓住的那类 bug。

**刻意不做**：

- **性能建模**。不做 cycle 级模拟，没有 bank 冲突、没有带宽模型。
- **字节精确的分形排布**。数据按逻辑 ND 存放，`zN` / `nZ` / `DOTx_ND` 只做标签检查。
  `--exact-layout` 预留给真正的分形寻址，尚未实现。
- **整问题宏 op**（`hivm.hir.matmul`、`mix_matmul`、`mix_group_matmul`）。这些携带
  tiling 和 epilogue 参数，不是一次 L1 tile 上的 MAC；故意不注册，让驱动明确报出来，
  而不是算出一个看着合理其实是错的结果。
- **SIMT 路径**。只做 SIMD。

没有 handler 的 op 会报 `unsupported op: <name>` 并非零退出。这是刻意的：
**一个清晰的空缺比 100 个 op 支持但抓不到 bug 有用得多。**

---

## 8. 命令行

```
npuir-interp <input.mlir>
  --entry=<name>              # 默认自动找 hacc.entry 函数
  --args=<spec>,<spec>,...    # 按参数顺序：<file>.npy | zeros | poison | arange | <数字>
  --out=<prefix>              # 把每个 GM 入参写成 <prefix>arg<N>.npy
  --block-dim=<N>             # 默认 1
  --sub-block-num=<N>         # 每个 AIV 的 sub-vector core 数，默认 1
  --sched=inorder|lazy|fuzz   # 默认 lazy
  --seed=<N>                  # fuzz 用
  --gm-size / --ub-size / --l1-size / --l0a-size / --l0b-size / --l0c-size
  --ssbuf-size / --host-size
  --use-target-sizes          # 从 dlti.target_system_spec 读片上容量（默认开）
  --dyn-gm-elems=<N>          # memref<?x...> 入参假定的元素个数
  --poison                    # 默认开
  --check=sync,race,deadlock,oob
  --check-raw-pointer-races
  --exact-layout              # 尚未实现
  --trace=<file>              # 逐 op 执行轨迹
  --max-steps=<N>
```

检查命中、死锁、或遇到不支持的 op 时退出码非零。

`.npy` 支持 C order、小端、v1.0。bf16 和 f8 没有对应的 NumPy 标量类型，按同宽度的
无符号整数读写（`<u2` / `|u1`），Python 侧用 `ml_dtypes` 重新解释即可。

---

## 9. 测试体系

```
test/functional/   数值；多数用 cmp 做 inorder / lazy 差分
test/sync/         期望 MISSING SYNC
test/race/         期望 DATA RACE
test/deadlock/     期望 DEADLOCK
test/oob/          期望越界 / 超容量报错
test/layout/       期望 layout 标签不匹配
test/errors/       期望明确拒绝，而不是给个错答案
test/Precision/    逐位精度扫描（开发工具，lit 不收）
unittests/          ShadowMemory / VectorClock / MemRefValue / 驻留标记合并的 gtest
```

**每一个检查能力都必须有一个"故意写错的 IR"用例。** 只有正向用例的检查器，和一个
根本没跑的检查器**无法区分**。`sync/`、`race/`、`deadlock/`、`oob/`、`layout/` 下的
每个文件都是一份正确的 kernel 删掉某一样东西，并在注释里写明删了什么。

### 功能用例覆盖

| 用例 | 覆盖内容 |
|---|---|
| `vecadd-inorder` / `vecadd-synced` | M0 基线；inorder 与 lazy 一致性 |
| `loop-reduce` | `scf.for` 循环携带状态、`vreduce` 累加 |
| `elementwise` | sub / mul / div / max / abs / sqrt、vcmp+vsel 与 i8 掩码 |
| `vcast-rounding` | 五种整数舍入模式，逐位校验 |
| `narrow-float` | f16 运算与 bf16 往返，按位模式校验 |
| `vector-casts` | 向量上的 truncf / fptosi / bitcast（真实 IR 的写法） |
| `vector-permutation` | permutation_map（minor identity 与 broadcast）、越界补齐 |
| `views` | 链式 `reinterpret_cast`、嵌套与降秩 subview、跨步 |
| `reshape-transpose` | `expand_shape` / `collapse_shape`、`vtranspose`、带结果的 `scf.if` |
| `vf-call` | 外提的向量函数：调用帧 + `vector` dialect |
| `cube-matmul` | `nd2nz` → `mmadL1` → `fixpipe`，跨 MTE2 / M / FIX |
| `mmad-real-dims` | L1 tile 有 padding 时按 `real_k` 裁剪 |
| `mmad-macro-pipes` | 宏 op 两段建模：MTE1 段退休后重填 L1，cube 仍用快照（§2.4） |
| `mmad-accumulator-width` | L0C 是 f32：f16 累加器会把 4099 算成 4096 |
| `float-edge-cases` | ±0 排序、NaN 传播、`relu(NaN)`（§5.1） |
| `round-to-odd` | round-to-odd 的下溢、上溢与不精确三种走向（§5.1） |
| `vector-reorder` | `vflip` / `vconcat` / `vpad` / `vgather` / `vinterleave` / `vdeinterleave` / `vsort` |
| `vector-scan` | 四种前缀扫描，正反两向，含 rank-2 |
| `sort-nan-order` | 带 NaN 的 `vsort`：全序而非未定义的比较器 |
| `bf16-cast` | f32→bf16 的 tie 与溢出、精确加宽回来、bf16→i32 |
| `f8-cast` | 两种 f8 的边界，含 e4m3fn 没有无穷这件事 |
| `indirect-access` | `indirect_load` / `indirect_store`（含 mask、`other`）、`stride_load` / `stride_store`（含 `numel` 截断） |
| `atomic-rmw` | 两个 block 对同一 GM 做幂等原子更新，且不互相误报 |
| `scalar-pipe` | PIPE_S 两个方向的 flag 都在位；标记合并后不触发上限 warning |
| `scalar-pipe-barrier` | `pipe_barrier[<PIPE_S>]` / `[<PIPE_ALL>]` 退休标量标记 |
| `scalar-war-released` | `set_flag[PIPE_S, PIPE_MTE2]`：标量读完 DMA 才能重填 |
| `scalar-cross-core` | `sync_block_set[<CUBE>, <PIPE_S>, …]`：标量作为跨核生产者 |
| `mix-cross-core` | AIC 生产、AIV 消费，跨核 flag 定序 |
| `barrier-loop` | `sync_block` 在循环里 rearm 三轮 |
| `block-lock` | `sync_block_lock` 串行化三个 block |
| `blockidx-tiling` | 四个 block 各切各的 tile —— 且不误报竞争 |
| `atomic-and-controlflow` | 四个 block 的原子 add、`scf.while`、`cf.br` |
| `loop-alloc` | 循环里的 `memref.alloc` 复用，不误报超容量 |

### 改动之后值得重跑的两个扫描

1. **差分**：每个功能 kernel 在 `inorder` 和 `lazy` 下的输出文件必须逐字节一致。
   唯一刻意的例外是 `vecadd-inorder`（那本来就是没插同步的 kernel）。
2. **fuzz**：每个功能 kernel 在一批 `--seed` 下都必须干净，每个负向用例都必须仍然
   命中。**只在默认调度下才成立的检查器不算检查器。**

3. **精度扫描**：改动任何算术路径之后跑一遍 `test/Precision/`（见 §5.1），
   四个脚本都必须报 0 个不一致。

仓库内的 lit 与 gtest 套件是稳定的回归基线。四个精度扫描耗时更长，因此作为独立的
开发者检查保留，不混入普通回归套件。

---

## 10. 确定性

输出（包括诊断）完全可复现：

- `DenseMap` / `DenseSet` 的**迭代顺序**不参与任何影响输出的决策（指针哈希在不同
  运行间不稳定）；需要有序遍历的地方用 `std::map` / `MapVector`。
- 调度器按 `CoreId` 字典序选核。
- fuzz 模式的随机源是固定 seed 的 `std::mt19937_64`。

---

## 11. 代码结构

```
include/bishengir/Tools/Interp/
  Value.h          RuntimeValue / MemRefValue / AddrSpace / 元素编解码
  Memory.h         Arena / VectorClock / ShadowMemory / 毒值
  PipeEngine.h     Pipe / Effect / FlagKey / 每核队列
  Interpreter.h    调度器、CoreState、帧、op 注册表、选项
lib/
  Value.cpp Memory.cpp PipeEngine.cpp Interpreter.cpp
  NpyIO.{h,cpp}    极简 .npy 读写
  OpUtils.{h,cpp}  handler 共用的取值 / 转换 / 迭代辅助
  OpsCommunity.cpp func / scf / cf / arith / math / index / memref / llvm
  OpsVector.cpp    vector dialect
  OpsElementwise.cpp  HIVM elementwise 宏表
  OpsMemory.cpp    HIVM copy 族与 view 族
  OpsSync.cpp      HIVM sync 族
  OpsMisc.cpp      query / vbrc / vreduce / vtranspose / mmad
  OpsShape.cpp     重排族（flip / concat / pad / gather / interleave / sort）与前缀扫描
  OpsIndirect.cpp  稀疏与跨步访问、原子读改写、load_scalar
tools/
  npuir-interp.cpp   main 与命令行
```

**对 HIVM dialect 零侵入**：不改任何 `.td` 文件，`lib/Dialect` 里没有一行知道解释器
的存在。op handler 住在一个按 op 名索引的注册表（`OpRegistry`）里。

> 方案原稿建议用 `ExternalModel`。这里改用按名索引的注册表：同样达到零侵入的目标，
> 样板代码少一个数量级，而且分发顺序不依赖指针哈希（对确定性有好处）。

### 怎么加一个 op

1. 在对应的 `Ops*.cpp` 里写一个 `ExecResult(Interpreter&, CoreState&, Operation*)`；
2. 在该文件的 `register*Ops()` 里 `registry.add("dialect.op", handler)`；
3. 纯计算 op 直接算完就写；**带 pipe 的 op 必须走 `issueEffect`**，把真正的数据搬运
   放进 commit 闭包里，并在 issue 时算好读写字节区间；
4. 加一个 `test/functional/` 用例，并尽量带上 inorder/lazy 差分。

---

## 12. 已知限制

| 限制 | 说明 |
|---|---|
| PIPE_S 驻留标记有 4096 条上限 | 合并不掉的散乱标量访问超过上限就丢弃（打一次 warning），此后针对它们的漏 flag 会漏报（见 2.3） |
| 跨核 flag 只按 `tpipe` 冲刷 | 消费侧的 `pipe` 不参与可见性建模；一个核内所有 pipe 共用一份向量时钟，所以"等错 pipe"这类 bug 抓不到 |
| 标量的跨核可见性 | 标量写立即落内存，所以"用错 `tpipe` 发布标量结果"不会被发现 —— 真机上这需要 cache 维护，未建模 |
| MIX 跨核 flag 的组语义 | AIC→AIV 按 generation 广播、AIV→AIC 按 sub-core 汇聚；由真实 SIMD kernel 形状推导，尚未用硬件规格独立验证（见 2.5） |
| `collectRanges` 会放大 | 超过约 1024 行的非连续视图退化为整段区间，可能过报共享 |
| `--exact-layout` 未实现 | 只有阶段一的标签检查 |
| 整问题宏 matmul 未注册 | 故意报错而不是算错 |
| f8 硬件行为未验证 | 两种 f8 的转换与边界已有回归和逐位扫描，但溢出约定尚未与真机核对 |
| PlanMemory 之前的 MIX IR | 裸 `memref.alloc()` 没有烘焙偏移，AIC 与 AIV 的分配无法被识别为同一块 UB —— IR 里本来就没写它们别名 |

---

## 13. 里程碑对照

| 里程碑 | 状态 |
|---|---|
| M0 driver + memref/scf/arith + load/store/vadd + inorder | ✅ |
| M1 elementwise 全表 + reduce + APFloat 精确 f16/bf16/vcast 六种舍入 | ✅ |
| M2 双核并发 + PipeEngine lazy + flag/barrier 完整语义 + 死锁检测 + 毒值 | ✅ |
| M3 ShadowMemory + 向量时钟 happens-before + race 报告 + fuzz + 多 block | ✅ |
| M4 mmadL1 / fixpipe / nd2nz / nz2nd | 🟡 handler 与数值用例已有；`--exact-layout` 未做 |
