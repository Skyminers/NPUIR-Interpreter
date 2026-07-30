# `npuir-interp` 使用指南

[返回文档索引](README.md) · [架构与实现](Architecture_zh.md)

在没有昇腾硬件的机器上，把编译后的 HIVM（NPUIR）跑起来。

它做两件事，**第一件是别的工具做不到的**：

1. **查同步**——验证编译器插入的 flag / barrier / lock 到底够不够。
2. **查数值**——无硬件跑通 kernel，和 numpy / torch 的 golden 比对。

> 想了解**为什么**这样设计（延迟提交模型、PIPE_S 建模、向量时钟等），
> 想了解**为什么**这样设计（延迟提交模型、PIPE_S 建模、向量时钟等），
> 请阅读[架构与实现](Architecture_zh.md)。本文只讲**怎么用**。

---

## 1. 构建

```bash
git submodule update --init --recursive
./build.sh --target npuir-interp
```

产物在 `build/bin/npuir-interp`。

---

## 2. 五分钟上手

```bash
# 跑一个 kernel，把每个 GM 入参的最终内容写成 .npy
build/bin/npuir-interp kernel.mlir \
    --args=a.npy,b.npy,zeros \
    --out=result.
```

- `--args` 按**入参顺序**逐个给定内容，个数要和函数签名一致；
- `--out=result.` 会写出 `result.arg0.npy`、`result.arg1.npy` …… 每个 GM 入参一个。

退出码就是结论：**0 = 干净，非 0 = 有问题**，问题会打在 stderr 上。所以可以直接
放进脚本里：

```bash
if build/bin/npuir-interp kernel.mlir --args=a.npy,b.npy,zeros; then
  echo "同步与数值都没问题"
fi
```

### 输入必须是 memref 形态

解释器只接受 **bufferization 之后**的 IR。喂 tensor 形态的 IR 会得到一句明确的
报错，而不是莫名其妙的 "unbound operand"。

典型的取 IR 方式：给 `bishengir-compile` 加 `--bishengir-print-ir-after=<pass>`，
把某个 pass 之后的 IR 存下来。

---

## 3. `--args` 能写什么

每一项对应一个函数入参：

| 写法 | 含义 |
|---|---|
| `foo.npy` | 从 `.npy` 文件读（小端，非 Fortran 序） |
| `zeros` | 全 0 |
| `arange` | 0, 1, 2, 3 …… 按元素序 |
| `poison` | 填毒值（浮点 NaN / 整型 `0xCD`） |
| `3.5` | 整个 buffer 填这个常量；标量入参就绑成这个值 |

给少了的入参按 `zeros` 处理，所以纯输出 buffer 可以不写。想让"忘了给输入"这件事
自己暴露出来，就显式写 `poison`——读到 NaN 比读到 0 好定位。

**bf16 和 f8 的 `.npy`**：NumPy 没有这两种标量类型，所以解释器按**原始位模式**读写
（bf16 用 `<u2`，f8 用 `|u1`）。Python 侧可以用 `ml_dtypes` 转换，或者直接操作位。

---

## 4. 三种调度模式：`--sched`

**这是最重要的一个开关。**

| 模式 | 行为 | 什么时候用 |
|---|---|---|
| `--sched=inorder` | 每条 effect 立即生效 | 只想要数值结果。最快，但**按构造原理发现不了任何同步问题** |
| `--sched=lazy`（默认） | effect 尽可能晚提交 | 日常检查 |
| `--sched=fuzz --seed=N` | lazy + 随机化核间交错 | 找那些"换个调度就崩"的脆弱同步 |

### 最强的自动判据：差分

同一份 IR 用 `inorder` 和 `lazy` 各跑一遍，**输出逐字节相同 ⇒ 同步是充分的；
不同 ⇒ 一定漏了同步。** 不需要 golden 数据，也不需要人看：

```bash
build/bin/npuir-interp kernel.mlir --args=a.npy,b.npy,zeros --sched=inorder --out=io.
build/bin/npuir-interp kernel.mlir --args=a.npy,b.npy,zeros --sched=lazy    --out=lz.
cmp io.arg2.npy lz.arg2.npy && echo OK
```

### fuzz 扫一遍

```bash
for s in $(seq 0 15); do
  build/bin/npuir-interp kernel.mlir --args=a.npy,b.npy,zeros \
      --sched=fuzz --seed=$s || echo "seed $s 有问题"
done
```

---

## 5. 多核

```bash
--block-dim=8        # 模拟 8 个 block
--sub-block-num=2    # 每个 AIV 下挂 2 个 sub-vector core（它们共享一块 UB）
```

MIX kernel（`*_mix_aic` / `*_mix_aiv`）会自动识别，AIC 和它的 AIV 共享同一块 UB。

> `--sub-block-num=2` 时跨核 flag 建模成**计数信号量**而不是广播锁存：两个
> sub-vector core 同时等一个只 set 过一次的 flag，第二个会一直阻塞。这条建模假设
> 尚未在真机上验证，默认值 1 回避了这个问题。

---

## 6. 看懂四种报告

### `MISSING SYNC` —— 核内漏 flag

```
MISSING SYNC on AIV#0.0: PIPE_V op touches data still in flight on PIPE_MTE2
  in flight  PIPE_MTE2  hivm.hir.load @ add.mlir:4:5
  consumes   PIPE_V  hivm.hir.vadd @ add.mlir:8:5
  the two are unordered without a set_flag[PIPE_MTE2, PIPE_V, <id>] / wait_flag pair between them
```

**直接告诉你该在哪两条 op 之间插哪一对 flag。** 最后一行就是修复方案。

### `DATA RACE` —— 跨核数据竞争

```
DATA RACE on gm %arg1 +0 [0x40, 0x60)
  W  AIC#0    kernel.mlir:28:5  hivm.hir.store  vc=<4,0>
  R  AIV#0.0  kernel.mlir:40:5  hivm.hir.load   vc=<0,1>
  no happens-before edge between these two accesses
  nearest sync op: hivm.hir.sync_block @ kernel.mlir:180
                   (barrier only - carries no data-dependency flag)
```

最后两行是关键：**最近的同步点是什么，以及它为什么不够**。上面这个例子里有
barrier 但没有数据依赖 flag，一眼就能定位到是哪个 pass 少插了东西。

### `DEADLOCK` —— 死锁

两种形态。**flag 等待环**会列出谁卡在哪个 flag 上、谁已经结束了：

```
DEADLOCK: circular flag wait
  AIV#0.0 blocked at :21  sync_block_wait[PIPE_MTE3->PIPE_MTE2 flag=7 block=0]
  AIC#0 is Done (returned at :16)
```

**barrier 到达数不一致**会列出期望的参与者和实际到达的：

```
DEADLOCK: barrier ALL:1 arrival mismatch
  expected participants: AIC#0 AIV#0.0 (2 cores)
  arrived:  AIC#0 @ :20
  AIV#0.0 is Done (returned at :32) and never reached ALL:1
  hint: a barrier that only some cores execute is the classic symptom of a
        barrier cloned into a conditional region
```

### 读到 NaN / `0xCD`

不是报告，是**症状**。新分配的 buffer 一律填毒值，所以"生产者的写还没提交、消费者
就读了"会直接读出 NaN，比"数值差了 1e-3"好定位一个数量级。

想确认某个 NaN 是不是漏同步造成的：`--sched=inorder` 再跑一遍，如果结果正常，那就
是漏同步。

---

## 7. 常用选项速查

```
--entry=<name>            入口函数（默认取带 hacc.entry 的那个/那些）
--args=<spec>,...         逐个入参的内容，见 §3
--out=<prefix>            把每个 GM 入参写成 <prefix>arg<N>.npy
--sched=inorder|lazy|fuzz 调度模式，见 §4
--seed=<N>                fuzz 的随机种子
--block-dim=<N>           模拟几个 block（默认 1）
--sub-block-num=<N>       每个 AIV 几个 sub-vector core（默认 1）

--check=sync,race,deadlock,oob    只开其中几项（默认全开）
--check=none                      全关，只要数值
--check-raw-pointer-races         连裸指针 flag 便签本的并发也报（默认不报）

--trace=<file>            逐条 op 的执行轨迹，带向量时钟
--max-steps=<N>           跑到这么多条 op 就中止（默认 2 亿，防死循环）
-v                        啰嗦模式

--ub-size / --l1-size / --l0a-size / --l0b-size / --l0c-size / --ssbuf-size
--gm-size / --host-size   各内存池容量（字节）
--dyn-gm-elems=<N>        动态形状 GM 入参假定的元素个数（默认 4096）
--poison=false            关掉毒值填充（一般不要关）
```

> 片上容量默认从 module 的 `dlti.target_system_spec` 读，和编译时的目标一致；
> 显式给了 `--ub-size` 之类才会覆盖它。

---

## 8. 三个典型场景

### A. 我怀疑某个 pass 少插了同步

```bash
build/bin/npuir-interp kernel.mlir --args=... --sched=lazy
```

看 `MISSING SYNC` / `DATA RACE`，报告里已经写了该插哪一对 flag。

### B. 我要确认这个 kernel 算得对

```bash
build/bin/npuir-interp kernel.mlir --args=a.npy,b.npy,zeros \
    --sched=inorder --check=none --out=got.
```

`--check=none` 关掉全部检查、`inorder` 保证顺序执行，此时它就是一个纯粹的数值
参考。再和 numpy / torch 的结果比。

### C. 我要把它接进 CI

```bash
# 干净就返回 0
build/bin/npuir-interp kernel.mlir --args=... --sched=lazy || exit 1
# 再加一道差分
build/bin/npuir-interp kernel.mlir --args=... --sched=inorder --out=io.
build/bin/npuir-interp kernel.mlir --args=... --sched=lazy    --out=lz.
cmp io.arg2.npy lz.arg2.npy
```

---

## 9. 跑不通的时候

| 现象 | 原因与处理 |
|---|---|
| `tensor-form operand or result on '<op>': ... run the remaining bufferization passes first` | IR 还没 bufferize。取更靠后的 pass 的输出 |
| `unsupported op: <name>` | 这个 op 还没实现。**故意报错而不是猜一个结果** —— 明确的空缺比悄悄算错好 |
| `ub capacity exceeded: need N more bytes, arena is M bytes and K are already in use` | 真的放不下（那就是被抓到的 bug），或者要调 `--ub-size` |
| `out of bounds access to gm at byte N (arena capacity M, element size E)` | 地址算错了。越界**一律致命**，因为那条访问根本没法执行 |
| 报了一堆 `MISSING SYNC` 但你确信 IR 是对的 | 先用 `--sched=inorder` 确认数值对不对；如果对，再看报告点的 pipe 对不对 |
| 跑很久不结束 | 调低 `--max-steps` 看它卡在哪，或者加 `--trace=t.log` |

---

## 10. 已知限制

交付时值得知道的几条：

- **`--exact-layout` 未实现**。数据始终按逻辑 ND 存放，`zN` / `nZ` / `DOTx_ND`
  只做标签一致性检查（生产者留的标签和消费者声明的对不对得上），不做字节精确的
  分形寻址。
- **整问题宏 op**（`hivm.hir.matmul`、`mix_matmul`、`mix_group_matmul`）**故意不
  注册**。它们携带 tiling 和 epilogue 参数，不是一次 L1 tile 上的 MAC，与其算一个
  看似合理的错数，不如明确报出来。
- **不做性能建模**。没有 cycle 数、没有 bank 冲突、没有带宽模型。
- **几条未在真机验证的建模假设**：跨核 flag 的广播语义（见 §5）、标量写的跨核
  可见性、f8e4m3fn 的溢出约定（取 NaN 还是饱和）。文档里都标注了。

---

## 11. 精度扫描（开发者）

改动过任何算术路径之后，跑一遍逐位精度扫描：

```bash
cd test/Precision
python3 sweep_binary.py      # + - * / max min、abs relu sqrt rec，f16 与 f32
python3 sweep_exhaustive.py  # 全部 65536 个 f16 位模式；vcast 每种舍入模式
python3 sweep_integer.py     # 回绕、INT_MIN、移位边界、除零
python3 sweep_lowprec.py     # bf16 与两种 f8
```

每个脚本逐 op 打一行，有任何不一致就非 0 退出。它们至今抓出过 4 个缺陷，共同点是
**算出来的数看起来完全合理**。细节见[精度扫描说明](../test/Precision/README.md)。

回归套本身：

```bash
cmake --build build --target check-npuir-interpreter
cmake --build build --target check-npuir-interpreter-unit
```
