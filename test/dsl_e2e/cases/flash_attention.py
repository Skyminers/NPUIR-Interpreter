import math

import triton
import triton.language as tl

if __package__:
    from ..common import E2ECase, case_main
else:
    from _direct import E2ECase, case_main


N_CTX = 32
HEAD_DIM = 16
BLOCK_M = 16
BLOCK_N = 16


@triton.jit
def kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Reference-sized FA2 forward: one query tile per program, streaming K/V
    # tiles while updating the row-wise online softmax state.
    tl.static_assert(BLOCK_M == BLOCK_N)
    tl.static_assert(BLOCK_N == HEAD_DIM)
    block_m = tl.program_id(axis=0)
    rows = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + rows[:, None] * HEAD_DIM + dims[None, :])

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    scale: tl.constexpr = 1.0 / (HEAD_DIM**0.5)

    for start_n in range(0, N_CTX, BLOCK_N):
        columns = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(k_ptr + columns[:, None] * HEAD_DIM + dims[None, :])
        scores = tl.dot(q, tl.trans(k)) * scale
        scores = tl.where(
            columns[None, :] <= rows[:, None], scores, -float("inf")
        )
        tile_max = tl.max(scores, axis=1)
        m_ij = tl.maximum(m_i, tile_max)
        probabilities = tl.exp(scores - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)

        v = tl.load(v_ptr + columns[:, None] * HEAD_DIM + dims[None, :])
        acc = acc * alpha[:, None]
        acc += tl.dot(probabilities.to(tl.float16), v)
        l_i = l_i * alpha + tl.sum(probabilities, axis=1)
        m_i = m_ij

    output = acc / l_i[:, None]
    tl.store(out_ptr + rows[:, None] * HEAD_DIM + dims[None, :], output)


def expected(index: int) -> float:
    row = index // HEAD_DIM
    feature = index % HEAD_DIM
    logits = [token / 16.0 for token in range(row + 1)]
    maximum = logits[-1]
    weights = [math.exp(logit - maximum) for logit in logits]
    denominator = sum(weights)
    return sum(
        weight * (token * HEAD_DIM + feature)
        for token, weight in enumerate(weights)
    ) / denominator


CASE = E2ECase(
    name="flash_attention",
    kernel=kernel,
    signature={
        "q_ptr": "*fp16",
        "k_ptr": "*fp16",
        "v_ptr": "*fp16",
        "out_ptr": "*fp32",
        "N_CTX": "constexpr",
        "HEAD_DIM": "constexpr",
        "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr",
    },
    constants={
        "N_CTX": N_CTX,
        "HEAD_DIM": HEAD_DIM,
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
    },
    args="zeros,zeros,0.0009765625,arange,arange,zeros,2,1,1",
    block_dim=N_CTX // BLOCK_M,
    output_arg=5,
    ops=(
        "#hivm.func_core_type<AIC>",
        "#hivm.func_core_type<AIV>",
        "hivm.hir.mmadL1",
        "hivm.hir.sync_block_set",
        "hivm.hir.sync_block_wait",
        "math.exp",
        "vector.multi_reduction",
    ),
    elements=N_CTX * HEAD_DIM,
    expected=expected,
    atol=1.0e-2,
    ttadapter_counts=(("mix_mode = \"mix\"", 1), ("linalg.matmul", 2)),
    arch="Ascend950PR_9579",
    sub_block_num=2,
    compile_args=(
        "--enable-mixed-cv=true",
        "--enable-auto-bind-sub-block=true",
        "--disable-ffts",
    ),
)


if __name__ == "__main__":
    raise SystemExit(case_main(CASE))
