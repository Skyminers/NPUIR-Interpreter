import math

import triton
import triton.language as tl

from ..common import E2ECase, case_main


N_CTX = 32
HEAD_DIM = 16
BLOCK_N = 16


@triton.jit
def kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Reference-sized FA2 forward: stream K/V while updating online softmax.
    tl.static_assert(BLOCK_N == HEAD_DIM)
    row = tl.program_id(axis=0)
    dims = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + row * HEAD_DIM + dims)

    # Vector state avoids the unsupported scalar memref lowering path.
    m_i = tl.full((BLOCK_N,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_N,), tl.float32)
    acc = tl.zeros((HEAD_DIM,), tl.float32)
    scale: tl.constexpr = 1.0 / (HEAD_DIM**0.5)

    for start_n in range(0, N_CTX, BLOCK_N):
        columns = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(k_ptr + columns[:, None] * HEAD_DIM + dims[None, :])
        scores = tl.sum(k * q[None, :], axis=1) * scale
        tile_max = tl.max(scores, axis=0)
        m_ij = tl.maximum(m_i, tile_max + scores * 0.0)
        probabilities = tl.exp(scores - m_ij)
        alpha = tl.exp(m_i - m_ij)

        v = tl.load(v_ptr + columns[:, None] * HEAD_DIM + dims[None, :])
        acc = acc * alpha + tl.sum(probabilities[:, None] * v, axis=0)
        tile_sum = tl.sum(probabilities, axis=0)
        l_i = l_i * alpha + tile_sum + scores * 0.0
        m_i = m_ij

    tl.store(out_ptr + row * HEAD_DIM + dims, acc / l_i)


def expected(index: int) -> float:
    feature = index % HEAD_DIM
    logits = [token / 16.0 for token in range(N_CTX)]
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
        "BLOCK_N": "constexpr",
    },
    constants={"N_CTX": N_CTX, "HEAD_DIM": HEAD_DIM, "BLOCK_N": BLOCK_N},
    args="0,zeros,zeros,0.0009765625,arange,arange,zeros,32,1,1",
    block_dim=N_CTX,
    output_arg=6,
    ops=("hivm.hir.vreduce", "hivm.hir.vexp", "hivm.hir.vmul"),
    elements=N_CTX * HEAD_DIM,
    expected=expected,
    atol=2.0e-3,
)


if __name__ == "__main__":
    raise SystemExit(case_main(CASE))
