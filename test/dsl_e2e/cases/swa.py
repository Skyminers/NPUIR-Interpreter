import math

import triton
import triton.language as tl

from ..common import E2ECase, case_main


N_CTX = 32
HEAD_DIM = 16
WINDOW_SIZE = 4


@triton.jit
def kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
):
    # Causal sliding-window attention over the current and previous tokens.
    row = tl.program_id(axis=0)
    dims = tl.arange(0, HEAD_DIM)
    columns = tl.arange(0, N_CTX)
    q = tl.load(q_ptr + row * HEAD_DIM + dims)
    k = tl.load(k_ptr + columns[:, None] * HEAD_DIM + dims[None, :])
    scores = tl.sum(k * q[None, :], axis=1) / (HEAD_DIM**0.5)
    visible = (columns <= row) & (columns + WINDOW_SIZE > row)
    scores = tl.where(visible, scores, -float("inf"))
    probabilities = tl.exp(scores - tl.max(scores, axis=0))
    denominator = tl.sum(probabilities, axis=0)
    v = tl.load(v_ptr + columns[:, None] * HEAD_DIM + dims[None, :])
    output = tl.sum(probabilities[:, None] * v, axis=0) / denominator
    tl.store(out_ptr + row * HEAD_DIM + dims, output)


def expected(index: int) -> float:
    row, feature = divmod(index, HEAD_DIM)
    first = max(0, row - WINDOW_SIZE + 1)
    tokens = range(first, row + 1)
    logits = [token / 16.0 for token in tokens]
    maximum = logits[-1]
    weights = [math.exp(logit - maximum) for logit in logits]
    denominator = sum(weights)
    return sum(
        weight * (token * HEAD_DIM + feature)
        for token, weight in zip(tokens, weights)
    ) / denominator


CASE = E2ECase(
    name="swa",
    kernel=kernel,
    signature={
        "q_ptr": "*fp16",
        "k_ptr": "*fp16",
        "v_ptr": "*fp16",
        "out_ptr": "*fp32",
        "N_CTX": "constexpr",
        "HEAD_DIM": "constexpr",
        "WINDOW_SIZE": "constexpr",
    },
    constants={
        "N_CTX": N_CTX,
        "HEAD_DIM": HEAD_DIM,
        "WINDOW_SIZE": WINDOW_SIZE,
    },
    args="0,zeros,zeros,0.0009765625,arange,arange,zeros,32,1,1",
    block_dim=N_CTX,
    output_arg=6,
    ops=("hivm.hir.vcmp", "hivm.hir.vsel", "hivm.hir.vreduce", "hivm.hir.vexp"),
    elements=N_CTX * HEAD_DIM,
    expected=expected,
    atol=2.0e-3,
)


if __name__ == "__main__":
    raise SystemExit(case_main(CASE))
