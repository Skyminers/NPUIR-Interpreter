import math

import triton
import triton.language as tl

from ..common import E2ECase, case_main


N_COLS = 30


@triton.jit
def kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < N_COLS
    offsets = row * N_COLS + columns
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N_COLS
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / N_COLS
    # Keep rstd vector-valued to avoid triton-ascend's scalar memref path.
    rstd = 1.0 / tl.sqrt(variance + centered * 0.0 + 1.0e-5)
    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + columns, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, centered * rstd * weight + bias, mask=mask)


def expected(index: int) -> float:
    column = index % N_COLS
    mean = (N_COLS - 1) / 2.0
    variance = sum((i - mean) ** 2 for i in range(N_COLS)) / N_COLS
    normalized = (column - mean) / math.sqrt(variance + 1.0e-5)
    return normalized * column + 1.0


CASE = E2ECase(
    name="layer_norm",
    kernel=kernel,
    signature={
        "x_ptr": "*fp32",
        "weight_ptr": "*fp32",
        "bias_ptr": "*fp32",
        "out_ptr": "*fp32",
        "N_COLS": "constexpr",
        "BLOCK_SIZE": "constexpr",
    },
    constants={"N_COLS": N_COLS, "BLOCK_SIZE": 32},
    args="0,zeros,zeros,arange,arange,1,zeros,2,1,1",
    block_dim=2,
    output_arg=6,
    ops=("hivm.hir.vreduce", "hivm.hir.vsqrt", "hivm.hir.vmul"),
    elements=2 * N_COLS,
    expected=expected,
    atol=1.0e-4,
)


if __name__ == "__main__":
    raise SystemExit(case_main(CASE))
