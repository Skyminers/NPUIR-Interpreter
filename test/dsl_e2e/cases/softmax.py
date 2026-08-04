import math

import triton
import triton.language as tl

from ..common import E2ECase, case_main


N_COLS = 30


@triton.jit
def kernel(x_ptr, out_ptr, N_COLS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < N_COLS
    offsets = row * N_COLS + columns
    values = tl.load(x_ptr + offsets, mask=mask, other=-float("inf"))
    numerator = tl.exp(values - tl.max(values, axis=0))
    tl.store(out_ptr + offsets, numerator / tl.sum(numerator, axis=0), mask=mask)


def expected(index: int) -> float:
    column = index % N_COLS
    denominator = sum(math.exp(i - (N_COLS - 1)) for i in range(N_COLS))
    return math.exp(column - (N_COLS - 1)) / denominator


CASE = E2ECase(
    name="softmax",
    kernel=kernel,
    signature={
        "x_ptr": "*fp32",
        "out_ptr": "*fp32",
        "N_COLS": "constexpr",
        "BLOCK_SIZE": "constexpr",
    },
    constants={"N_COLS": N_COLS, "BLOCK_SIZE": 32},
    args="0,zeros,zeros,arange,zeros,2,1,1",
    block_dim=2,
    output_arg=4,
    ops=("hivm.hir.vreduce", "hivm.hir.vexp", "hivm.hir.vdiv"),
    elements=2 * N_COLS,
    expected=expected,
    atol=2.0e-6,
)


if __name__ == "__main__":
    raise SystemExit(case_main(CASE))
