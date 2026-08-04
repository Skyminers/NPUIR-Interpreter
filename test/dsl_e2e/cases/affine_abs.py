import triton
import triton.language as tl

from ..common import E2ECase, case_main


@triton.jit
def kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, tl.abs(x - 4.0) * 2.0, mask=mask)


CASE = E2ECase(
    name="affine_abs",
    kernel=kernel,
    signature={
        "x_ptr": "*fp32",
        "out_ptr": "*fp32",
        "n_elements": "i32",
        "BLOCK_SIZE": "constexpr",
    },
    constants={"BLOCK_SIZE": 1024},
    args="0,zeros,zeros,arange,zeros,1500,2,1,1",
    block_dim=2,
    output_arg=4,
    ops=("hivm.hir.vmul", "hivm.hir.vabs"),
    elements=1500,
    expected=lambda i: float(abs(i - 4) * 2),
)


if __name__ == "__main__":
    raise SystemExit(case_main(CASE))
