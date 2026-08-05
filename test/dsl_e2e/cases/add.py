import triton
import triton.language as tl

if __package__:
    from ..common import E2ECase, case_main
else:
    from _direct import E2ECase, case_main


@triton.jit
def kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.debug_barrier()
    tl.store(out_ptr + offsets, x + y, mask=mask)


CASE = E2ECase(
    name="add",
    kernel=kernel,
    signature={
        "x_ptr": "*fp32",
        "y_ptr": "*fp32",
        "out_ptr": "*fp32",
        "n_elements": "i32",
        "BLOCK_SIZE": "constexpr",
    },
    constants={"BLOCK_SIZE": 1024},
    args="0,zeros,zeros,arange,1,zeros,1500,2,1,1",
    block_dim=2,
    output_arg=5,
    ops=("hivm.hir.vadd",),
    elements=1500,
    expected=lambda i: float(i + 1),
)


if __name__ == "__main__":
    raise SystemExit(case_main(CASE))
