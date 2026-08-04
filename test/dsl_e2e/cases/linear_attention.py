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
    # Positive-feature linear attention: phi(x) = x + 1 for nonnegative data.
    tl.static_assert(BLOCK_N == HEAD_DIM)
    row = tl.program_id(axis=0)
    dims = tl.arange(0, HEAD_DIM)
    phi_q = tl.load(q_ptr + row * HEAD_DIM + dims) + 1.0
    numerator = tl.zeros((HEAD_DIM,), tl.float32)
    denominator = tl.zeros((HEAD_DIM,), tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        columns = start_n + tl.arange(0, BLOCK_N)
        phi_k = (
            tl.load(k_ptr + columns[:, None] * HEAD_DIM + dims[None, :]) + 1.0
        )
        weights = tl.sum(phi_k * phi_q[None, :], axis=1)
        v = tl.load(v_ptr + columns[:, None] * HEAD_DIM + dims[None, :])
        numerator += tl.sum(weights[:, None] * v, axis=0)
        tile_denominator = tl.sum(weights, axis=0)
        denominator += tile_denominator + phi_q * 0.0

    tl.store(out_ptr + row * HEAD_DIM + dims, numerator / denominator)


def expected(index: int) -> float:
    feature = index % HEAD_DIM
    weights = [256 * token + 136 for token in range(N_CTX)]
    denominator = sum(weights)
    return sum(
        weight * (token * HEAD_DIM + feature)
        for token, weight in enumerate(weights)
    ) / denominator


CASE = E2ECase(
    name="linear_attention",
    kernel=kernel,
    signature={
        "q_ptr": "*fp32",
        "k_ptr": "*fp32",
        "v_ptr": "*fp32",
        "out_ptr": "*fp32",
        "N_CTX": "constexpr",
        "HEAD_DIM": "constexpr",
        "BLOCK_N": "constexpr",
    },
    constants={"N_CTX": N_CTX, "HEAD_DIM": HEAD_DIM, "BLOCK_N": BLOCK_N},
    args="0,zeros,zeros,0.001,arange,arange,zeros,32,1,1",
    block_dim=N_CTX,
    output_arg=6,
    ops=("hivm.hir.vreduce", "hivm.hir.vmul", "hivm.hir.vadd", "hivm.hir.vdiv"),
    elements=N_CTX * HEAD_DIM,
    expected=expected,
    atol=2.0e-3,
)


if __name__ == "__main__":
    raise SystemExit(case_main(CASE))
