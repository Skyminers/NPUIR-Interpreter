# Precision sweeps for `npuir-interp`

These are developer tools, not lit tests (`lit.local.cfg` keeps lit out of this
directory). They generate one kernel per operation, feed it tens of thousands
of bit patterns, and compare **every output bit** against an independent
reference written in `fp.py`.

A lit test can tell you an op is wired up. Only a sweep like this tells you the
result is the right value down to the last bit, and the cases that matter -
signed zeros, NaN operands, subnormals, exact ties, underflow - are precisely
the ones nobody thinks to write a hand test for.

## Running

```
./build.sh --target npuir-interp
cd test/Precision
python3 sweep_binary.py       # + - * / max min, abs relu sqrt rec, f16 and f32
python3 sweep_exhaustive.py   # all 65536 f16 patterns; every vcast round mode
python3 sweep_integer.py      # wrap-around, INT_MIN, shift counts, div by zero
python3 sweep_lowprec.py      # bf16 and the two f8 formats, end to end
```

Each prints one line per (op, type) and exits non-zero if anything mismatched.
`NPUIR_INTERP` overrides the interpreter path (the default is the in-tree
`build/bin/npuir-interp`); `NPUIR_INTERP_SCRATCH` overrides where the
generated kernels and `.npy` files go.

## The low-precision formats

bf16 and f8 have almost no arithmetic ops of their own in HIVM — the verifiers
of `vadd` and friends reject bf16, and there is nothing at all for f8 — so
their correctness lives in `vcast` and in the handful of ops that do accept
them. `sweep_lowprec.py` therefore treats the conversion table as the subject:
every bf16 or f8 value in a real kernel arrives through one of these.

Which conversions exist was established by probing the verifier, not by reading
the ODS table, which is incomplete:

```
f32  -> bf16      rint round floor ceil trunc     bf16 -> f32   rint round
f32  -> f8e4m3fn  rint                            f8e4m3fn -> f32   rint
f32  -> f8e5m2    rint                            f8e5m2   -> f32   rint
bf16 -> i32       rint round floor ceil trunc     i8   -> bf16  rint
```

Widening is swept exhaustively — all 65536 bf16 patterns and all 256 of each
f8 — because it has to be exact, so a mismatch is unambiguous. On top of that
the sweep covers the bf16 prefix scans (which accumulate in bf16 itself),
`mmadL1` with bf16 operands into the f32 L0C, and the poison pattern for each
format.

The f8 formats are not the same shape as each other, and treating either as
"f16 with fewer bits" gets it wrong:

- **f8e5m2** is IEEE-shaped: 1-5-2, infinities, largest finite 57344.
- **f8e4m3fn** has **no infinity**. Its only NaN encodings are `0x7f` and
  `0xff`, which makes the all-ones exponent an ordinary binade except for its
  last pattern — so its largest finite value is **448**, not the 240 an
  IEEE-shaped reading of the exponent field would give, and overflow produces
  NaN because there is nothing else to produce.

`fp.py` models that explicitly (`Fmt.nonfinite`), and self-checks by
round-tripping every representable pattern of every format through every
rounding mode.

> The overflow convention is worth flagging: the interpreter follows APFloat,
> which follows the OCP paper in producing NaN. Hardware that saturates to the
> largest finite value instead would disagree, and that has not been checked
> against a device.

## Why double is a valid oracle

`fp.py` computes in Python floats, i.e. IEEE double, and then rounds once to
the target format with `Fmt.round`. That is correctly rounded for `+ - * /` and
`sqrt` on f16, bf16 and f32: safe double rounding needs the intermediate to
carry at least `2p+2` bits, which is 24 for f16, 18 for bf16 and 50 for f32,
and double carries 53. `Fmt.round` is checked by round-tripping every f16,
bf16 and f8e5m2 bit pattern through it.

Where the reference is a *definition* rather than an arithmetic result -
round-to-odd, the five integral rounding modes, IEEE `maximum`/`minimum` -
`fp.py` and the sweeps implement the definition directly instead.

## Semantics the sweeps encode

These came from reading what the compiler itself emits, not from guessing, and
each one was a real interpreter bug before it was a test:

- **`vmax`/`vmin` on floats are IEEE 754-2019 `maximum`/`minimum`.**
  `LowerToLoops` turns them into `arith.maximumf`/`minimumf`, so NaN propagates
  *and* `-0` sorts below `+0`. A plain `a > b ? a : b` gets the zeros wrong in
  both directions, because neither zero compares greater.
- **`vrelu` is `maximumf(0, x)`**, per `HIVMToArith`. So `relu(NaN)` is NaN.
  Testing the sign bit instead answers `+0` for a NaN that happens to have its
  sign bit set - a result that depends on a bit carrying no meaning.
- **Round-to-odd never underflows to zero.** Its entire purpose is that
  narrowing through it and then rounding to nearest matches a single correctly
  rounded step; answering zero for a tiny nonzero input destroys exactly the
  information it exists to keep.

The `.mlir` tests in `../functional/float-edge-cases.mlir`,
`../functional/round-to-odd.mlir` and `../functional/mmad-accumulator-width.mlir`
pin these three so they stay fixed without anyone having to rerun the sweeps.
