"""Bit-exact f16 / bf16 / f32 / f8 helpers plus a minimal .npy reader/writer.

No numpy is assumed, so everything is done with struct + integer maths. Python
floats are IEEE double; for f8 (2- or 3-bit), bf16 (8-bit), f16 (11-bit) and
f32 (24-bit) significands, a single double operation followed by one
round-to-nearest-even into the narrow format is correctly rounded - safe double
rounding needs 2p+2 bits of intermediate and double carries 53 - so double is a
valid oracle for + - * / and sqrt.

The formats differ in more than width. f8e4m3fn has no infinity: its only NaN
encodings are 0x7f and 0xff, so the all-ones exponent is an ordinary binade
except for its last pattern, its largest finite value is 448 rather than the
240 an IEEE-shaped reading would give, and overflow produces NaN instead of
infinity. f8e5m2 is IEEE-shaped. An oracle that gets this wrong disagrees with
the interpreter about every large value, so `Fmt` models the non-finite
behaviour explicitly.
"""
import math
import struct

# ---------------------------------------------------------------- formats

#: All-ones exponent means Inf (mantissa 0) or NaN (mantissa nonzero).
IEEE754 = "ieee754"
#: No infinity: the single all-ones pattern is NaN, everything else is finite,
#: and an operation that would overflow produces NaN.
NAN_ONLY = "nan_only"


class Fmt:
    def __init__(self, name, ebits, mbits, nbytes, nonfinite=IEEE754):
        self.name = name
        self.ebits = ebits
        self.mbits = mbits
        self.nbytes = nbytes
        self.nonfinite = nonfinite
        self.bias = (1 << (ebits - 1)) - 1
        self.emin = 1 - self.bias
        self.sign_bit = 1 << (ebits + mbits)
        self.payload_mask = self.sign_bit - 1
        self.top_exp = (1 << ebits) - 1
        self.max_man = (1 << mbits) - 1
        if nonfinite == IEEE754:
            self.max_finite_bits = ((self.top_exp - 1) << mbits) | self.max_man
            self.nan_bits = (self.top_exp << mbits) | (1 << (mbits - 1))
            self.inf_bits = self.top_exp << mbits
        else:
            self.max_finite_bits = (self.top_exp << mbits) | (self.max_man - 1)
            self.nan_bits = (self.top_exp << mbits) | self.max_man
            self.inf_bits = None

    # -- decoding ---------------------------------------------------------

    def is_nan(self, bits):
        payload = bits & self.payload_mask
        exp = payload >> self.mbits
        man = payload & self.max_man
        if exp != self.top_exp:
            return False
        return man != 0 if self.nonfinite == IEEE754 else man == self.max_man

    def is_inf(self, bits):
        if self.nonfinite != IEEE754:
            return False
        return (bits & self.payload_mask) == self.inf_bits

    def bits_to_float(self, bits):
        """Exact value of a bit pattern, as a Python float."""
        sign = -1.0 if bits & self.sign_bit else 1.0
        if self.is_nan(bits):
            return float("nan")
        if self.is_inf(bits):
            return sign * float("inf")
        payload = bits & self.payload_mask
        exp = payload >> self.mbits
        man = payload & self.max_man
        if exp == 0:
            return sign * man * 2.0 ** (self.emin - self.mbits)
        return sign * (man + (1 << self.mbits)) * \
            2.0 ** (exp - self.bias - self.mbits)

    def max_finite(self):
        return self.bits_to_float(self.max_finite_bits)

    # -- neighbours -------------------------------------------------------

    def next_away(self, bits):
        """Next pattern away from zero, or None past the largest finite."""
        payload = bits & self.payload_mask
        if payload >= self.max_finite_bits:
            return None
        return (bits & self.sign_bit) | (payload + 1)

    def next_toward(self, bits):
        """Next pattern toward zero; the zeros stay put."""
        payload = bits & self.payload_mask
        if payload == 0:
            return bits
        return (bits & self.sign_bit) | (payload - 1)

    def bracket(self, x):
        """(below, above) finite patterns with below <= x <= above.

        Both ends are the same pattern when x is exactly representable, and an
        end is None when x lies past the finite range on that side.
        """
        if x != x or abs(x) == float("inf"):
            return None, None
        if abs(x) > self.max_finite():
            # Past the finite range on one side; the other side is the largest
            # finite value of that sign.
            if x > 0:
                return self.max_finite_bits, None
            return None, self.sign_bit | self.max_finite_bits

        near = self._round_nearest(x, ties_even=True)
        v = self.bits_to_float(near)
        if v == x:
            return near, near
        neg = _signbit(x)
        # `near` is one step off; the other neighbour is one step in the
        # direction that crosses x. Stepping is in magnitude, so which of
        # next_away / next_toward moves "up" depends on the sign.
        if v < x:
            other = self.next_toward(near) if neg else self.next_away(near)
            return near, other
        other = self.next_away(near) if neg else self.next_toward(near)
        return other, near

    # -- rounding ---------------------------------------------------------

    def _overflow(self, neg):
        """What the format produces for a magnitude past the largest finite."""
        sign = self.sign_bit if neg else 0
        if self.nonfinite == IEEE754:
            return sign | self.inf_bits
        return sign | self.nan_bits

    def _round_nearest(self, x, ties_even):
        if x != x:
            return self.nan_bits
        neg = _signbit(x)
        sign = self.sign_bit if neg else 0
        ax = abs(x)
        if ax == float("inf"):
            return self._overflow(neg)
        if ax == 0.0:
            return sign

        m, e = math.frexp(ax)
        e -= 1
        m *= 2.0                                      # 1 <= m < 2

        if e < self.emin:                             # subnormal range
            q = 2.0 ** (self.emin - self.mbits)
            n = _round_int(ax / q, ties_even)
            if n > self.max_finite_bits:
                return self._overflow(neg)
            return sign | int(n)

        n = _round_int(m * (1 << self.mbits), ties_even)
        if n == (1 << (self.mbits + 1)):              # carried into next binade
            n >>= 1
            e += 1
        payload = ((e + self.bias) << self.mbits) | (int(n) - (1 << self.mbits))
        if payload > self.max_finite_bits:
            return self._overflow(neg)
        return sign | payload

    def round(self, x, mode="rint"):
        """Round `x` into this format under a HIVM rounding mode.

        `rint` is nearest-ties-to-even, `round` is nearest-ties-away,
        `floor` / `ceil` / `trunc` are the directed modes, `odd` is
        round-to-odd.
        """
        if x != x:
            return self.nan_bits
        if mode == "rint":
            return self._round_nearest(x, ties_even=True)
        if mode == "round":
            return self._round_nearest(x, ties_even=False)

        neg = _signbit(x)
        if abs(x) == float("inf"):
            return self._overflow(neg)

        lower, upper = self.bracket(x)
        if lower is not None and lower == upper:
            return lower                              # exact
        if mode == "floor":
            return lower if lower is not None else self._overflow(True)
        if mode == "ceil":
            return upper if upper is not None else self._overflow(False)
        if mode == "trunc":
            # Toward zero can never overflow: the inner neighbour is finite.
            inner = upper if neg else lower
            return inner if inner is not None else \
                ((self.sign_bit if neg else 0) | self.max_finite_bits)
        if mode == "odd":
            # The bracketing neighbour whose significand is odd. Past the
            # finite range that is the largest finite value, whose significand
            # is all ones and so already odd.
            for cand in (lower, upper):
                if cand is not None and (cand & 1):
                    return cand
            return (self.sign_bit if neg else 0) | self.max_finite_bits
        raise ValueError("unknown rounding mode " + mode)


def _signbit(x):
    return struct.pack("<d", x)[7] & 0x80 != 0


def _round_int(v, ties_even):
    """Round a non-negative float to an integer under the given tie rule."""
    f = math.floor(v)
    diff = v - f
    if diff > 0.5:
        return f + 1
    if diff < 0.5:
        return f
    if ties_even:
        return f if int(f) % 2 == 0 else f + 1
    return f + 1                                      # ties away from zero


F16 = Fmt("f16", 5, 10, 2)
BF16 = Fmt("bf16", 8, 7, 2)
F32 = Fmt("f32", 8, 23, 4)
F8E4M3 = Fmt("f8e4m3fn", 4, 3, 1, nonfinite=NAN_ONLY)
F8E5M2 = Fmt("f8e5m2", 5, 2, 1)

# ------------------------------------------------------------------- npy


def write_npy(path, dtype, shape, payload):
    header = "{'descr': '%s', 'fortran_order': False, 'shape': (%s), }" % (
        dtype,
        ",".join(str(s) for s in shape) + ("," if len(shape) == 1 else ""))
    prefix = 10 + len(header) + 1
    header += " " * ((64 - prefix % 64) % 64) + "\n"
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY" + bytes([1, 0]))
        f.write(struct.pack("<H", len(header)))
        f.write(header.encode())
        f.write(payload)


def read_npy(path):
    with open(path, "rb") as f:
        blob = f.read()
    hlen = struct.unpack("<H", blob[8:10])[0]
    header = blob[10:10 + hlen].decode()
    dtype = header.split("'descr':")[1].split("'")[1]
    return dtype, blob[10 + hlen:]


def pack_bits(values, nbytes):
    out = bytearray()
    for v in values:
        out += int(v).to_bytes(nbytes, "little")
    return bytes(out)


def unpack_bits(payload, nbytes):
    return [int.from_bytes(payload[i:i + nbytes], "little")
            for i in range(0, len(payload), nbytes)]
