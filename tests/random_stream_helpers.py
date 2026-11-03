"""SplitMix64 on Python integers, from its published constants (Steele, Lea and Flood 2014).

Seeded results draw from counter-based streams: a key mixes a seed with a salt
and with the numbers of what it draws for, and draw ``i`` of a key mixes the
key plus ``i`` golden-ratio increments. These helpers compute the same
numbers without NumPy or Numba, so a test can pin a stream to the bit.
"""

MASK = 2**64 - 1
GOLDEN = 0x9E3779B97F4A7C15


def splitmix64(value: int) -> int:
    """SplitMix64's finaliser of a 64-bit integer."""
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
    return value ^ (value >> 31)


def child_key(key: int, value: int) -> int:
    """The key that mixes ``key`` with a non-negative integer ``value``."""
    return splitmix64(key ^ splitmix64((value + GOLDEN) & MASK))


def draw(key: int, counter: int) -> int:
    """Draw ``counter`` of stream ``key``: the finaliser of the key plus ``counter`` increments."""
    return splitmix64((key + counter * GOLDEN) & MASK)


def uniform(key: int, counter: int) -> float:
    """Draw ``counter`` of stream ``key`` as a number in [0, 1), from its top 53 bits."""
    return (draw(key, counter) >> 11) * 2.0**-53
