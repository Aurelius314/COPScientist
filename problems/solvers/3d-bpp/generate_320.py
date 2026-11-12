"""Export the 320 Martello-Pisinger-Vigo 3D-BPP benchmark instances.

This is a generator-only port of test3dbpp.c for classes 1-8.  Its RNG
emulates the original HP9000/32-bit unsigned-long arithmetic, so the
instances do not depend on Python's or the host C library's RNG.
"""

from __future__ import annotations

import argparse
from pathlib import Path


MASK32 = (1 << 32) - 1
SIZES = (50, 100, 150, 200)
REPLICATES = 10  # TESTS in test3dbpp.c
BIN_SIZES = {1: 100, 2: 100, 3: 100, 4: 100, 5: 100,
             6: 10, 7: 40, 8: 100}


class OriginalRandom:
    """srand48x/lrand48x/rint from test3dbpp.c with 32-bit wraparound."""

    def __init__(self, seed: int) -> None:
        self.high = seed & MASK32
        self.low = 0x330E

    def randint(self, start: int, end: int) -> int:
        self.high = (self.high * 0xDEECE66D + self.low * 0x5DEEC) & MASK32
        low = self.low * 0xE66D + 0xB
        self.high = (self.high + (low >> 16)) & MASK32
        self.low = low & 0xFFFF
        return (self.high >> 1) % (end - start + 1) + start


def make_item(rng: OriginalRandom, bin_size: int, instance_class: int) -> tuple[int, int, int]:
    # randomtype() in test3dbpp.c: classes 1-5 mix the five item types.
    item_type = instance_class
    if instance_class <= 5:
        sampled_type = rng.randint(1, 10)
        if sampled_type <= 5:
            item_type = sampled_type

    b = bin_size
    if item_type == 1:
        return rng.randint(1, b // 2), rng.randint(2 * b // 3, b), rng.randint(2 * b // 3, b)
    if item_type == 2:
        return rng.randint(2 * b // 3, b), rng.randint(1, b // 2), rng.randint(2 * b // 3, b)
    if item_type == 3:
        return rng.randint(2 * b // 3, b), rng.randint(2 * b // 3, b), rng.randint(1, b // 2)
    if item_type == 4:
        return rng.randint(b // 2, b), rng.randint(b // 2, b), rng.randint(b // 2, b)
    if item_type == 5:
        return rng.randint(1, b // 2), rng.randint(1, b // 2), rng.randint(1, b // 2)
    if item_type == 6:
        return rng.randint(1, 10), rng.randint(1, 10), rng.randint(1, 10)
    if item_type == 7:
        return rng.randint(1, 35), rng.randint(1, 35), rng.randint(1, 35)
    return rng.randint(1, 100), rng.randint(1, 100), rng.randint(1, 100)


def generate(output_dir: Path) -> int:
    count = 0
    for instance_class, bin_size in BIN_SIZES.items():
        for n in SIZES:
            for replicate in range(1, REPLICATES + 1):
                # The original main() calls srand(v+n) before each maketest().
                rng = OriginalRandom(replicate + n)
                items = [make_item(rng, bin_size, instance_class) for _ in range(n)]
                path = output_dir / f"CLASS_{instance_class}" / f"N{n}" / f"instance_{replicate:02d}.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                lines = [f"{n} {bin_size} {bin_size} {bin_size}"]
                lines.extend(f"{w} {h} {d}" for w, h, d in items)
                path.write_text("\n".join(lines) + "\n", encoding="ascii")
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "instances_320")
    args = parser.parse_args()
    count = generate(args.output)
    print(f"Generated {count} instances in {args.output.resolve()}")


if __name__ == "__main__":
    main()
