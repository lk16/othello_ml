//! Count-last-flip: 2× the discs `player` captures by playing at the board's
//! sole empty square — the hot primitive of the 1-empty leaf solver `solve_1`.
//!
//! All variants share the signature `fn(pos: u32, player: u64) -> i32` and
//! assume `pos` is the only empty square (so the opponent is `!player & !pos`).
//! They are benchmarked against each other (see `docs/speedup-plan.md` Step 23);
//! the production entry point [`count_last_flip`] is the portable table variant.
//!
//! - [`table`] — per-line `COUNT_FLIP` lookup, gathered by shift/multiply.
//! - `via_flip` — full flip mask via the production flip, then `2×popcount`.
//! - `bmi2` — x86-64 `PEXT` line gather (compiled only on x86-64).

mod table;
mod tables;
mod via_flip;

#[cfg(target_arch = "x86_64")]
mod bmi2;

// Production entry point: the portable per-line table variant. BMI2 PEXT is
// microcoded and slow on AMD, and `cfg(target_feature = "bmi2")` cannot tell
// fast Intel PEXT from slow AMD PEXT, so we do not auto-select it — the same
// reasoning as the flip primitive (see `othello::flip`). Re-run `bench-count-flip`
// on an Intel (Haswell+) box before wiring any per-target override.
pub(crate) use table::count_last_flip;

/// Micro-benchmark every compiled count-last-flip variant over near-full boards
/// (one empty square — the `solve_1` domain), printing ns/flip. Invoked by the
/// `bench-count-flip` subcommand. SIMD variants are timed only when the running
/// CPU supports them.
#[allow(unsafe_code)]
pub fn bench_count_flip_variants() {
    use std::hint::black_box;
    use std::time::Instant;

    // Collect (pos, player) samples: a random near-full board with exactly one
    // empty square, the invariant `count_last_flip` requires.
    let mut samples: Vec<(u32, u64)> = Vec::with_capacity(200_000);
    let mut rng: u64 = 0x1234_5678_9ABC_DEF1;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    while samples.len() < 200_000 {
        let pos = (next() % 64) as u32;
        let player = next() & !(1u64 << pos); // pos forced empty; rest is opponent
        samples.push((pos, player));
    }

    const REPEATS: usize = 300;
    let total = (samples.len() * REPEATS) as f64;

    macro_rules! time_variant {
        ($name:expr, $f:expr) => {{
            let mut acc = 0i32;
            for &(pos, p) in &samples {
                acc ^= $f(pos, p);
            }
            black_box(acc);
            let start = Instant::now();
            let mut acc = 0i32;
            for _ in 0..REPEATS {
                for &(pos, p) in &samples {
                    acc ^= $f(pos, p);
                }
            }
            black_box(acc);
            eprintln!(
                "  {:<12} {:.3} ns/flip",
                $name,
                start.elapsed().as_nanos() as f64 / total
            );
        }};
    }

    eprintln!(
        "count-last-flip micro-bench: {} sites x {} repeats",
        samples.len(),
        REPEATS
    );
    time_variant!("table", table::count_last_flip);
    time_variant!("via_flip", via_flip::count_last_flip);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("bmi2") {
            time_variant!("bmi2", |pos, p| unsafe { bmi2::count_last_flip(pos, p) });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic xorshift RNG (no external dependency).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    /// Run `variant` against the proven `table` reference over a structured
    /// square × pattern battery plus a large near-full fuzz, and assert equality.
    fn check_against_reference(name: &str, variant: impl Fn(u32, u64) -> i32) {
        const PATTERNS: &[u64] = &[
            0x0000_0000_0000_0000,
            0xFFFF_FFFF_FFFF_FFFF,
            0xAAAA_AAAA_AAAA_AAAA,
            0x5555_5555_5555_5555,
            0xFF00_FF00_FF00_FF00,
            0xF0F0_F0F0_F0F0_F0F0,
            0x8040_2010_0804_0201,
            0x0102_0408_1020_4080,
            0xC3C3_C3C3_C3C3_C3C3,
            0x1234_5678_9ABC_DEF0,
        ];
        // Structured: every square × a spread of player patterns (move forced empty).
        for pos in 0u32..64 {
            let empty = 1u64 << pos;
            for &pat in PATTERNS {
                let player = pat & !empty;
                assert_eq!(
                    variant(pos, player),
                    table::count_last_flip(pos, player),
                    "{name}: structured mismatch at pos={pos} pat={pat:#x}"
                );
            }
        }

        // Random fuzz: arbitrary player discs on a full board with one empty.
        let mut rng = Rng(0x0123_4567_89AB_CDEF);
        for _ in 0..500_000 {
            let pos = (rng.next() % 64) as u32;
            let player = rng.next() & !(1u64 << pos);
            assert_eq!(
                variant(pos, player),
                table::count_last_flip(pos, player),
                "{name}: fuzz mismatch at pos={pos} player={player:#x}"
            );
        }
    }

    #[test]
    fn via_flip_matches_table() {
        check_against_reference("via_flip", via_flip::count_last_flip);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    #[allow(unsafe_code)]
    fn bmi2_matches_table() {
        if !is_x86_feature_detected!("bmi2") {
            eprintln!("skipping bmi2 count-last-flip test: CPU lacks BMI2");
            return;
        }
        check_against_reference("bmi2", |pos, p| unsafe { bmi2::count_last_flip(pos, p) });
    }
}
