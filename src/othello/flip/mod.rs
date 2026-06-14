//! Flip computation: the discs `player` captures by playing at `mv`.
//!
//! All variants share the signature `fn(mv: u32, player: u64, opponent: u64) ->
//! u64` and assume `mv` is empty. They are benchmarked against each other (see
//! `docs/speedup-plan.md` Step 11); the production entry point [`flip`] is
//! selected per target by `cfg(target_feature)`.
//!
//! - [`specialized`] — per-square const-generic ray-scan (function-pointer table).
//! - [`generic`] — runtime inlinable ray-scan.
//! - `carry64` — portable line-table (gather/lookup/scatter).
//! - `bmi2` / `avx2` — x86-64 SIMD variants (compiled only on x86-64).

mod carry64;
mod generic;
mod line;
mod specialized;

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod bmi2;

// Production entry point: the portable carry-64 line-table variant. It was the
// fastest in the Step 11 benchmark (see docs/speedup-plan.md) — about 2.5x the
// per-square specialization on the AMD dev box, and faster there than the SIMD
// variants. BMI2 PEXT/PDEP is microcoded and ~20x slower on AMD, and
// `cfg(target_feature = "bmi2")` cannot tell fast Intel PEXT from slow AMD PEXT,
// so we do not auto-select it; carry64 is the robust portable default.
pub(crate) use carry64::flip;

/// Micro-benchmark every compiled flip variant over realistic flip sites
/// (every legal move of positions sampled from random self-play), printing
/// ns/flip. Invoked by the `bench-flip` subcommand. SIMD variants are timed
/// only when the running CPU supports them.
#[allow(unsafe_code)]
pub(crate) fn bench_variants() {
    use crate::othello::position::Position;
    use std::hint::black_box;
    use std::time::Instant;

    // Collect (move, player, opponent) flip sites from random self-play.
    let mut samples: Vec<(u32, u64, u64)> = Vec::with_capacity(200_000);
    let mut rng: u64 = 0x1234_5678_9ABC_DEF1;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    while samples.len() < 200_000 {
        let mut pos = Position::initial();
        loop {
            let moves = pos.get_moves();
            if moves == 0 {
                let passed = pos.pass_move();
                if passed.get_moves() == 0 {
                    break;
                }
                pos = passed;
                continue;
            }
            let mut m = moves;
            while m != 0 {
                let cell = m.trailing_zeros();
                m &= m - 1;
                samples.push((cell, pos.player, pos.opponent));
            }
            let pick = next() % moves.count_ones() as u64;
            let mut m = moves;
            for _ in 0..pick {
                m &= m - 1;
            }
            pos = pos.do_move(m.trailing_zeros());
            if samples.len() >= 200_000 {
                break;
            }
        }
    }

    const REPEATS: usize = 300;
    let total = (samples.len() * REPEATS) as f64;

    macro_rules! time_variant {
        ($name:expr, $f:expr) => {{
            let mut acc = 0u64;
            for &(mv, p, o) in &samples {
                acc ^= $f(mv, p, o);
            }
            black_box(acc);
            let start = Instant::now();
            let mut acc = 0u64;
            for _ in 0..REPEATS {
                for &(mv, p, o) in &samples {
                    acc ^= $f(mv, p, o);
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
        "flip micro-bench: {} sites x {} repeats",
        samples.len(),
        REPEATS
    );
    time_variant!("specialized", specialized::flip);
    time_variant!("generic", generic::flip);
    time_variant!("carry64", carry64::flip);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("bmi2") {
            time_variant!("bmi2", |mv, p, o| unsafe { bmi2::flip(mv, p, o) });
        }
        if is_x86_feature_detected!("avx2") {
            time_variant!("avx2", |mv, p, o| unsafe { avx2::flip(mv, p, o) });
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

    /// Run `variant` against the proven `specialized` reference over a structured
    /// square × pattern battery plus a large random fuzz, and assert equality.
    fn check_against_reference(name: &str, variant: impl Fn(u32, u64, u64) -> u64) {
        // Structured: every square × a spread of disc patterns (the move square
        // forced empty).
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
        for sq in 0u32..64 {
            let empty = 1u64 << sq;
            for &pat in PATTERNS {
                let player = pat & !empty;
                let opponent = !player & !empty;
                assert_eq!(
                    variant(sq, player, opponent),
                    specialized::flip(sq, player, opponent),
                    "{name}: structured mismatch at sq={sq} pat={pat:#x}"
                );
            }
        }

        // Random fuzz: disjoint player/opponent, move on an empty square.
        let mut rng = Rng(0x0123_4567_89AB_CDEF);
        for _ in 0..500_000 {
            let player = rng.next();
            let opponent = rng.next() & !player;
            let empties = !(player | opponent);
            if empties == 0 {
                continue;
            }
            // Pick a pseudo-random empty square.
            let sq = {
                let mut e = empties;
                let skip = (rng.next() % e.count_ones() as u64) as u32;
                for _ in 0..skip {
                    e &= e - 1;
                }
                e.trailing_zeros()
            };
            assert_eq!(
                variant(sq, player, opponent),
                specialized::flip(sq, player, opponent),
                "{name}: fuzz mismatch at sq={sq} player={player:#x} opponent={opponent:#x}"
            );
        }
    }

    #[test]
    fn generic_matches_specialized() {
        check_against_reference("generic", generic::flip);
    }

    #[test]
    fn carry64_matches_specialized() {
        check_against_reference("carry64", carry64::flip);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    #[allow(unsafe_code)]
    fn bmi2_matches_specialized() {
        if !is_x86_feature_detected!("bmi2") {
            eprintln!("skipping bmi2 flip test: CPU lacks BMI2");
            return;
        }
        check_against_reference("bmi2", |mv, p, o| unsafe { bmi2::flip(mv, p, o) });
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    #[allow(unsafe_code)]
    fn avx2_matches_specialized() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("skipping avx2 flip test: CPU lacks AVX2");
            return;
        }
        check_against_reference("avx2", |mv, p, o| unsafe { avx2::flip(mv, p, o) });
    }
}
