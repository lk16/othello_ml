//! Mobility computation: the bitboard of legal moves for `player` against
//! `opponent`. Called at (almost) every interior search node, so a hot primitive.
//!
//! All variants share the signature `fn(player: u64, opponent: u64) -> u64`.
//! They are benchmarked against each other (see `docs/speedup-plan.md` Step 24);
//! the production entry point [`get_moves`] is the portable scalar variant.
//!
//! - [`scalar`] — branchless 8-direction Kogge-Stone fill (portable).
//! - `avx2` — four ray axes in parallel across 256-bit lanes (x86-64 only).

mod scalar;

#[cfg(target_arch = "x86_64")]
mod avx2;

// Production entry point: the portable scalar fill. The AVX2 variant is kept for
// `bench-get-moves` and future per-target tuning, but is not auto-selected — same
// reasoning as the flip primitive (see `othello::flip`): production is baseline
// x86-64, and a SIMD pick must be measured per target before wiring. Re-run
// `bench-get-moves` on an AVX2 box and `cfg`-gate before any override.
pub(crate) use scalar::get_moves;

/// Micro-benchmark every compiled mobility variant over realistic boards
/// (positions sampled from random self-play), printing ns/call. Invoked by the
/// `bench-get-moves` subcommand. SIMD variants are timed only when the running
/// CPU supports them.
#[allow(unsafe_code)]
pub fn bench_get_moves_variants() {
    use crate::othello::position::Position;
    use std::hint::black_box;
    use std::time::Instant;

    // Collect (player, opponent) boards from random self-play.
    let mut samples: Vec<(u64, u64)> = Vec::with_capacity(200_000);
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
            samples.push((pos.player, pos.opponent));
            if samples.len() >= 200_000 {
                break;
            }
            let moves = pos.get_moves();
            if moves == 0 {
                let passed = pos.pass_move();
                if passed.get_moves() == 0 {
                    break;
                }
                pos = passed;
                continue;
            }
            let pick = next() % moves.count_ones() as u64;
            let mut m = moves;
            for _ in 0..pick {
                m &= m - 1;
            }
            pos = pos.do_move(m.trailing_zeros());
        }
    }

    const REPEATS: usize = 300;
    let total = (samples.len() * REPEATS) as f64;

    macro_rules! time_variant {
        ($name:expr, $f:expr) => {{
            let mut acc = 0u64;
            for &(p, o) in &samples {
                acc ^= $f(p, o);
            }
            black_box(acc);
            let start = Instant::now();
            let mut acc = 0u64;
            for _ in 0..REPEATS {
                for &(p, o) in &samples {
                    acc ^= $f(p, o);
                }
            }
            black_box(acc);
            eprintln!(
                "  {:<12} {:.3} ns/call",
                $name,
                start.elapsed().as_nanos() as f64 / total
            );
        }};
    }

    eprintln!(
        "get_moves micro-bench: {} boards x {} repeats",
        samples.len(),
        REPEATS
    );
    time_variant!("scalar", scalar::get_moves);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            time_variant!("avx2", |p, o| unsafe { avx2::get_moves(p, o) });
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

    /// Run `variant` against the proven `scalar` reference over a structured
    /// pattern battery plus a large random fuzz, and assert equality.
    fn check_against_reference(name: &str, variant: impl Fn(u64, u64) -> u64) {
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
            0x8100_0000_0000_0081, // corners
            0xFF81_8181_8181_81FF, // edges
        ];
        // Structured: every pair of disjoint patterns.
        for &a in PATTERNS {
            for &b in PATTERNS {
                let player = a;
                let opponent = b & !a;
                assert_eq!(
                    variant(player, opponent),
                    scalar::get_moves(player, opponent),
                    "{name}: structured mismatch player={player:#x} opponent={opponent:#x}"
                );
            }
        }

        // Random fuzz: disjoint player/opponent.
        let mut rng = Rng(0x0123_4567_89AB_CDEF);
        for _ in 0..1_000_000 {
            let player = rng.next();
            let opponent = rng.next() & !player;
            assert_eq!(
                variant(player, opponent),
                scalar::get_moves(player, opponent),
                "{name}: fuzz mismatch player={player:#x} opponent={opponent:#x}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    #[allow(unsafe_code)]
    fn avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("skipping avx2 get_moves test: CPU lacks AVX2");
            return;
        }
        check_against_reference("avx2", |p, o| unsafe { avx2::get_moves(p, o) });
    }
}
