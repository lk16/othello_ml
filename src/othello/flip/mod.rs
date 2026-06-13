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

mod generic;
mod specialized;

#[cfg(target_arch = "x86_64")]
mod bmi2;
#[cfg(target_arch = "x86_64")]
mod line;

// Production entry point. Baseline x86-64 (and non-x86) uses the per-square
// specialization; richer targets are wired up in a later step.
pub(crate) use specialized::flip;

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
    #[cfg(target_arch = "x86_64")]
    #[allow(unsafe_code)]
    fn bmi2_matches_specialized() {
        if !is_x86_feature_detected!("bmi2") {
            eprintln!("skipping bmi2 flip test: CPU lacks BMI2");
            return;
        }
        check_against_reference("bmi2", |mv, p, o| unsafe { bmi2::flip(mv, p, o) });
    }
}
