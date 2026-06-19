//! Conjugate-gradient least-squares trainer (Edax `eval_builder` method).
//!
//! The eval model is **linear**: `prediction = Σ feature weights`, with weights
//! bucketed per empties value (`Weights`). For a fixed bucket this is a convex
//! least-squares problem with a unique optimum in prediction space, so we fit it
//! directly with **conjugate gradient + exact (closed-form) line search** instead
//! of hand-tuned SGD — no learning rate, deterministic convergence to the true
//! minimum. This mirrors Edax's `eval_builder_conjugate_gradient`
//! (`src/eval_builder.c`): per-ply fit, Polak-Ribière directions, frequency-
//! normalized gradient with rare-pattern damping/zeroing.
//!
//! Each empties bucket is an **independent** problem (its weights only affect
//! positions with that empties count), so the 61 buckets are solved separately
//! and in parallel — no cross-bucket coupling, no SGD-style merge.
//!
//! ## Tied weights and multiplicity
//!
//! Weights are tied by symmetry shape, so one physical weight can be hit several
//! times by a single position (e.g. all four corners sharing a config → coeff 4).
//! Every per-example slot list keeps these **repeats**, so prediction, gradient,
//! and the line-search directional derivative all carry the correct integer
//! multiplicity (exactly as Edax's `feature[i][j]` loop does).

use crate::training::weights::{empty_range_index, Weights, EMPTY_RANGE_COUNT};
use crate::training::TrainingExample;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Configuration for the conjugate-gradient least-squares fit.
#[derive(Debug, Clone)]
pub struct CgConfig {
    /// Maximum CG iterations per bucket.
    pub max_iter: usize,
    /// Minimum iterations before the convergence test can stop a bucket.
    pub min_iter: usize,
    /// Convergence threshold on `|RMSE_prev − RMSE_now|` (discs).
    pub accuracy: f64,
    /// CG restart frequency (steepest-descent restart every N iters; 0 = never).
    pub restart: usize,
    /// Ridge (L2) coefficient, **per example** — keeps each bucket strictly
    /// convex/well-conditioned and is the principled weight regularizer (0 = off).
    /// Scale-invariant: internally applied as `ridge·N` per bucket so one value
    /// transfers across data sizes (see `solve_bucket`). Critical here because
    /// each bucket is heavily underdetermined (≈115k weights ≫ examples).
    pub ridge: f64,
    /// Frequency floor: weights whose config appears `< min_count` times across
    /// the bucket are frozen at their initial value (gradient zeroed). Edax uses 3.
    pub min_count: u32,
    /// Enable Edax frequency-normalized gradient (diagonal preconditioner:
    /// `g_k ·= 1/N_k`, with rare configs damped/zeroed). When off, plain gradient.
    pub freq_norm: bool,
    /// Threads for **across-bucket** parallelism (buckets are independent).
    pub threads: usize,
    /// Verbose debug: print a per-bucket, per-iteration convergence log instead of
    /// the default single-line `\r` progress bar. Off by default (CLI always off).
    pub verbose: bool,
}

impl Default for CgConfig {
    fn default() -> Self {
        CgConfig {
            max_iter: 200,
            min_iter: 1,
            accuracy: 1e-4,
            restart: 50,
            ridge: 1e-3,
            min_count: 3,
            freq_norm: true,
            threads: 1,
            verbose: false,
        }
    }
}

/// Compiled examples for a single empties bucket.
///
/// `slots` is the flattened per-example weight-slot lists: example `i` occupies
/// `slots[i*stride .. (i+1)*stride]`, holding `stride` slot indices **with
/// repeats** (one per active feature instance). `targets[i]` is its label.
struct BucketData {
    stride: usize,
    slots: Vec<u32>,
    targets: Vec<f64>,
}

impl BucketData {
    fn n_examples(&self) -> usize {
        self.targets.len()
    }
}

/// Fit `weights` to `examples` by per-bucket conjugate-gradient least-squares.
///
/// Replaces the SGD epoch loop: extracts feature slots once, groups examples by
/// empties bucket, then solves each bucket independently (optionally across
/// `config.threads` threads) and writes the result back.
pub fn train_least_squares(weights: &mut Weights, examples: &[TrainingExample], config: &CgConfig) {
    let features = weights.features().clone();
    let stride = features.count();
    let offsets = weights.shape_offsets();
    let feature_to_shape = weights.feature_to_shape().to_vec();

    // Compile: one slot list per example, grouped by empties bucket.
    let mut buckets: Vec<BucketData> = (0..EMPTY_RANGE_COUNT)
        .map(|_| BucketData {
            stride,
            slots: Vec::new(),
            targets: Vec::new(),
        })
        .collect();

    let n = examples.len();
    let compile_start = Instant::now();
    for (i, ex) in examples.iter().enumerate() {
        let empties = ex.position.empties();
        let b = empty_range_index(empties);
        let indices = features.extract(&ex.position);
        let bucket = &mut buckets[b];
        for (f, &pattern) in indices.iter().enumerate() {
            let slot = offsets[feature_to_shape[f]] + pattern as usize;
            bucket.slots.push(slot as u32);
        }
        bucket.targets.push(ex.target_score as f64);
        if !config.verbose && i % 200_000 == 0 {
            print_cg_progress("compiling examples", i, n, compile_start);
        }
    }
    if !config.verbose {
        print_cg_progress("compiling examples", n, n, compile_start);
        eprintln!(); // finish the compile progress line
    }

    let total_nonempty = buckets.iter().filter(|b| b.n_examples() > 0).count();
    eprintln!(
        "CG least-squares: {} examples across {} buckets (stride={}, ridge={}/example, freq_norm={}, threads={})",
        n, total_nonempty, stride, config.ridge, config.freq_norm, config.threads,
    );
    let start = Instant::now();

    // Initial weight vectors per bucket (immutable read), solved into owned Vecs.
    let inits: Vec<Vec<f32>> = (0..EMPTY_RANGE_COUNT)
        .map(|b| weights.read_bucket(b))
        .collect();

    // Shared completed-bucket counter for the `\r` progress line (parallel-safe).
    let done = AtomicUsize::new(0);
    let n_threads = config.threads.max(1);
    let results: Vec<(usize, Vec<f32>)> = if n_threads <= 1 {
        (0..EMPTY_RANGE_COUNT)
            .map(|b| {
                let w = solve_bucket(&buckets[b], &inits[b], config, b);
                report_bucket(&buckets[b], &done, total_nonempty, start, config.verbose);
                (b, w)
            })
            .collect()
    } else {
        // Round-robin bucket indices across threads so work (uneven bucket sizes)
        // is spread out; each thread owns disjoint buckets and returns owned Vecs.
        std::thread::scope(|s| {
            let buckets = &buckets;
            let inits = &inits;
            let done = &done;
            let handles: Vec<_> = (0..n_threads)
                .map(|t| {
                    s.spawn(move || {
                        let mut out = Vec::new();
                        let mut b = t;
                        while b < EMPTY_RANGE_COUNT {
                            let w = solve_bucket(&buckets[b], &inits[b], config, b);
                            report_bucket(&buckets[b], done, total_nonempty, start, config.verbose);
                            out.push((b, w));
                            b += n_threads;
                        }
                        out
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        })
    };
    if !config.verbose {
        eprintln!(); // finish the bucket progress line
    }

    for (b, w) in results {
        weights.write_bucket(b, &w);
    }

    eprintln!(
        "CG least-squares complete | time: {:.1}s",
        start.elapsed().as_secs_f64()
    );
}

/// Count a finished (non-empty) bucket and refresh the `\r` solve-progress line.
/// No-op in `verbose` mode (the per-iteration log serves as progress there) and
/// for empty buckets (which solve instantly and shouldn't skew the ETA).
fn report_bucket(
    bucket: &BucketData,
    done: &AtomicUsize,
    total: usize,
    start: Instant,
    verbose: bool,
) {
    if verbose || bucket.n_examples() == 0 {
        return;
    }
    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
    print_cg_progress("solving buckets", d, total, start);
}

/// Single-line `\r` progress indicator with rate and ETA (matches the style of
/// `run_eval_check`/`build_examples` in the binary).
fn print_cg_progress(what: &str, done: usize, total: usize, start: Instant) {
    let elapsed = start.elapsed().as_secs_f64();
    let rate = done as f64 / elapsed.max(0.001);
    let pct = if total > 0 {
        done as f64 / total as f64 * 100.0
    } else {
        100.0
    };
    let eta = if rate > 0.0 {
        (total.saturating_sub(done)) as f64 / rate
    } else {
        0.0
    };
    eprint!("\r  {what} {done}/{total} ({pct:.0}%) | {rate:.0}/s | ETA {eta:.0}s        ");
    let _ = std::io::stderr().flush();
}

/// Solve a single empties bucket: conjugate gradient with exact line search.
///
/// Objective: `J(w) = Σ_i (target_i − pred_i)² + ridge·‖w‖²`, `pred_i = Σ_{slots_i} w`.
/// Returns the fitted flat weight vector (same layout/length as `init`). Buckets
/// with no examples return `init` unchanged.
fn solve_bucket(data: &BucketData, init: &[f32], config: &CgConfig, bucket: usize) -> Vec<f32> {
    let i_count = data.n_examples();
    let k = init.len();
    let stride = data.stride;
    if i_count == 0 || k == 0 {
        return init.to_vec();
    }

    let slots = &data.slots;
    let targets = &data.targets;

    // Weights in f64 during the solve (f32 storage round-trips on write-back).
    let mut w: Vec<f64> = init.iter().map(|&x| x as f64).collect();

    // Per-slot occurrence counts N_k (instances hitting weight k), for the
    // frequency preconditioner / rare-config freezing.
    let mut nfreq = vec![0u32; k];
    for &slot in slots {
        nfreq[slot as usize] += 1;
    }

    // Residuals e_i = target_i − pred_i.
    let mut e = vec![0f64; i_count];
    for i in 0..i_count {
        let base = i * stride;
        let mut pred = 0.0;
        for j in 0..stride {
            pred += w[slots[base + j] as usize];
        }
        e[i] = targets[i] - pred;
    }

    let mut g = vec![0f64; k]; // preconditioned descent direction (−∇J, scaled)
    let mut prev_g = vec![0f64; k];
    let mut h = vec![0f64; k]; // conjugate direction
    let mut b_dir = vec![0f64; i_count]; // directional derivative per example

    // Ridge is specified **per-example** so it is scale-invariant across data
    // sizes. Our data term is an implicit *sum* of squared errors over the
    // bucket's `i_count` examples, so minimizing `mean_MSE + ridge·‖w‖²` is the
    // same (up to a constant factor) as minimizing the sum form with an
    // effective ridge of `ridge·N`. Without this, the data term grows with N
    // while a fixed ridge does not, so the same ridge would mean wildly
    // different regularization at 100 files vs the full corpus (and per bucket).
    let ridge = config.ridge * i_count as f64;

    let mut rmse_prev = rmse(&e);

    for iter in 1..=config.max_iter {
        // Negative gradient of J, preconditioned by config frequency factor.
        //   data part: −∂E/∂w_k = 2 Σ_i e_i·c_ik   (c_ik = #slots of i equal to k)
        //   ridge part: −2·ridge·w_k
        g.fill(0.0);
        for i in 0..i_count {
            let ei = e[i];
            let base = i * stride;
            for j in 0..stride {
                g[slots[base + j] as usize] += ei;
            }
        }
        for kk in 0..k {
            let grad = 2.0 * g[kk] - 2.0 * ridge * w[kk];
            let n = nfreq[kk];
            let precond = if !config.freq_norm {
                1.0
            } else if n < config.min_count {
                0.0
            } else if n < 20 {
                0.1
            } else {
                1.0 / n as f64
            };
            g[kk] = grad * precond;
        }

        // Polak-Ribière conjugate direction (with PR+ safeguard and restarts).
        let gamma = if iter == 1 || (config.restart != 0 && iter % config.restart == 1) {
            0.0
        } else {
            let mut num = 0.0;
            let mut den = 0.0;
            for kk in 0..k {
                num += g[kk] * (g[kk] - prev_g[kk]);
                den += prev_g[kk] * prev_g[kk];
            }
            if den < f64::EPSILON {
                0.0
            } else {
                (num / den).max(0.0)
            }
        };
        for kk in 0..k {
            h[kk] = g[kk] + gamma * h[kk];
        }

        // Exact line search: minimize J(w + λ·h).
        //   λ = (Σ e_i b_i − ridge Σ w_k h_k) / (Σ b_i² + ridge Σ h_k²)
        // where b_i = Σ_{slots_i} h  (directional derivative of pred_i).
        let mut a_acc = 0.0; // Σ e_i b_i
        let mut bb_acc = 0.0; // Σ b_i²
        for i in 0..i_count {
            let base = i * stride;
            let mut bi = 0.0;
            for j in 0..stride {
                bi += h[slots[base + j] as usize];
            }
            b_dir[i] = bi;
            a_acc += e[i] * bi;
            bb_acc += bi * bi;
        }
        let mut wh = 0.0; // Σ w_k h_k
        let mut hh = 0.0; // Σ h_k²
        for kk in 0..k {
            wh += w[kk] * h[kk];
            hh += h[kk] * h[kk];
        }
        let denom = bb_acc + ridge * hh;
        if denom <= 0.0 {
            break;
        }
        let lambda = (a_acc - ridge * wh) / denom;
        if !lambda.is_finite() || lambda == 0.0 {
            break;
        }

        // Update weights and residuals incrementally.
        for kk in 0..k {
            w[kk] += lambda * h[kk];
        }
        for i in 0..i_count {
            e[i] -= lambda * b_dir[i];
        }

        prev_g.copy_from_slice(&g);

        let rmse_now = rmse(&e);
        if config.verbose {
            eprintln!(
                "  bucket {bucket:2} iter {iter:3} | RMSE {rmse_now:8.4} | λ {lambda:.6} | γ {gamma:.4}"
            );
        }
        if iter > config.min_iter && (rmse_prev - rmse_now).abs() <= config.accuracy {
            break;
        }
        rmse_prev = rmse_now;
    }

    w.iter().map(|&x| x as f32).collect()
}

/// Root-mean-square of the residual vector.
fn rmse(e: &[f64]) -> f64 {
    if e.is_empty() {
        return 0.0;
    }
    (e.iter().map(|&x| x * x).sum::<f64>() / e.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quiet(ridge: f64, freq_norm: bool) -> CgConfig {
        CgConfig {
            max_iter: 500,
            min_iter: 1,
            accuracy: 1e-9,
            restart: 50,
            ridge,
            min_count: 1,
            freq_norm,
            threads: 1,
            verbose: false,
        }
    }

    fn max_abs_residual(data: &BucketData, w: &[f32]) -> f64 {
        let stride = data.stride;
        let mut m = 0.0f64;
        for i in 0..data.n_examples() {
            let base = i * stride;
            let mut pred = 0.0;
            for j in 0..stride {
                pred += w[data.slots[base + j] as usize] as f64;
            }
            m = m.max((data.targets[i] - pred).abs());
        }
        m
    }

    #[test]
    fn recovers_linear_least_squares() {
        // True model over 3 weights; each example sums two distinct slots.
        // pred = w[s0] + w[s1]; targets generated from w* = [1, -2, 0.5].
        let w_star = [1.0f64, -2.0, 0.5];
        let pairs = [(0usize, 1usize), (0, 2), (1, 2), (0, 1), (1, 2), (0, 2)];
        let mut slots = Vec::new();
        let mut targets = Vec::new();
        for &(s0, s1) in &pairs {
            slots.push(s0 as u32);
            slots.push(s1 as u32);
            targets.push(w_star[s0] + w_star[s1]);
        }
        let data = BucketData {
            stride: 2,
            slots,
            targets,
        };
        let init = vec![0.0f32; 3];
        // No ridge / no freq-norm: must drive residuals to ~0 (consistent system).
        let w = solve_bucket(&data, &init, &quiet(0.0, false), 0);
        assert!(
            max_abs_residual(&data, &w) < 1e-4,
            "residual too large: {}",
            max_abs_residual(&data, &w)
        );
    }

    #[test]
    fn handles_slot_multiplicity() {
        // A single weight hit TWICE by one example: pred = 2·w0, target = 4 ⇒ w0 = 2.
        // Exercises that repeats carry coefficient 2 in pred, gradient, and the
        // line-search directional derivative.
        let data = BucketData {
            stride: 2,
            slots: vec![0, 0],
            targets: vec![4.0],
        };
        let init = vec![0.0f32];
        let w = solve_bucket(&data, &init, &quiet(0.0, false), 0);
        assert!((w[0] - 2.0).abs() < 1e-4, "w0 = {} (expected 2.0)", w[0]);
        assert!(max_abs_residual(&data, &w) < 1e-4);
    }

    #[test]
    fn empty_bucket_returns_init_unchanged() {
        let data = BucketData {
            stride: 2,
            slots: Vec::new(),
            targets: Vec::new(),
        };
        let init = vec![1.5f32, -3.0, 0.25];
        let w = solve_bucket(&data, &init, &quiet(1e-3, true), 0);
        assert_eq!(w, init);
    }

    #[test]
    fn frequency_floor_freezes_rare_weights() {
        // min_count = 2: weight 1 appears once ⇒ frozen at its init; weight 0
        // appears 3 times ⇒ free. With freq_norm on, w[1] must stay put.
        let data = BucketData {
            stride: 1,
            slots: vec![0, 0, 0, 1],
            targets: vec![2.0, 2.0, 2.0, 9.0],
        };
        let init = vec![0.0f32, 7.0];
        let mut cfg = quiet(0.0, true);
        cfg.min_count = 2;
        let w = solve_bucket(&data, &init, &cfg, 0);
        assert!((w[1] - 7.0).abs() < 1e-9, "rare weight moved: {}", w[1]);
        assert!((w[0] - 2.0).abs() < 1e-3, "free weight wrong: {}", w[0]);
    }

    #[test]
    fn ridge_shrinks_toward_zero() {
        // Single weight, target 10 from one example (pred = w0).
        // Ridge minimizes (10 − w0)² + ridge·w0² ⇒ w0 = 10/(1+ridge) < 10.
        let data = BucketData {
            stride: 1,
            slots: vec![0],
            targets: vec![10.0],
        };
        let init = vec![0.0f32];
        let ridge = 0.5;
        let w = solve_bucket(&data, &init, &quiet(ridge, false), 0);
        let expected = 10.0 / (1.0 + ridge);
        assert!(
            (w[0] as f64 - expected).abs() < 1e-3,
            "w0 = {} (expected {expected})",
            w[0]
        );
    }

    #[test]
    fn ridge_is_scale_invariant_in_n() {
        // Per-example ridge ⇒ duplicating every example must NOT change the fit.
        // Single weight, (slots=[0], target=10), ridge=0.5: minimizing
        // mean_MSE + ridge·w² gives w = 10/(1+ridge) regardless of how many
        // identical copies of the example we feed.
        let ridge = 0.5;
        let expected = 10.0 / (1.0 + ridge);
        let one = BucketData {
            stride: 1,
            slots: vec![0],
            targets: vec![10.0],
        };
        let many = BucketData {
            stride: 1,
            slots: vec![0; 50],
            targets: vec![10.0; 50],
        };
        let w1 = solve_bucket(&one, &[0.0f32], &quiet(ridge, false), 0);
        let w50 = solve_bucket(&many, &[0.0f32], &quiet(ridge, false), 0);
        assert!((w1[0] as f64 - expected).abs() < 1e-3, "N=1: {}", w1[0]);
        assert!((w50[0] as f64 - expected).abs() < 1e-3, "N=50: {}", w50[0]);
        assert!(
            (w1[0] - w50[0]).abs() < 1e-3,
            "N changed the fit: {} vs {}",
            w1[0],
            w50[0]
        );
    }
}
