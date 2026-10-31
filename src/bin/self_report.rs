/*
    Self-Report, Stage 1: does a model have privileged access to its own loss?

    Two identical MicroGPTs, A and B, learn the same names from different seeds.
    On names neither has seen, each model's own loss is measured exactly. Then a
    linear readout (ridge regression) is fitted on hidden states to predict a
    model's loss:

      self   A's hidden states -> A's loss      (B's -> B's)
      cross  B's hidden states -> A's loss      (A's -> B's)

    B knows names as well as A does but has no access to A's internals, so
    whatever "self" predicts beyond "cross" is information A holds about its own
    processing that the text, and a sibling model of the text, do not carry.
    That margin is the stage-1 measure of privileged self-access.

    Baselines: name length (+ length^2) and bag-of-characters, the obvious text
    statistics behind "this name is hard".

    The readout is fitted on half of the probe names and scored on the other half
    (R^2, Pearson r); a bootstrap over the scored names gives a 95% interval on
    the self-minus-cross margin.

        cargo run --release --bin self_report -- [--steps 4000] [--emb 16] \
            [--layer 2] [--head 4] [--ctx 16] [--probe 2000] [--seed 7]
*/

use microgpt_rust::{build_vocab, linear, load_training_data, rmsnorm, softmax, Mat2, Val, Vec1, GPT};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use std::time::Instant;

struct Cfg {
    steps: usize, emb: usize, layer: usize, head: usize, ctx: usize, ff: usize,
    lr: f64, probe: usize, seed: u64, input: String,
    stage: usize, inject_layer: usize, sigma_max: f64,
    rho: Vec<f64>, report_steps: usize, twin: String,
    train_seed: Option<u64>, base_file: Option<String>,
}

fn parse_args() -> Cfg {
    let mut c = Cfg { steps: 4000, emb: 16, layer: 2, head: 4, ctx: 16, ff: 4, lr: 0.01,
                      probe: 2000, seed: 7, input: "input.txt".into(),
                      stage: 1, inject_layer: 0, sigma_max: 1.0,
                      rho: vec![0.0, 0.5, 0.9, 1.0], report_steps: 10000, twin: "ab".into(),
                      train_seed: None, base_file: None };
    let a: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i + 1 < a.len() {
        let v = &a[i + 1];
        match a[i].as_str() {
            "--steps" => c.steps = v.parse().unwrap(), "--emb" => c.emb = v.parse().unwrap(),
            "--layer" => c.layer = v.parse().unwrap(), "--head" => c.head = v.parse().unwrap(),
            "--ctx" => c.ctx = v.parse().unwrap(), "--ff" => c.ff = v.parse().unwrap(),
            "--lr" => c.lr = v.parse().unwrap(), "--probe" => c.probe = v.parse().unwrap(),
            "--seed" => c.seed = v.parse().unwrap(), "--data" => c.input = v.clone(),
            "--stage" => c.stage = v.parse().unwrap(),
            "--inject-layer" => c.inject_layer = v.parse().unwrap(),
            "--sigma-max" => c.sigma_max = v.parse().unwrap(),
            "--rho" => c.rho = v.split(',').map(|x| x.parse().unwrap()).collect(),
            "--report-steps" => c.report_steps = v.parse().unwrap(),
            "--twin" => c.twin = v.clone(),
            "--train-seed" => c.train_seed = Some(v.parse().unwrap()),
            "--base-file" => c.base_file = Some(v.clone()),
            other => panic!("unknown flag {other}"),
        }
        i += 2;
    }
    c
}

// --- a seeded GPT (the library's mat() uses thread_rng, so runs would not repeat) ---
fn seeded_mat(rng: &mut StdRng, r: usize, c: usize, scale: f64) -> Mat2 {
    (0..r).map(|_| (0..c).map(|_| Val::new(rng.gen_range(-1.0..1.0) * scale)).collect()).collect()
}

fn seeded_gpt(seed: u64, v: usize, cfg: &Cfg) -> GPT {
    let mut rng = StdRng::seed_from_u64(seed);
    let (d, l, s) = (cfg.emb, cfg.layer, 0.1);
    let mut layers = |r: usize, c: usize| (0..l).map(|_| seeded_mat(&mut rng, r, c, s)).collect::<Vec<_>>();
    let (wq, wk, wv, wo) = (layers(d, d), layers(d, d), layers(d, d), layers(d, d));
    let (fc1, fc2) = (layers(cfg.ff * d, d), layers(d, cfg.ff * d));
    GPT { wte: seeded_mat(&mut rng, v, d, s), wpe: seeded_mat(&mut rng, cfg.ctx, d, s),
          lm_head: seeded_mat(&mut rng, v, d, s), wq, wk, wv, wo, fc1, fc2,
          n_head: cfg.head, rms_eps: 1e-5 }
}

// The library's forward, but also returning the final normed hidden state: the
// "internals" a self-report may read. Kept line-for-line with GPT::forward.
fn forward_h(m: &GPT, t: usize, pos: usize, k: &mut [Vec<Vec1>], v: &mut [Vec<Vec1>]) -> (Vec1, Vec<f64>) {
    forward_noisy(m, t, pos, k, v, None)
}

// Stage 3's perturbation: after layer `layer`, add Gaussian noise of size
// sigma * rms(x) to every residual-stream component (fresh noise per position).
pub struct Noise<'a> { layer: usize, sigma: f64, rng: &'a mut StdRng }

fn gauss(rng: &mut StdRng) -> f64 {
    let (u1, u2): (f64, f64) = (rng.gen::<f64>().max(1e-12), rng.gen());
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn forward_noisy(m: &GPT, t: usize, pos: usize, k: &mut [Vec<Vec1>], v: &mut [Vec<Vec1>],
                 mut noise: Option<&mut Noise>) -> (Vec1, Vec<f64>) {
    let mut x: Vec1 = m.wte[t].iter().zip(&m.wpe[pos]).map(|(t, p)| t + p).collect();
    let hd = x.len() / m.n_head;
    for i in 0..m.wq.len() {
        let xn = rmsnorm(&x, m.rms_eps);
        let q = linear(&xn, &m.wq[i]);
        k[i].push(linear(&xn, &m.wk[i]));
        v[i].push(linear(&xn, &m.wv[i]));
        let mut att = vec![];
        for h in 0..m.n_head {
            let r = h * hd..(h + 1) * hd;
            let scale = Val::new((hd as f64).sqrt().recip());
            let scores: Vec1 = k[i].iter().map(|kt| q[r.clone()].iter().zip(&kt[r.clone()])
                .map(|(a, b)| a * b).fold(Val::new(0.), |a, b| a + b) * &scale).collect();
            let w = softmax(&scores);
            let mut out = (0..hd).map(|_| Val::new(0.)).collect::<Vec1>();
            for (t, wt) in w.iter().enumerate() {
                for (j, val) in v[i][t][r.clone()].iter().enumerate() { out[j] = &out[j] + &(wt * val); }
            }
            att.extend(out);
        }
        x = x.iter().zip(linear(&att, &m.wo[i])).map(|(x, a)| x + a).collect();
        let xn = rmsnorm(&x, m.rms_eps);
        let h = linear(&xn, &m.fc1[i]).iter().map(|v| v.relu()).collect::<Vec1>();
        x = x.iter().zip(linear(&h, &m.fc2[i])).map(|(x, m)| x + m).collect();
        if let Some(n) = noise.as_deref_mut() {
            if n.layer == i && n.sigma > 0.0 {
                let rms = (x.iter().map(|v| v.data().powi(2)).sum::<f64>() / x.len() as f64).sqrt();
                x = x.iter().map(|v| v + &Val::new(n.sigma * rms * gauss(&mut *n.rng))).collect();
            }
        }
    }
    let hn = rmsnorm(&x, m.rms_eps);
    let hidden = hn.iter().map(|v| v.data()).collect();
    (linear(&hn, &m.lm_head), hidden)
}

fn tokens(name: &str, chars: &[char], ctx: usize) -> Vec<usize> {
    let end = chars.len();
    let mut t: Vec<usize> = std::iter::once(end)
        .chain(name.chars().map(|c| chars.iter().position(|&x| x == c).unwrap()))
        .chain(std::iter::once(end)).collect();
    t.truncate(ctx);
    t
}

fn train(m: &GPT, names: &[&str], chars: &[char], cfg: &Cfg, seed: u64, tag: &str) -> f64 {
    let params = m.params();
    let (mut mo, mut ve) = (vec![0.; params.len()], vec![0.; params.len()]);
    let (b1, b2, eps) = (0.85, 0.99, 1e-8);
    let mut order: Vec<usize> = (0..names.len()).collect();
    order.shuffle(&mut StdRng::seed_from_u64(seed ^ 0x5eed));
    let (mut ema, t0) = (0.0, Instant::now());
    for step in 0..cfg.steps {
        let toks = tokens(names[order[step % order.len()]], chars, cfg.ctx);
        let (mut kc, mut vc) = (vec![vec![]; cfg.layer], vec![vec![]; cfg.layer]);
        let mut loss = Val::new(0.);
        for p in 0..toks.len() - 1 {
            let (logits, _) = forward_h(m, toks[p], p, &mut kc, &mut vc);
            loss = &loss - &softmax(&logits)[toks[p + 1]].log();
        }
        loss = &loss * &Val::new(((toks.len() - 1) as f64).recip());
        ema = if step == 0 { loss.data() } else { 0.99 * ema + 0.01 * loss.data() };
        for p in &params { p.zero(); }
        loss.backward();
        let lr = cfg.lr * (1. - step as f64 / cfg.steps as f64);
        for (i, p) in params.iter().enumerate() {
            let g = p.grad();
            mo[i] = b1 * mo[i] + (1. - b1) * g;
            ve[i] = b2 * ve[i] + (1. - b2) * g * g;
            let (mh, vh) = (mo[i] / (1. - b1.powi(step as i32 + 1)), ve[i] / (1. - b2.powi(step as i32 + 1)));
            p.0.borrow_mut().data -= lr * mh / (vh.sqrt() + eps);
        }
        if step % 500 == 0 || step + 1 == cfg.steps {
            eprintln!("  [{tag}] step {step:5}  loss(ema) {ema:.4}  {:.0}s", t0.elapsed().as_secs_f64());
        }
    }
    ema
}

// Per name: the model's own mean NLL, and features from its hidden states
// (mean over positions, plus the state after reading the end token).
fn probe(m: &GPT, name: &str, chars: &[char], cfg: &Cfg) -> (f64, Vec<f64>) {
    let toks = tokens(name, chars, cfg.ctx);
    let (mut kc, mut vc) = (vec![vec![]; cfg.layer], vec![vec![]; cfg.layer]);
    let (mut nll, mut sum, mut last) = (0.0, vec![0.0; cfg.emb], vec![0.0; cfg.emb]);
    for p in 0..toks.len() {
        let (logits, h) = forward_h(m, toks[p], p, &mut kc, &mut vc);
        if p + 1 < toks.len() {
            nll -= softmax(&logits)[toks[p + 1]].data().ln();
        }
        for j in 0..cfg.emb { sum[j] += h[j]; }
        last = h;
    }
    let n = toks.len() as f64;
    let mut f: Vec<f64> = sum.iter().map(|s| s / n).collect();
    f.extend(last);
    (nll / (toks.len() - 1) as f64, f)
}

// --- ridge regression (closed form; feature dims here are <= 33) ---
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i][c].abs().partial_cmp(&a[j][c].abs()).unwrap()).unwrap();
        a.swap(c, p); b.swap(c, p);
        for r in c + 1..n {
            let f = a[r][c] / a[c][c];
            for k in c..n { a[r][k] -= f * a[c][k]; }
            b[r] -= f * b[c];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        x[r] = (b[r] - (r + 1..n).map(|k| a[r][k] * x[k]).sum::<f64>()) / a[r][r];
    }
    x
}

fn standardize(train: &[Vec<f64>], test: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let d = train[0].len();
    let mu: Vec<f64> = (0..d).map(|j| train.iter().map(|r| r[j]).sum::<f64>() / train.len() as f64).collect();
    let sd: Vec<f64> = (0..d).map(|j| (train.iter().map(|r| (r[j] - mu[j]).powi(2)).sum::<f64>()
        / train.len() as f64).sqrt().max(1e-9)).collect();
    let z = |rows: &[Vec<f64>]| rows.iter().map(|r| {
        let mut v: Vec<f64> = r.iter().enumerate().map(|(j, x)| (x - mu[j]) / sd[j]).collect();
        v.push(1.0);
        v
    }).collect::<Vec<_>>();
    (z(train), z(test))
}

fn ridge_fit(x: &[Vec<f64>], y: &[f64], lambda: f64) -> Vec<f64> {
    let d = x[0].len();
    let mut a = vec![vec![0.0; d]; d];
    let mut b = vec![0.0; d];
    for (r, t) in x.iter().zip(y) {
        for i in 0..d { b[i] += r[i] * t; for j in 0..d { a[i][j] += r[i] * r[j]; } }
    }
    for i in 0..d - 1 { a[i][i] += lambda; }        // no penalty on the bias
    solve(a, b)
}

fn predict(x: &[Vec<f64>], w: &[f64]) -> Vec<f64> {
    x.iter().map(|r| r.iter().zip(w).map(|(a, b)| a * b).sum()).collect()
}

fn r2(y: &[f64], p: &[f64]) -> f64 {
    let mu = y.iter().sum::<f64>() / y.len() as f64;
    let ss_tot: f64 = y.iter().map(|t| (t - mu).powi(2)).sum();
    1.0 - y.iter().zip(p).map(|(t, q)| (t - q).powi(2)).sum::<f64>() / ss_tot
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f64 = a.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum();
    cov / (va * vb).sqrt()
}

// Fit on `fit`, choosing lambda on an inner 80/20 split; return test predictions.
fn readout(fx: &[Vec<f64>], fy: &[f64], tx: &[Vec<f64>]) -> Vec<f64> {
    let (zf, zt) = standardize(fx, tx);
    let cut = fy.len() * 4 / 5;
    let best = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0].iter().copied().max_by(|&a, &b| {
        let s = |l: f64| r2(&fy[cut..], &predict(&zf[cut..], &ridge_fit(&zf[..cut], &fy[..cut], l)));
        s(a).partial_cmp(&s(b)).unwrap()
    }).unwrap();
    predict(&zt, &ridge_fit(&zf, fy, best))
}

fn bootstrap_margin(y: &[f64], p_self: &[f64], p_cross: &[f64], seed: u64) -> (f64, f64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = y.len();
    let mut d: Vec<f64> = (0..1000).map(|_| {
        let idx: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
        let pick = |v: &[f64]| idx.iter().map(|&i| v[i]).collect::<Vec<_>>();
        let yy = pick(y);
        r2(&yy, &pick(p_self)) - r2(&yy, &pick(p_cross))
    }).collect();
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (d[25], d[974])
}

// Stage 3 probe: loss, hidden-state features (as in stage 1), and outputs-only
// features: statistics of the predicted distributions that need no knowledge of
// the true next character (mean / sd / last entropy, mean top probability, length).
fn probe3(m: &GPT, name: &str, chars: &[char], cfg: &Cfg, mut noise: Option<&mut Noise>)
          -> (f64, Vec<f64>, Vec<f64>) {
    let toks = tokens(name, chars, cfg.ctx);
    let (mut kc, mut vc) = (vec![vec![]; cfg.layer], vec![vec![]; cfg.layer]);
    let (mut nll, mut sum, mut last) = (0.0, vec![0.0; cfg.emb], vec![0.0; cfg.emb]);
    let (mut ents, mut tops) = (vec![], vec![]);
    for p in 0..toks.len() {
        let (logits, h) = forward_noisy(m, toks[p], p, &mut kc, &mut vc, noise.as_deref_mut());
        if p + 1 < toks.len() {
            let q: Vec<f64> = softmax(&logits).iter().map(|v| v.data()).collect();
            nll -= q[toks[p + 1]].ln();
            ents.push(-q.iter().filter(|&&x| x > 0.0).map(|x| x * x.ln()).sum::<f64>());
            tops.push(q.iter().cloned().fold(0.0, f64::max));
        }
        for j in 0..cfg.emb { sum[j] += h[j]; }
        last = h;
    }
    let n = toks.len() as f64;
    let mut f: Vec<f64> = sum.iter().map(|s| s / n).collect();
    f.extend(last);
    let k = ents.len() as f64;
    let me = ents.iter().sum::<f64>() / k;
    let sd = (ents.iter().map(|e| (e - me).powi(2)).sum::<f64>() / k).sqrt();
    let out = vec![me, sd, *ents.last().unwrap(), tops.iter().sum::<f64>() / k, k];
    (nll / k, f, out)
}

fn stage3(a: &GPT, b: &GPT, probe_names: &[&str], chars: &[char], cfg: &Cfg,
          text_bag: &[Vec<f64>]) -> serde_json::Value {
    let mut srng = StdRng::seed_from_u64(cfg.seed ^ 0x51);
    let sigma: Vec<f64> = probe_names.iter().map(|_| srng.gen_range(0.0..cfg.sigma_max)).collect();
    let t0 = Instant::now();
    let (mut l0, mut ls, mut hs, mut os, mut hb) = (vec![], vec![], vec![], vec![], vec![]);
    for (i, n) in probe_names.iter().enumerate() {
        l0.push(probe3(a, n, chars, cfg, None).0);
        let mut nrng = StdRng::seed_from_u64(cfg.seed * 1_000_003 + i as u64);
        let mut noise = Noise { layer: cfg.inject_layer, sigma: sigma[i], rng: &mut nrng };
        let (l, h, o) = probe3(a, n, chars, cfg, Some(&mut noise));
        ls.push(l); hs.push(h); os.push(o);
        hb.push(probe3(b, n, chars, cfg, None).1);
    }
    eprintln!("  stage 3: probed {} names x 3 passes in {:.0}s", probe_names.len(),
              t0.elapsed().as_secs_f64());
    let dl: Vec<f64> = ls.iter().zip(&l0).map(|(s, c)| s - c).collect();
    println!("\nstage 3: noise after layer {} with sigma ~ U(0, {}) x rms(residual), fresh per position",
             cfg.inject_layer, cfg.sigma_max);
    println!("clean loss {:.4}  perturbed loss {:.4}  mean change {:+.4}  r(sigma, change) = {:.3}",
             l0.iter().sum::<f64>() / l0.len() as f64, ls.iter().sum::<f64>() / ls.len() as f64,
             dl.iter().sum::<f64>() / dl.len() as f64, pearson(&sigma, &dl));
    let both: Vec<Vec<f64>> = hs.iter().zip(&os).map(|(h, o)| [h.clone(), o.clone()].concat()).collect();
    let h = cfg.probe / 2;
    let split = |v: &[Vec<f64>]| (v[..h].to_vec(), v[h..].to_vec());
    let mut out = serde_json::Map::new();
    println!("\n{:<20} {:<36} {:>7}  {}", "target", "predictor", "R2", "95% CI");
    for (tname, y) in [("perturbed loss", &ls), ("change from noise", &dl), ("noise level sigma", &sigma)] {
        let (fy, ty) = (&y[..h], &y[h..]);
        let mut rows = serde_json::Map::new();
        for (label, feats) in [("self: own perturbed hidden states", &hs),
                               ("outputs only (no targets)", &os),
                               ("self + outputs", &both),
                               ("cross: sibling's clean states", &hb),
                               ("text: bag of characters", &text_bag.to_vec())] {
            let (fx, tx) = split(feats);
            let p = readout(&fx, fy, &tx);
            let (rr, (lo, hi)) = (r2(ty, &p), bootstrap_r2(ty, &p, cfg.seed));
            println!("{:<20} {:<36} {:>7.3}  [{:+.3}, {:+.3}]", tname, label, rr, lo, hi);
            rows.insert(label.into(), serde_json::json!({"r2": rr, "ci95": [lo, hi]}));
        }
        // the key contrast: own internal state vs own outputs, on the same names
        let (fx, tx) = split(&hs);
        let p_self = readout(&fx, fy, &tx);
        let (fx, tx) = split(&os);
        let p_out = readout(&fx, fy, &tx);
        let m = r2(ty, &p_self) - r2(ty, &p_out);
        let (lo, hi) = bootstrap_margin(ty, &p_self, &p_out, cfg.seed);
        println!("{:<20} self - outputs-only margin: {:+.3}  95% CI [{:+.3}, {:+.3}]\n", tname, m, lo, hi);
        rows.insert("self_minus_outputs".into(), serde_json::json!({"r2": m, "ci95": [lo, hi]}));
        out.insert(tname.into(), serde_json::Value::Object(rows));
    }
    serde_json::json!({"inject_layer": cfg.inject_layer, "sigma_max": cfg.sigma_max,
                       "clean_loss": l0.iter().sum::<f64>() / l0.len() as f64,
                       "perturbed_loss": ls.iter().sum::<f64>() / ls.len() as f64,
                       "r_sigma_change": pearson(&sigma, &dl), "results": out})
}

fn bootstrap_r2(y: &[f64], p: &[f64], seed: u64) -> (f64, f64) {
    let mut rng = StdRng::seed_from_u64(seed ^ 0xb007);
    let n = y.len();
    let mut s: Vec<f64> = (0..1000).map(|_| {
        let idx: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
        let yy: Vec<f64> = idx.iter().map(|&i| y[i]).collect();
        let pp: Vec<f64> = idx.iter().map(|&i| p[i]).collect();
        r2(&yy, &pp)
    }).collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (s[25], s[974])
}

/*
    Stage 4: the model reports, and its training history decides what about.

    One base model is pretrained on names, then deep-copied into twins A and B
    (identical weights). Each twin learns to emit a report token R0..R4 after a
    name. The report position carries a public cue token C0..C4 in the input (a
    "public accompaniment", like a visible injury), while noise of level
    sigma ~ U(0, sigma_max) is injected inside, after layer `inject_layer` (the
    "channel", as in stage 3). In training, the cue equals sigma's bin (of 5) with
    probability rho and is uniform otherwise. The twins see the same names, noise
    and cues step for step; the only difference is the gate, what the report is
    scored against:

      A (own channel)  target = sigma's bin
      B (situation)    target = the cue

    LM loss on the name is kept, so both stay name models. At test, on unseen
    names, sigma's bin s and the cue c are drawn independently (decoupled), and
    each twin's expected report E = sum_k k p(R_k) is regressed on s and c. A
    slope of 1 is perfect tracking. Because sigma acts only on internal states,
    any slope on s is the report reading the system's own condition.

    Predictions, fixed before the full run (commit before results):
      P1 rho = 0: A tracks s (slope > 0, CI above 0) and not c (slope ~ 0);
         B tracks c (slope near 1) and not s.
      P2 the gate alone does not fix what the report tracks: as rho rises, A's
         slope on c rises and its slope on s falls, and at rho = 0.9 A's slope
         on c exceeds its slope on s. Same nominal objective, different anchor.
      P3 rho = 1: A's and B's targets coincide, so the twins are the same network
         and give identical reports (a check on the construction). Their gate
         histories differ only counterfactually, which no behavioural test shows.
    Run plan: seeds 7, 8, 9; --steps 10000 (base) --report-steps 10000 --probe 2000;
    rho 0, 0.5, 0.9, 1; a prediction holds if it holds for all three seeds.
    Result (7ee6995): P1-P3 held for all three seeds.

    Crossover sweep (fixed after 7ee6995, before any sweep run). Twin A only,
    rho = 0.50, 0.55, ..., 0.90, seeds 7, 8, 9, same steps; one job per (seed, rho)
    with --twin a. rho* is where A's slope on c minus its slope on s changes sign,
    by linear interpolation between adjacent grid points.
      P4 matching: A switches to the cue when the cue predicts its target better
         than its own-state reading does. The cue's accuracy is rho + (1 - rho)/5;
         A's own-state accuracy is its rho = 0 accuracy (0.668, 0.679, 0.681 for
         seeds 7, 8, 9), so predicted rho* = (acc - 0.2) / 0.8 = 0.585, 0.599,
         0.601. P4 holds if every seed's rho* is within 0.075 of its prediction.
      Check: the sweep runs on a different machine (x86) from stage 4 (arm64), so
         rho 0.5 and 0.9 are reruns; their slopes should fall inside the stage-4
         runs' 95% intervals.
    Sweep result (results/stage4_sweep, scripts/stage4_crossover.py): P4 failed,
    rho* = 0.62, 0.73, 0.81 against 0.59-0.60 predicted; the check failed at rho 0.9
    for seeds 7 and 9 (the intervals cover test names, not training runs).

    Band sweep (fixed after 0199c57, before any band run). Twin A, on cyber2 (x86),
    rho = 0.60, 0.65, 0.70, 0.75, 0.80; bases 7, 8, 9 (pretrained once each, reused
    with --base-file); training seeds 101-105 (--train-seed), plus the sweep's own
    run (training seed = base seed) as a sixth run per cell. Same steps and test
    set. gap = slope on c - slope on s; a run is cue-anchored if gap > 0.
    Scored by scripts/stage4_band.py.
      P5 run dependence: for at least one rho, every base's six runs include both
         a self-anchored and a cue-anchored run.
      P6 run variation dominates test sampling: the mean over (base, rho) cells of
         the SD of gap across the six runs is at least 0.10 (the bootstrap
         half-widths of the slopes are about 0.01-0.02).
      P7 rho still matters: pooled over bases and runs, the fraction of
         cue-anchored runs at rho 0.80 exceeds that at rho 0.60 by at least 0.30.
    Band result (results/stage4_band): P5-P7 held. Every base is mixed at rho 0.70;
    the within-cell SD of gap is 0.217 against bootstrap half-widths of 0.024;
    the cue-anchored fraction rises from 0.22 at 0.60 to 0.89 at 0.80.
    Not pre-registered (scripts/stage4_loss_anchor.py, after the band result): at
    every rho the self-anchored runs end with lower training report loss than the
    cue-anchored ones (0.61-0.64 against 0.78-0.92 at rho 0.60-0.70; Spearman of gap
    with loss +0.76 to +0.82). Cue-anchored runs sit just under the cue-only loss
    (1.07, 0.98, 0.88), so they barely use their own state. The objective prefers
    the own-state anchor across the band, yet most runs at rho >= 0.70 take the cue.
*/
fn clone_gpt(m: &GPT) -> GPT {
    let c = |x: &Mat2| x.iter().map(|r| r.iter().map(|v| Val::new(v.data())).collect()).collect::<Mat2>();
    let cs = |xs: &Vec<Mat2>| xs.iter().map(c).collect::<Vec<_>>();
    GPT { wte: c(&m.wte), wpe: c(&m.wpe), lm_head: c(&m.lm_head), wq: cs(&m.wq), wk: cs(&m.wk),
          wv: cs(&m.wv), wo: cs(&m.wo), fc1: cs(&m.fc1), fc2: cs(&m.fc2),
          n_head: m.n_head, rms_eps: m.rms_eps }
}

// Token ids beyond the base vocabulary: cues C0..C4, then reports R0..R4.
struct Extra { cue0: usize, rep0: usize }
const BINS: usize = 5;

// One decoupling-or-not draw: noise level, its bin, and the cue.
fn draw4(rng: &mut StdRng, rho: f64, sigma_max: f64) -> (f64, usize, usize) {
    let u: f64 = rng.gen();
    let s = ((u * BINS as f64) as usize).min(BINS - 1);
    let c = if rng.gen::<f64>() < rho { s } else { rng.gen_range(0..BINS) };
    (u * sigma_max, s, c)
}

// Name, then the cue; returns the mean LM loss on the name and the five report
// logits read at the cue position.
fn run4(m: &GPT, name: &str, chars: &[char], cfg: &Cfg, ex: &Extra, cue: usize,
        noise: &mut Noise) -> (Val, Vec1) {
    let mut toks = tokens(name, chars, cfg.ctx - 1);
    toks.push(ex.cue0 + cue);
    let (mut kc, mut vc) = (vec![vec![]; cfg.layer], vec![vec![]; cfg.layer]);
    let mut loss = Val::new(0.);
    let mut rep = vec![];
    for p in 0..toks.len() {
        let (logits, _) = forward_noisy(m, toks[p], p, &mut kc, &mut vc, Some(&mut *noise));
        if p + 2 < toks.len() {
            loss = &loss - &softmax(&logits)[toks[p + 1]].log();
        } else if p + 1 == toks.len() {
            rep = logits[ex.rep0..ex.rep0 + BINS].to_vec();
        }
    }
    (&loss * &Val::new(((toks.len() - 2) as f64).recip()), rep)
}

fn train_report(m: &GPT, names: &[&str], chars: &[char], cfg: &Cfg, ex: &Extra, rho: f64,
                own_gate: bool, tag: &str) -> (f64, f64) {
    let params = m.params();
    let (mut mo, mut ve) = (vec![0.; params.len()], vec![0.; params.len()]);
    let (b1, b2, eps) = (0.85, 0.99, 1e-8);
    // --train-seed varies the report-training run (name order, noise, cues) while the
    // base model and the test set stay fixed by --seed
    let ts = cfg.train_seed.unwrap_or(cfg.seed);
    let mut order: Vec<usize> = (0..names.len()).collect();
    order.shuffle(&mut StdRng::seed_from_u64(ts ^ 0x4ead));
    let (mut ema_lm, mut ema_rep, t0) = (0.0, 0.0, Instant::now());
    for step in 0..cfg.report_steps {
        // the same draws for both twins: only the target differs
        let mut rng = StdRng::seed_from_u64(ts * 7_000_003 + step as u64);
        let (sigma, s, c) = draw4(&mut rng, rho, cfg.sigma_max);
        let mut noise = Noise { layer: cfg.inject_layer, sigma, rng: &mut rng };
        let (lm, rep) = run4(m, names[order[step % order.len()]], chars, cfg, ex, c, &mut noise);
        let target = if own_gate { s } else { c };
        let rl = -softmax(&rep)[target].log();
        let loss = &lm + &rl;
        let a = if step == 0 { 1.0 } else { 0.01 };
        ema_lm = (1. - a) * ema_lm + a * lm.data();
        ema_rep = (1. - a) * ema_rep + a * rl.data();
        for p in &params { p.zero(); }
        loss.backward();
        let lr = cfg.lr * (1. - step as f64 / cfg.report_steps as f64);
        for (i, p) in params.iter().enumerate() {
            let g = p.grad();
            mo[i] = b1 * mo[i] + (1. - b1) * g;
            ve[i] = b2 * ve[i] + (1. - b2) * g * g;
            let (mh, vh) = (mo[i] / (1. - b1.powi(step as i32 + 1)), ve[i] / (1. - b2.powi(step as i32 + 1)));
            p.0.borrow_mut().data -= lr * mh / (vh.sqrt() + eps);
        }
        if step % 1000 == 0 || step + 1 == cfg.report_steps {
            eprintln!("  [{tag}] step {step:5}  lm(ema) {ema_lm:.4}  report CE(ema) {ema_rep:.4}  {:.0}s",
                      t0.elapsed().as_secs_f64());
        }
    }
    (ema_lm, ema_rep)
}

// OLS of y on [s, c, 1]; returns (slope on s, slope on c).
fn slopes(s: &[f64], c: &[f64], y: &[f64]) -> (f64, f64) {
    let x: Vec<Vec<f64>> = s.iter().zip(c).map(|(a, b)| vec![*a, *b, 1.0]).collect();
    let w = ridge_fit(&x, y, 0.0);
    (w[0], w[1])
}

fn ci(mut v: Vec<f64>) -> [f64; 2] {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    [v[25], v[974]]
}

fn stage4(base: &GPT, train_names: &[&str], probe_names: &[&str], chars: &[char], cfg: &Cfg,
          ex: &Extra) -> serde_json::Value {
    let mut runs = vec![];
    for &rho in &cfg.rho {
        // --twin a / b / ab: which twins to train (one job per twin for parallel runs)
        let twins: Vec<_> = [("A (own channel)", true), ("B (situation)", false)].into_iter()
            .filter(|(t, _)| cfg.twin.contains(&t[..1].to_lowercase()))
            .map(|(t, own)| (t, clone_gpt(base), own)).collect();
        let mut rows = serde_json::Map::new();
        let mut reports: Vec<Vec<f64>> = vec![];
        println!("\nstage 4, rho = {rho}: cue = sigma's bin with prob rho in training; decoupled at test");
        println!("{:<18} {:>8} {:>18} {:>8} {:>18} {:>7} {:>7} {:>7}", "twin", "slope s", "95% CI",
                 "slope c", "95% CI", "acc s", "acc c", "LM");
        for (tag, m, own) in &twins {
            let (lm_ema, rep_ema) = train_report(m, train_names, chars, cfg, ex, rho, *own,
                                                 &format!("rho {rho} {}", &tag[..1]));
            let (mut s, mut c, mut e, mut hit_s, mut hit_c, mut lm) = (vec![], vec![], vec![], 0, 0, 0.0);
            for (i, n) in probe_names.iter().enumerate() {
                let mut rng = StdRng::seed_from_u64(cfg.seed * 9_000_011 + i as u64);
                let (sigma, si, ci_) = draw4(&mut rng, 0.0, cfg.sigma_max);
                let mut noise = Noise { layer: cfg.inject_layer, sigma, rng: &mut rng };
                let (l, rep) = run4(m, n, chars, cfg, ex, ci_, &mut noise);
                let p: Vec<f64> = softmax(&rep).iter().map(|v| v.data()).collect();
                let k = (0..BINS).max_by(|&a, &b| p[a].partial_cmp(&p[b]).unwrap()).unwrap();
                hit_s += (k == si) as usize;
                hit_c += (k == ci_) as usize;
                s.push(si as f64); c.push(ci_ as f64);
                e.push(p.iter().enumerate().map(|(k, q)| k as f64 * q).sum());
                lm += l.data();
            }
            let n = probe_names.len();
            let (bs, bc) = slopes(&s, &c, &e);
            let mut brng = StdRng::seed_from_u64(cfg.seed ^ 0xb004);
            let (vs, vc): (Vec<f64>, Vec<f64>) = (0..1000).map(|_| {
                let idx: Vec<usize> = (0..n).map(|_| brng.gen_range(0..n)).collect();
                let pick = |v: &[f64]| idx.iter().map(|&i| v[i]).collect::<Vec<_>>();
                slopes(&pick(&s), &pick(&c), &pick(&e))
            }).unzip();
            let (cs, cc) = (ci(vs), ci(vc));
            let (acc_s, acc_c) = (hit_s as f64 / n as f64, hit_c as f64 / n as f64);
            println!("{:<18} {:>8.3} [{:+.3}, {:+.3}] {:>8.3} [{:+.3}, {:+.3}] {:>7.3} {:>7.3} {:>7.4}",
                     tag, bs, cs[0], cs[1], bc, cc[0], cc[1], acc_s, acc_c, lm / n as f64);
            rows.insert(tag.to_string(), serde_json::json!({
                "slope_sigma_bin": bs, "slope_sigma_ci95": cs, "slope_cue": bc, "slope_cue_ci95": cc,
                "acc_vs_sigma_bin": acc_s, "acc_vs_cue": acc_c, "probe_lm_loss_noisy": lm / n as f64,
                "train_lm_ema": lm_ema, "train_report_ce_ema": rep_ema}));
            reports.push(e);
        }
        let diff = (reports.len() == 2).then(|| reports[0].iter().zip(&reports[1])
            .map(|(a, b)| (a - b).abs()).fold(0.0, f64::max));
        if let Some(d) = diff { println!("max |E_A - E_B| over probe names: {d:.2e}"); }
        runs.push(serde_json::json!({"rho": rho, "twins": rows, "max_abs_report_diff": diff}));
    }
    serde_json::json!({"inject_layer": cfg.inject_layer, "sigma_max": cfg.sigma_max, "bins": BINS,
                       "report_steps": cfg.report_steps, "runs": runs})
}

fn main() {
    let cfg = parse_args();
    let raw = load_training_data(&cfg.input);
    let chars = build_vocab(&raw);
    let vocab = chars.len() + 1;
    let mut names: Vec<&str> = raw.lines().filter(|l| !l.trim().is_empty()).collect();
    names.shuffle(&mut StdRng::seed_from_u64(cfg.seed));
    let (probe_names, train_names) = names.split_at(cfg.probe);
    println!("names: {} train, {} probe (unseen by both models); emb {} layer {} head {} ctx {}; {} steps",
             train_names.len(), probe_names.len(), cfg.emb, cfg.layer, cfg.head, cfg.ctx, cfg.steps);

    if cfg.stage == 4 {
        let ex = Extra { cue0: vocab, rep0: vocab + BINS };
        let base = seeded_gpt(cfg.seed * 2 + 1, vocab + 2 * BINS, &cfg);
        println!("params per model: {}", base.params().len());
        // --base-file: reuse a pretrained base (exact f64 round trip) or save this one
        let saved = cfg.base_file.as_ref().filter(|p| std::path::Path::new(p).exists());
        let ema = if let Some(p) = saved {
            let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
            assert_eq!(v["seed"].as_u64(), Some(cfg.seed), "{p} is another seed's base");
            for (q, x) in base.params().iter().zip(v["params"].as_array().unwrap()) {
                q.0.borrow_mut().data = x.as_f64().unwrap();
            }
            eprintln!("  [base] loaded {p}");
            v["ema"].as_f64().unwrap()
        } else {
            let ema = train(&base, train_names, &chars, &cfg, 1, "base");
            if let Some(p) = &cfg.base_file {
                let params: Vec<f64> = base.params().iter().map(|q| q.data()).collect();
                std::fs::write(p, serde_json::json!({"seed": cfg.seed, "steps": cfg.steps,
                                                     "ema": ema, "params": params}).to_string()).unwrap();
            }
            ema
        };
        let res = stage4(&base, train_names, probe_names, &chars, &cfg, &ex);
        let out = serde_json::json!({
            "config": {"stage": 4, "steps": cfg.steps, "emb": cfg.emb, "layer": cfg.layer,
                       "head": cfg.head, "ctx": cfg.ctx, "probe": cfg.probe, "seed": cfg.seed,
                       "rho": cfg.rho, "report_steps": cfg.report_steps, "twin": cfg.twin,
                       "train_seed": cfg.train_seed},
            "base_train_loss_ema": ema, "stage4": res});
        let mut tag = if cfg.twin == "ab" { String::new() } else {
            format!("r{}_{}_", cfg.rho.iter().map(|r| r.to_string()).collect::<Vec<_>>().join("-"), cfg.twin) };
        if let Some(t) = cfg.train_seed { tag += &format!("t{t}_"); }
        let path = format!("results/self_twins_s{}_{tag}{}.json", cfg.seed,
                           chrono::Local::now().format("%Y%m%d_%H%M%S"));
        std::fs::write(&path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
        println!("wrote {path}");
        return;
    }

    let (a, b) = (seeded_gpt(cfg.seed * 2 + 1, vocab, &cfg), seeded_gpt(cfg.seed * 2 + 2, vocab, &cfg));
    println!("params per model: {}", a.params().len());
    let ema_a = train(&a, train_names, &chars, &cfg, 1, "A");
    let ema_b = train(&b, train_names, &chars, &cfg, 2, "B");

    if cfg.stage == 3 {
        let bag: Vec<Vec<f64>> = probe_names.iter().map(|n| {
            let mut v = vec![0.0; chars.len()];
            for c in n.chars() { v[chars.iter().position(|&x| x == c).unwrap()] += 1.0; }
            v }).collect();
        let res = stage3(&a, &b, probe_names, &chars, &cfg, &bag);
        let out = serde_json::json!({
            "config": {"stage": 3, "steps": cfg.steps, "emb": cfg.emb, "layer": cfg.layer,
                       "head": cfg.head, "ctx": cfg.ctx, "probe": cfg.probe, "seed": cfg.seed},
            "train_loss_ema": {"A": ema_a, "B": ema_b}, "stage3": res});
        let path = format!("results/self_perturb_{}.json", chrono::Local::now().format("%Y%m%d_%H%M%S"));
        std::fs::write(&path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
        println!("wrote {path}");
        return;
    }

    let t0 = Instant::now();
    let (pa, pb): (Vec<_>, Vec<_>) = probe_names.iter()
        .map(|n| (probe(&a, n, &chars, &cfg), probe(&b, n, &chars, &cfg))).unzip();
    eprintln!("  probed {} names in {:.0}s", probe_names.len(), t0.elapsed().as_secs_f64());
    let (loss_a, feat_a): (Vec<f64>, Vec<Vec<f64>>) = pa.into_iter().unzip();
    let (loss_b, feat_b): (Vec<f64>, Vec<Vec<f64>>) = pb.into_iter().unzip();

    let text_len: Vec<Vec<f64>> = probe_names.iter().map(|n| {
        let l = n.chars().count() as f64; vec![l, l * l] }).collect();
    let text_bag: Vec<Vec<f64>> = probe_names.iter().map(|n| {
        let mut v = vec![0.0; chars.len()];
        for c in n.chars() { v[chars.iter().position(|&x| x == c).unwrap()] += 1.0; }
        v }).collect();

    let h = cfg.probe / 2;
    let split = |v: &Vec<Vec<f64>>| (v[..h].to_vec(), v[h..].to_vec());
    let mut report = serde_json::Map::new();
    println!("\nloss on unseen names: A mean {:.4}, B mean {:.4}; A~B per-name r = {:.3}",
             loss_a.iter().sum::<f64>() / loss_a.len() as f64,
             loss_b.iter().sum::<f64>() / loss_b.len() as f64, pearson(&loss_a, &loss_b));
    println!("\n{:<10} {:<34} {:>7} {:>7}", "target", "predictor", "R2", "r");
    for (tname, y, own, other) in [("A's loss", &loss_a, &feat_a, &feat_b),
                                    ("B's loss", &loss_b, &feat_b, &feat_a)] {
        let (fy, ty) = (&y[..h], &y[h..]);
        let preds = [("self (own hidden states)", own), ("cross (sibling's hidden states)", other),
                     ("length baseline", &text_len), ("bag-of-characters baseline", &text_bag)]
            .map(|(label, feats)| {
                let (fx, tx) = split(feats);
                (label, readout(&fx, fy, &tx))
            });
        let mut rows = serde_json::Map::new();
        for (label, p) in &preds {
            let (rr, pr) = (r2(ty, p), pearson(ty, p));
            println!("{:<10} {:<34} {:>7.3} {:>7.3}", tname, label, rr, pr);
            rows.insert(label.to_string(), serde_json::json!({"r2": rr, "pearson": pr}));
        }
        let margin = r2(ty, &preds[0].1) - r2(ty, &preds[1].1);
        let (lo, hi) = bootstrap_margin(ty, &preds[0].1, &preds[1].1, cfg.seed);
        println!("{:<10} privileged-access margin (self - cross R2): {:+.3}  95% CI [{:+.3}, {:+.3}]\n",
                 tname, margin, lo, hi);
        rows.insert("margin".into(), serde_json::json!({"self_minus_cross_r2": margin, "ci95": [lo, hi]}));
        report.insert(tname.into(), serde_json::Value::Object(rows));
    }
    // Residual test: d = A's loss - B's loss cancels the difficulty both models share
    // (mostly the text), leaving each model's idiosyncratic part. A's states can only
    // see A's part of d, B's states only B's; text should see ~nothing.
    let d: Vec<f64> = loss_a.iter().zip(&loss_b).map(|(a, b)| a - b).collect();
    let var = |v: &[f64]| { let m = v.iter().sum::<f64>() / v.len() as f64;
                            v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64 };
    let both: Vec<Vec<f64>> = feat_a.iter().zip(&feat_b).map(|(a, b)| [a.clone(), b.clone()].concat()).collect();
    println!("residual target d = A's loss - B's loss: var(d)/var(A's loss) = {:.3}",
             var(&d) / var(&loss_a));
    let (fy, ty) = (&d[..h], &d[h..]);
    let mut res = serde_json::Map::new();
    for (label, feats) in [("A's hidden states", &feat_a), ("B's hidden states", &feat_b),
                           ("both models' states", &both), ("bag-of-characters (text control)", &text_bag)] {
        let (fx, tx) = split(feats);
        let p = readout(&fx, fy, &tx);
        let (rr, (lo, hi)) = (r2(ty, &p), bootstrap_r2(ty, &p, cfg.seed));
        println!("{:<10} {:<34} {:>7.3}  95% CI [{:+.3}, {:+.3}]", "A - B", label, rr, lo, hi);
        res.insert(label.into(), serde_json::json!({"r2": rr, "pearson": pearson(ty, &p), "ci95": [lo, hi]}));
    }
    let out = serde_json::json!({
        "config": {"steps": cfg.steps, "emb": cfg.emb, "layer": cfg.layer, "head": cfg.head,
                   "ctx": cfg.ctx, "ff": cfg.ff, "lr": cfg.lr, "probe": cfg.probe, "seed": cfg.seed},
        "train_loss_ema": {"A": ema_a, "B": ema_b},
        "loss_correlation_A_B": pearson(&loss_a, &loss_b),
        "results": report,
        "residual_A_minus_B": {"var_ratio": var(&d) / var(&loss_a), "predictors": res},
    });
    let path = format!("results/self_report_{}.json", chrono::Local::now().format("%Y%m%d_%H%M%S"));
    std::fs::write(&path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("wrote {path}");
}
