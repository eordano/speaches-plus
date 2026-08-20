use std::path::{Path, PathBuf};

pub const FP8_E4M3_MAX: f32 = 448.0;
pub const INT8_MAX: f32 = 127.0;
const MIN_CU_SOURCES_SCANNED: usize = 30;

fn cuda_src_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda")
}

fn skip_subpath(p: &Path) -> bool {
    p.components()
        .any(|c| c.as_os_str() == "marlin" || c.as_os_str() == "cutlass")
}

fn collect_cu(dir: &Path, out: &mut Vec<PathBuf>) {
    let rd = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return,
    };
    for e in rd.flatten() {
        let p = e.path();
        if skip_subpath(&p) {
            continue;
        }
        if p.is_dir() {
            collect_cu(&p, out);
        } else {
            let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("");
            if ext == "cu" || ext == "cuh" {
                out.push(p);
            }
        }
    }
}

#[derive(Debug)]
struct RoundTripHit {
    file: String,
    scale_line: usize,
    inv_line: usize,
    var: String,
    peak: String,
    scale_text: String,
    inv_text: String,
}

fn ident_before(s: &str, eq: usize) -> Option<String> {
    let lhs = s[..eq].trim_end();
    let tok = lhs.rsplit(|c: char| c.is_whitespace() || c == '*').next()?;
    let tok = tok.trim();
    if tok.is_empty() || !tok.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return None;
    }
    if tok.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(tok.to_string())
}

fn reciprocal_of(line: &str, var: &str) -> bool {
    let forms = [
        format!("1.0f / {var}"),
        format!("1.0f/{var}"),
        format!("1.f / {var}"),
        format!("1.f/{var}"),
        format!("1.0 / {var}"),
        format!("1.0/{var}"),
        format!("__frcp_rn({var})"),
        format!("__fdividef(1.0f, {var})"),
    ];
    forms.iter().any(|f| {
        line.match_indices(f.as_str()).any(|(i, _)| {
            let after = i + f.len();
            line[after..]
                .chars()
                .next()
                .is_none_or(|c| !(c.is_alphanumeric() || c == '_'))
        })
    })
}

fn scan_round_trip_inverses() -> Vec<RoundTripHit> {
    let mut files = Vec::new();
    collect_cu(&cuda_src_root(), &mut files);
    files.sort();
    let mut hits = Vec::new();
    for f in files {
        let text = match std::fs::read_to_string(&f) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let lines: Vec<&str> = text.lines().collect();
        let rel = f
            .strip_prefix(env!("CARGO_MANIFEST_DIR"))
            .unwrap_or(&f)
            .display()
            .to_string();
        for (i, line) in lines.iter().enumerate() {
            for peak in [FP8_E4M3_MAX, INT8_MAX] {
                let peak_lit = format!("/ {peak:.1}f");
                let peak_lit_tight = format!("/{peak:.1}f");
                if !line.contains(&peak_lit) && !line.contains(&peak_lit_tight) {
                    continue;
                }
                let eq = match line.find('=') {
                    Some(e) => e,
                    None => continue,
                };
                let var = match ident_before(line, eq) {
                    Some(v) => v,
                    None => continue,
                };
                let end = (i + 12).min(lines.len());
                for (j, cand) in lines.iter().enumerate().take(end).skip(i + 1) {
                    if reciprocal_of(cand, &var) {
                        hits.push(RoundTripHit {
                            file: rel.clone(),
                            scale_line: i + 1,
                            inv_line: j + 1,
                            var: var.clone(),
                            peak: format!("{peak:.1}"),
                            scale_text: line.trim().to_string(),
                            inv_text: cand.trim().to_string(),
                        });
                        break;
                    }
                }
            }
        }
    }
    hits
}

#[test]
fn cuda_quantizers_must_not_compute_inverse_scale_by_round_trip() {
    let hits = scan_round_trip_inverses();
    let mut scanned = Vec::new();
    collect_cu(&cuda_src_root(), &mut scanned);
    eprintln!(
        "scanned {} .cu/.cuh sources under {}",
        scanned.len(),
        cuda_src_root().display()
    );
    assert!(
        scanned.len() >= MIN_CU_SOURCES_SCANNED,
        "the scan walked {} .cu/.cuh sources under {}, below the {MIN_CU_SOURCES_SCANNED} this \
         gate was calibrated on. `collect_cu` swallows a read_dir error and returns an empty \
         list, so a moved or renamed cuda/ directory makes an empty-hit-set look exactly like a \
         clean tree. An empty walk is a finding, not a pass.",
        scanned.len(),
        cuda_src_root().display()
    );
    for h in &hits {
        eprintln!(
            "ROUND-TRIP INVERSE: {}:{} `{}` then {}:{} `{}` (var `{}`, peak {})",
            h.file, h.scale_line, h.scale_text, h.file, h.inv_line, h.inv_text, h.var, h.peak
        );
    }
    eprintln!(
        "CONTRACT (fp8_contract_e4m3.rs::inverse_scale_must_be_448_over_amax_not_one_over_scale): \
         an fp8 quantizer must compute inv as 448/amax in ONE division. `scale = amax/448; \
         inv = 1/scale` is two roundings and is not byte-equivalent."
    );
    let (fp8, int8): (Vec<&RoundTripHit>, Vec<&RoundTripHit>) =
        hits.iter().partition(|h| h.peak == "448.0");
    eprintln!(
        "int8 (peak 127.0) round-trip sites: {} - NOT asserted here. CUDA rowquant_i8 and the \
         WGSL rowquant_i8 (wgsl/gemv_bf16.wgsl rq_recip_normal) deliberately share the \
         reciprocal-of-scale form and parity_gemv_bf16_i8.rs pins them byte-exact to each \
         other. Changing one without the other breaks that gate.",
        int8.len()
    );
    assert!(
        fp8.is_empty(),
        "{} CUDA fp8 quantizer(s) derive the inverse scale by reciprocating a rounded scale: {:#?}",
        fp8.len(),
        fp8
    );
}

#[test]
fn the_round_trip_scan_detects_the_pattern_it_is_meant_to_catch() {
    let sample = "    float scale = red[0] / 448.0f;\n    float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;\n";
    let lines: Vec<&str> = sample.lines().collect();
    let eq = lines[0].find('=').unwrap();
    let var = ident_before(lines[0], eq).expect("lhs identifier");
    assert_eq!(var, "scale");
    assert!(lines[0].contains("/ 448.0f"));
    assert!(reciprocal_of(lines[1], &var));
    assert!(!reciprocal_of("float inv = 448.0f / amax;", "amax"));
    assert!(!reciprocal_of("float x = 1.0f / scaleb;", "scale"));
}

#[cfg(feature = "cuda")]
mod device {
    use super::FP8_E4M3_MAX;
    use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
    use half::bf16;
    use nv_kernels::cuda;
    use std::ffi::c_void;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn require() -> bool {
        std::env::var("NV_KERNELS_PARITY_REQUIRE").as_deref() == Ok("1")
    }

    fn stream(test: &str) -> Option<Arc<CudaStream>> {
        match CudaContext::new(0) {
            Ok(c) => Some(c.default_stream()),
            Err(e) => {
                if require() {
                    panic!("{test}: no CUDA device 0: {e}");
                }
                eprintln!("{test}: SKIP no CUDA device 0: {e}");
                None
            }
        }
    }

    fn e4m3_magnitudes() -> &'static [f64; 127] {
        static T: std::sync::OnceLock<[f64; 127]> = std::sync::OnceLock::new();
        T.get_or_init(|| {
            let mut t = [0f64; 127];
            for (mag, slot) in t.iter_mut().enumerate() {
                let e = (mag >> 3) as i32;
                let m = (mag & 7) as f64;
                *slot = if e == 0 {
                    m * 2f64.powi(-9)
                } else {
                    (1.0 + m / 8.0) * 2f64.powi(e - 7)
                };
            }
            t
        })
    }

    fn ref_encode_e4m3(x: f32) -> u8 {
        if x.is_nan() {
            return 0x7f;
        }
        let sign: u8 = if x.is_sign_negative() { 0x80 } else { 0x00 };
        let a = x.abs() as f64;
        if a >= FP8_E4M3_MAX as f64 {
            return sign | 0x7e;
        }
        let t = e4m3_magnitudes();
        let hi = t.partition_point(|v| *v <= a);
        if hi == 0 {
            return sign;
        }
        let lo = hi - 1;
        if hi >= t.len() {
            return sign | lo as u8;
        }
        let dl = a - t[lo];
        let dh = t[hi] - a;
        let pick = if dl < dh {
            lo
        } else if dh < dl {
            hi
        } else if lo % 2 == 0 {
            lo
        } else {
            hi
        };
        sign | pick as u8
    }

    fn finite_amax(row: &[u16]) -> f32 {
        row.iter().fold(0f32, |a, b| {
            let v = bf16::from_bits(*b).to_f32();
            if v.is_finite() {
                a.max(v.abs())
            } else {
                a
            }
        })
    }

    fn run_rowquant(
        stream: &Arc<CudaStream>,
        w: &[u16],
        n: usize,
        k: usize,
    ) -> (Vec<u8>, Vec<f32>) {
        #[allow(deprecated)]
        let dw: CudaSlice<u16> = stream.clone_htod(w).unwrap();
        let mut dq: CudaSlice<u8> = stream.alloc_zeros::<u8>(n * k).unwrap();
        let mut ds: CudaSlice<f32> = stream.alloc_zeros::<f32>(n).unwrap();
        let rc = {
            let (pw, _a) = dw.device_ptr(stream);
            let (pq, _b) = dq.device_ptr_mut(stream);
            let (ps, _c) = ds.device_ptr_mut(stream);
            unsafe {
                cuda::rowquant_e4m3(
                    stream.cu_stream() as *mut c_void,
                    pw as *const u16,
                    pq as *mut u8,
                    ps as *mut f32,
                    n as i32,
                    k as i32,
                )
            }
        };
        assert_eq!(rc, 0, "rowquant_e4m3 rc={rc} (n={n} k={k})");
        stream.synchronize().unwrap();
        #[allow(deprecated)]
        let q = stream.memcpy_dtov(&dq).unwrap();
        #[allow(deprecated)]
        let s = stream.memcpy_dtov(&ds).unwrap();
        (q, s)
    }

    struct Counts {
        direct: usize,
        round_trip: usize,
        total: usize,
        first_direct: Option<(usize, usize, f32, u8, u8)>,
    }

    fn compare(w: &[u16], q: &[u8], n: usize, k: usize) -> Counts {
        let mut c = Counts {
            direct: 0,
            round_trip: 0,
            total: 0,
            first_direct: None,
        };
        for r in 0..n {
            let row = &w[r * k..(r + 1) * k];
            let amax = finite_amax(row);
            let inv_direct = if amax > 0.0 { FP8_E4M3_MAX / amax } else { 0.0 };
            let sc = amax / FP8_E4M3_MAX;
            let inv_rt = if sc > 0.0 { 1.0f32 / sc } else { 0.0 };
            let same_inv = inv_direct.to_bits() == inv_rt.to_bits();
            for i in 0..k {
                let v = bf16::from_bits(row[i]).to_f32();
                if !v.is_finite() {
                    continue;
                }
                let got = q[r * k + i];
                let want_d = ref_encode_e4m3(v * inv_direct);
                c.total += 1;
                if got != want_d {
                    c.direct += 1;
                    if c.first_direct.is_none() {
                        c.first_direct = Some((r, i, v, got, want_d));
                    }
                }
                let want_r = if same_inv {
                    want_d
                } else {
                    ref_encode_e4m3(v * inv_rt)
                };
                if got != want_r {
                    c.round_trip += 1;
                }
            }
        }
        c
    }

    struct Lcg(u64);
    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        fn unit(&mut self) -> f32 {
            (self.next_u32() & 0x7f_ffff) as f32 / 8388608.0
        }
    }

    #[test]
    fn cuda_rowquant_e4m3_encodes_against_448_over_amax() {
        let Some(st) = stream("cuda_rowquant_e4m3_encodes_against_448_over_amax") else {
            return;
        };
        let n = 512usize;
        let k = 256usize;
        let mut rng = Lcg(0x5eed_1234_9abc_def0);
        let mut w = vec![0u16; n * k];
        for r in 0..n {
            let amax = 0.01 + rng.unit() * 8.0;
            for i in 0..k {
                let frac = (i as f32 + 0.5) / k as f32;
                let s = if rng.next_u32() & 1 == 0 { 1.0 } else { -1.0 };
                w[r * k + i] = bf16::from_f32(s * amax * frac).to_bits();
            }
            w[r * k] = bf16::from_f32(amax).to_bits();
        }
        let (q, sc) = run_rowquant(&st, &w, n, k);
        let c = compare(&w, &q, n, k);
        eprintln!(
            "rowquant_e4m3 synthetic n={n} k={k}: vs inv=448/amax -> {}/{} byte mismatches; \
             vs inv=1/(amax/448) -> {}/{} byte mismatches",
            c.direct, c.total, c.round_trip, c.total
        );
        if let Some((r, i, v, got, want)) = c.first_direct {
            eprintln!(
                "  first divergence row {r} col {i} v={v:e} kernel=0x{got:02x} want=0x{want:02x}"
            );
        }
        let mut sdiff = 0usize;
        for (r, s) in sc.iter().enumerate() {
            let amax = finite_amax(&w[r * k..(r + 1) * k]);
            if s.to_bits() != (amax / FP8_E4M3_MAX).to_bits() {
                sdiff += 1;
            }
        }
        eprintln!("  row_scale mismatches vs amax/448: {sdiff}/{n}");
        assert_eq!(sdiff, 0, "row_scale must be amax/448 exactly");
        assert_eq!(
            c.direct, 0,
            "rowquant_e4m3 must encode with inv = 448/amax (one division), not 1/(amax/448)"
        );
    }

    #[test]
    fn cuda_rowquant_e4m3_amax_ignores_non_finite() {
        let Some(st) = stream("cuda_rowquant_e4m3_amax_ignores_non_finite") else {
            return;
        };
        let n = 4usize;
        let k = 64usize;
        let mut w = vec![0u16; n * k];
        for r in 0..n {
            for i in 0..k {
                let v = ((i as f32) - 32.0) * 0.03;
                w[r * k + i] = bf16::from_f32(v).to_bits();
            }
        }
        w[16] = bf16::from_f32(f32::INFINITY).to_bits();
        w[k + 17] = bf16::from_f32(f32::NEG_INFINITY).to_bits();
        w[2 * k + 18] = bf16::from_f32(f32::NAN).to_bits();
        let (q, sc) = run_rowquant(&st, &w, n, k);
        for r in 0..n {
            let amax = finite_amax(&w[r * k..(r + 1) * k]);
            eprintln!(
                "  row {r}: finite amax={amax:e} kernel scale={:e} want={:e}",
                sc[r],
                amax / FP8_E4M3_MAX
            );
        }
        let c = compare(&w, &q, n, k);
        eprintln!(
            "rowquant_e4m3 non-finite rows: vs finite-amax inv=448/amax -> {}/{} byte mismatches",
            c.direct, c.total
        );
        for r in 0..n {
            let amax = finite_amax(&w[r * k..(r + 1) * k]);
            assert_eq!(
                sc[r].to_bits(),
                (amax / FP8_E4M3_MAX).to_bits(),
                "row {r}: amax reduction must skip non-finite values like wgpu row_amax does"
            );
        }
        assert_eq!(c.direct, 0, "finite entries must survive a poisoned row");
    }

    fn laguna_dir() -> Option<PathBuf> {
        if let Ok(d) = std::env::var("NV_LAGUNA_DIR") {
            return Some(PathBuf::from(d));
        }
        let home = std::env::var("HOME").ok()?;
        let p = PathBuf::from(home)
            .join(".cache/huggingface/hub/models--poolside--Laguna-XS-2.1-NVFP4/snapshots/main");
        p.is_dir().then_some(p)
    }

    fn json_after(h: &str, from: usize, key: &str) -> Option<usize> {
        h[from..].find(key).map(|i| from + i + key.len())
    }

    fn parse_two_u64(h: &str, at: usize) -> Option<(u64, u64)> {
        let s = h[at..].trim_start();
        let s = s.strip_prefix('[')?;
        let end = s.find(']')?;
        let mut it = s[..end].split(',');
        let a = it.next()?.trim().parse().ok()?;
        let b = it.next()?.trim().parse().ok()?;
        Some((a, b))
    }

    fn load_bf16_tensor(dir: &PathBuf, name: &str) -> Option<(Vec<u16>, usize, usize)> {
        use std::io::{Read, Seek, SeekFrom};
        let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .collect();
        shards.sort();
        for p in shards {
            let mut f = std::fs::File::open(&p).ok()?;
            let mut len8 = [0u8; 8];
            if f.read_exact(&mut len8).is_err() {
                continue;
            }
            let hlen = u64::from_le_bytes(len8) as usize;
            let mut hbuf = vec![0u8; hlen];
            if f.read_exact(&mut hbuf).is_err() {
                continue;
            }
            let hdr = std::str::from_utf8(&hbuf).ok()?;
            let key = format!("\"{name}\":");
            let Some(at) = hdr.find(&key) else { continue };
            let dt = json_after(hdr, at, "\"dtype\":")?;
            if !hdr[dt..].trim_start().starts_with("\"BF16\"") {
                return None;
            }
            let sh = json_after(hdr, at, "\"shape\":")?;
            let (rows, cols) = parse_two_u64(hdr, sh)?;
            let off = json_after(hdr, at, "\"data_offsets\":")?;
            let (a, b) = parse_two_u64(hdr, off)?;
            f.seek(SeekFrom::Start(8 + hlen as u64 + a)).ok()?;
            let mut raw = vec![0u8; (b - a) as usize];
            f.read_exact(&mut raw).ok()?;
            let mut out = vec![0u16; raw.len() / 2];
            for (i, o) in out.iter_mut().enumerate() {
                *o = u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]);
            }
            return Some((out, rows as usize, cols as usize));
        }
        None
    }

    #[test]
    #[ignore = "loads the real Laguna checkpoint off disk; run explicitly with --ignored"]
    fn cuda_rowquant_e4m3_on_real_laguna_weights() {
        let dir = laguna_dir().expect("Laguna snapshot not found; set NV_LAGUNA_DIR");
        let Some(st) = stream("cuda_rowquant_e4m3_on_real_laguna_weights") else {
            return;
        };
        let names = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.self_attn.g_proj.weight",
            "lm_head.weight",
        ];
        let mut grand_direct = 0usize;
        let mut grand_rt = 0usize;
        let mut grand_total = 0usize;
        for name in names {
            let Some((w, n, k)) = load_bf16_tensor(&dir, name) else {
                eprintln!("{name}: not found as BF16, skipped");
                continue;
            };
            assert_eq!(w.len(), n * k, "{name}: element count");
            let (q, _sc) = run_rowquant(&st, &w, n, k);
            let c = compare(&w, &q, n, k);
            let pct = 100.0 * c.direct as f64 / c.total as f64;
            eprintln!(
                "{name} [{n}x{k}]: kernel vs inv=448/amax -> {}/{} ({pct:.6}%) bytes differ; \
                 kernel vs inv=1/(amax/448) -> {}/{} bytes differ",
                c.direct, c.total, c.round_trip, c.total
            );
            if let Some((r, i, v, got, want)) = c.first_direct {
                eprintln!(
                    "  first divergence row {r} col {i} v={v:e} kernel=0x{got:02x} want=0x{want:02x}"
                );
            }
            grand_direct += c.direct;
            grand_rt += c.round_trip;
            grand_total += c.total;
        }
        eprintln!(
            "TOTAL real Laguna bf16 weights: vs 448/amax {grand_direct}/{grand_total} \
             ({:.6}%) differ; vs 1/(amax/448) {grand_rt}/{grand_total} ({:.6}%) differ",
            100.0 * grand_direct as f64 / grand_total as f64,
            100.0 * grand_rt as f64 / grand_total as f64
        );
        assert!(grand_total > 0, "no real weight tensor was loaded");
        assert_eq!(
            grand_direct, 0,
            "real Laguna weights must quantize with inv = 448/amax"
        );
    }
}
