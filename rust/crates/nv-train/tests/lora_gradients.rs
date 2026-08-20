
use candle_core::{DType, Device, Tensor, Var};
use nv_train::{lora_delta, LoraConfig, LoraTrainable};

const R: usize = 4;
const IN: usize = 6;
const OUT: usize = 5;
const N: usize = 3;

fn seeded(shape: (usize, usize), seed: u64) -> Tensor {
    let n = shape.0 * shape.1;
    let mut s = seed;
    let data: Vec<f32> = (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect();
    Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
}

fn loss_of(a: &Tensor, b: &Tensor, x: &Tensor, scaling: f64) -> f64 {
    let a_h: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
    let b_h: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
    let x_h: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
    let (r, inf) = (a.dims()[0], a.dims()[1]);
    let (outf, n) = (b.dims()[0], x.dims()[0]);
    let mut total = 0f64;
    for row in 0..n {

        let mut xr = vec![0f64; r];
        for (k, xr_k) in xr.iter_mut().enumerate() {
            for i in 0..inf {
                *xr_k += x_h[row * inf + i] as f64 * a_h[k * inf + i] as f64;
            }
        }

        for o in 0..outf {
            let mut acc = 0f64;
            for (k, xr_k) in xr.iter().enumerate() {
                acc += xr_k * b_h[o * r + k] as f64;
            }
            total += acc * scaling;
        }
    }
    total
}

fn perturbed(t: &Tensor, idx: usize, eps: f64) -> Tensor {
    let dims = t.dims().to_vec();
    let mut v: Vec<f32> = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    v[idx] += eps as f32;
    Tensor::from_vec(v, (dims[0], dims[1]), t.device()).unwrap()
}

#[test]
fn the_lora_gradients_match_a_central_difference_of_the_loss() {

    let scaling = LoraConfig { r: R, alpha: 8.0, dropout: 0.0 }.scaling();
    let a = Var::from_tensor(&seeded((R, IN), 11)).unwrap();
    let b = Var::from_tensor(&seeded((OUT, R), 22)).unwrap();
    let x = seeded((N, IN), 33);

    let out = lora_delta(a.as_tensor(), b.as_tensor(), scaling, &x, None).unwrap();
    let grads = out.sum_all().unwrap().backward().unwrap();

    let da = grads.get(&a).expect("A received no gradient at all").clone();
    let db = grads.get(&b).expect("B received no gradient at all").clone();

    let eps = 1e-2f64;
    for (label, var, analytic, idx) in [
        ("A", &a, &da, 0usize),
        ("A", &a, &da, R * IN - 1),
        ("B", &b, &db, 0usize),
        ("B", &b, &db, OUT * R - 1),
    ] {
        let base = var.as_tensor();
        let (hi, lo) = if label == "A" {
            (
                loss_of(&perturbed(base, idx, eps), b.as_tensor(), &x, scaling),
                loss_of(&perturbed(base, idx, -eps), b.as_tensor(), &x, scaling),
            )
        } else {
            (
                loss_of(a.as_tensor(), &perturbed(base, idx, eps), &x, scaling),
                loss_of(a.as_tensor(), &perturbed(base, idx, -eps), &x, scaling),
            )
        };
        let numeric = (hi - lo) / (2.0 * eps);
        let got = analytic.flatten_all().unwrap().to_vec1::<f32>().unwrap()[idx] as f64;

        let tol = 1e-2 * numeric.abs().max(1.0);
        assert!(
            (numeric - got).abs() < tol,
            "d(loss)/d{label}[{idx}]: candle says {got}, an independent f64 central \
             difference says {numeric}"
        );
    }
}

#[test]
fn a_fresh_adapter_trains_b_first_and_only_then_a() {

    let base = Tensor::zeros((OUT, IN), DType::F32, &Device::Cpu).unwrap();
    let lora = LoraTrainable::new(&base, LoraConfig { r: R, alpha: 8.0, dropout: 0.0 }, 7, &Device::Cpu)
        .unwrap();
    let x = seeded((N, IN), 44);
    let vars = lora.trainable_vars();
    assert_eq!(vars.len(), 2, "A and B are the trainable pair");

    let grads = lora_delta(lora.a_tensor(), lora.b_tensor(), lora.scaling(), &x, None)
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();
    for (i, v) in vars.iter().enumerate() {
        let g = grads.get(v).unwrap_or_else(|| {
            panic!(
                "trainable var {i} is absent from the GradStore -- it is a graph leaf, so \
                 training moves it nowhere"
            )
        });
        let host: Vec<f32> = g.flatten_all().unwrap().to_vec1().unwrap();
        assert!(host.iter().all(|x| x.is_finite()), "var {i} has a non-finite gradient");
    }
    let db: Vec<f32> = grads.get(&vars[1]).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        db.iter().any(|v| *v != 0.0),
        "B must move on the first step, or the adapter can never leave the identity"
    );

    vars[0].set(&seeded((R, IN), 123)).unwrap();
    vars[1].set(&seeded((OUT, R), 321)).unwrap();
    let grads = lora_delta(lora.a_tensor(), lora.b_tensor(), lora.scaling(), &x, None)
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();
    for (i, v) in vars.iter().enumerate() {
        let host: Vec<f32> = grads
            .get(v)
            .unwrap_or_else(|| panic!("var {i} absent once B is nonzero"))
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(
            host.iter().any(|x| *x != 0.0),
            "var {i} still has an all-zero gradient once B is off zero, so the optimiser \
             is a no-op on it"
        );
    }
}

#[test]
fn a_zeroed_b_makes_a_receive_nothing_which_is_this_gate_s_negative_control() {

    let a = Var::from_tensor(&seeded((R, IN), 55)).unwrap();
    let b = Var::from_tensor(&Tensor::zeros((OUT, R), DType::F32, &Device::Cpu).unwrap()).unwrap();
    let x = seeded((N, IN), 66);

    let out = lora_delta(a.as_tensor(), b.as_tensor(), 1.0, &x, None).unwrap();
    let grads = out.sum_all().unwrap().backward().unwrap();

    let da: Vec<f32> = grads.get(&a).expect("A present").flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        da.iter().all(|v| *v == 0.0),
        "with B zeroed the A gradient must vanish; if it does not, the chain rule is \
         not being applied through B and the previous row proves nothing"
    );
    let db: Vec<f32> = grads.get(&b).expect("B present").flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        db.iter().any(|v| *v != 0.0),
        "B must still receive a gradient at B == 0, or LoRA could never leave its \
         zero initialisation"
    );
}

#[test]
fn a_detached_factor_silently_receives_no_gradient_at_all() {

    let a = Var::from_tensor(&seeded((R, IN), 77)).unwrap();
    let b = Var::from_tensor(&seeded((OUT, R), 88)).unwrap();
    let x = seeded((N, IN), 99);

    let detached_a = a.as_tensor().detach();
    let out = lora_delta(&detached_a, b.as_tensor(), 1.0, &x, None).unwrap();
    let grads = out.sum_all().unwrap().backward().unwrap();

    assert!(
        grads.get(&a).is_none(),
        "a detached factor must not appear in the GradStore; if candle ever starts \
         tracking through detach, the eleven BackpropOp::none sites in nv-layers stop \
         being silent and this gate should be revisited"
    );
    assert!(grads.get(&b).is_some(), "the undetached factor still trains");
}
