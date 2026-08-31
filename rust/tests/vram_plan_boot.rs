#![cfg(feature = "cuda")]

use std::path::PathBuf;

use speaches_plus::oapi::chat_engine::NvEngineChat;
use speaches_plus::oapi::vram_plan;

const BOOT_TEST_ENV: &str = "NV_VRAM_PLAN_BOOT_TEST";

fn gated_model_dir() -> Option<PathBuf> {
    if std::env::var(BOOT_TEST_ENV).as_deref() != Ok("1") {
        eprintln!("SKIP: set {BOOT_TEST_ENV}=1 plus NV_CHAT_MODEL_DIR to run this real-boot gate");
        return None;
    }
    let dir = PathBuf::from(
        std::env::var_os("NV_CHAT_MODEL_DIR")
            .expect("NV_VRAM_PLAN_BOOT_TEST=1 requires NV_CHAT_MODEL_DIR"),
    );
    assert!(dir.is_dir(), "NV_CHAT_MODEL_DIR={} is not a dir", dir.display());
    Some(dir)
}

#[test]
#[ignore]
fn over_budget_boot_refuses_with_the_itemized_plan_before_any_cuda_allocation() {
    let Some(dir) = gated_model_dir() else { return };
    std::env::set_var("NV_VRAM_BUDGET_GIB", "2.0");
    let err = NvEngineChat::try_load(&dir)
        .err()
        .expect("a 2 GiB budget cannot hold any served checkpoint; try_load must refuse");
    std::env::remove_var("NV_VRAM_BUDGET_GIB");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("vram plan exceeds"),
        "refusal must come from the boot plan, got: {msg}"
    );
    assert!(
        msg.contains("weights-resident"),
        "refusal must carry the itemized plan, got: {msg}"
    );
    assert!(
        msg.contains("smallest changes that would fit, ranked:"),
        "refusal must carry the ranked smallest-change list, got: {msg}"
    );
    assert!(
        vram_plan::process_vram_bytes_via_nvidia_smi().unwrap_or(0) == 0,
        "the refusal happened before device init, so this process must hold no VRAM"
    );
}

#[test]
#[ignore]
fn composed_process_plan_prints_and_verdicts_without_touching_the_gpu() {
    if std::env::var("NV_VRAM_PLAN_COMPOSE_TEST").as_deref() != Ok("1") {
        eprintln!(
            "SKIP: set NV_VRAM_PLAN_COMPOSE_TEST=1 plus NV_CHAT_MODEL_DIRS=<q38>,<gemma4> to \
             dry-run the process-cumulative plan from safetensors headers alone"
        );
        return;
    }
    let dirs: Vec<PathBuf> = std::env::var("NV_CHAT_MODEL_DIRS")
        .expect("NV_VRAM_PLAN_COMPOSE_TEST=1 requires NV_CHAT_MODEL_DIRS")
        .split(',')
        .map(PathBuf::from)
        .collect();
    let budget = vram_plan::env_budget_bytes();
    let mut refused = Vec::new();
    let mut records_held_live_like_the_engines_they_stand_for = Vec::new();
    for dir in &dirs {
        let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
        let boot = if raw.contains("\"gemma4\"") || raw.contains("gemma4_text") {
            vram_plan::plan_gemma4(dir, &raw, budget)
        } else {
            vram_plan::plan_qwen35(dir, &raw, true, false, budget)
        }
        .expect("plan");
        println!("{}", boot.plan.render());
        if !boot.plan.fits() {
            println!(
                "smallest changes that would fit, ranked:\n{}",
                boot.plan.ranked_changes_rendered()
            );
            refused.push(dir.display().to_string());
        }
        records_held_live_like_the_engines_they_stand_for
            .push(vram_plan::record_boot_plan(&boot.plan));
    }
    println!(
        "process planned total: {:.2} GiB vs budget {:.2} GiB; refused: {refused:?}",
        vram_plan::recorded_planned_total_bytes() as f64 / vram_plan::GIB,
        budget as f64 / vram_plan::GIB
    );
    assert!(
        std::env::var("NV_VRAM_PLAN_COMPOSE_EXPECT_REFUSAL").as_deref() != Ok("1")
            || !refused.is_empty(),
        "NV_VRAM_PLAN_COMPOSE_EXPECT_REFUSAL=1 but every plan fit"
    );
}

#[test]
#[ignore]
fn boot_complete_drop_boot_at_the_same_budget_admits_the_second_engine_on_the_true_ledger() {
    let Some(dir) = gated_model_dir() else { return };
    let first = NvEngineChat::try_load(&dir).expect("first boot under the budget");
    assert!(
        vram_plan::recorded_planned_total_bytes() > 0,
        "a live engine must stay counted"
    );
    drop(first);
    assert_eq!(
        vram_plan::recorded_planned_total_bytes(),
        0,
        "a dropped engine must leave the ledger empty; its phantom bytes are why the c=16 \
         concurrency harness needed NV_VRAM_BUDGET_GIB=128 for a true peak that fits 86"
    );
    let second = NvEngineChat::try_load(&dir)
        .expect("the second sequential boot must be admitted at the budget the first fit");
    drop(second);
}

#[test]
#[ignore]
fn real_boot_lands_within_tolerance_of_the_printed_plan() {
    let Some(dir) = gated_model_dir() else { return };
    let engine = NvEngineChat::try_load(&dir).expect("boot under the default budget");
    let planned = vram_plan::recorded_planned_total_bytes();
    assert!(planned > 0, "try_load must record a boot plan");
    let measured = vram_plan::process_vram_bytes_via_nvidia_smi()
        .expect("nvidia-smi must report this process after a real boot");
    let tol = (vram_plan::env_gib(
        vram_plan::VRAM_PLAN_TOLERANCE_ENV,
        vram_plan::RECONCILE_TOLERANCE_GIB_DEFAULT_COVERS_ALLOCATOR_ROUNDING_AND_GRAPH_POOLS,
    ) * vram_plan::GIB) as u64;
    eprintln!(
        "plan-vs-measured: planned {:.2} GiB, measured {:.2} GiB, tolerance {:.2} GiB",
        planned as f64 / vram_plan::GIB,
        measured as f64 / vram_plan::GIB,
        tol as f64 / vram_plan::GIB
    );
    assert!(
        measured <= planned + tol,
        "measured {:.2} GiB exceeds planned {:.2} GiB + tolerance {:.2} GiB; a component \
         allocated outside its declared plan entry",
        measured as f64 / vram_plan::GIB,
        planned as f64 / vram_plan::GIB,
        tol as f64 / vram_plan::GIB
    );
    assert!(
        measured + 40 * (1 << 30) > planned,
        "planned {:.2} GiB is wildly above measured {:.2} GiB; the plan formulas overstate \
         by more than the whole-card margin and would refuse boots that fit",
        planned as f64 / vram_plan::GIB,
        measured as f64 / vram_plan::GIB
    );
    drop(engine);
}
