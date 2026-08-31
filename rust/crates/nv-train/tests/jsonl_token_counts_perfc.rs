use nv_train::load_jsonl_token_dataset;
use nv_train::qwen38::Q38Config;
use std::path::PathBuf;

#[test]
#[ignore = "prints per-row token counts for a real dataset; set NV_Q38_TOKEN_COUNTS=1"]
fn per_row_token_counts_for_the_jsonl_datasets_the_trainer_will_step_over() {
    if std::env::var("NV_Q38_TOKEN_COUNTS").as_deref() != Ok("1") {
        eprintln!("skip: NV_Q38_TOKEN_COUNTS unset");
        return;
    }
    let base = PathBuf::from(std::env::var("NV_Q38_AB_BASE").expect("NV_Q38_AB_BASE"));
    let cfg = Q38Config::from_hf_json_file(&base.join("config.json")).unwrap();
    let max_seq = 512usize;
    for var in ["NV_Q38_AB_DATA", "NV_Q38_AB_EVAL_DATA"] {
        let path = PathBuf::from(std::env::var(var).expect(var));
        let rows = load_jsonl_token_dataset(&path, &base, cfg.vocab_size, max_seq).unwrap();
        let per_row: Vec<usize> = rows.iter().map(|r| r.ids.len() - 1).collect();
        let total: usize = per_row.iter().sum();
        println!(
            "[jsonl-token-counts] {var} {} rows {} trained_positions_per_step {total} per_row {:?}",
            path.display(),
            rows.len(),
            per_row
        );
    }
}
