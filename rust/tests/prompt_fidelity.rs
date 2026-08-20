#[path = "../crates/nv-models/tests/prompt_scan/mod.rs"]
mod prompt_scan;

const TESTS_DIR_RS_FILES_WALKED_TODAY: usize = 61;

const WHY_THE_FLOOR_IS_PINNED_TO_THE_EXACT_WALK: &str =
    "This floor stood at 12 while the walk was recorded at 24 and had since grown to 34, so \
     two thirds of rust/tests could have vanished and the scan would still have reported clean. \
     A floor set to a fraction of the walk re-accumulates that slack with every test file that \
     lands, which is exactly how a floor of 12 came to sit under a walk of 34. Pinning it to the \
     exact count is the only version that cannot drift in either direction: deleting a harness \
     trips the scanner's own floor, and adding one trips this equality and has to be \
     acknowledged by bumping the number. Deriving the floor from the walk would make the bound a \
     restatement of the value it bounds, and such a bound survives every mutation.";

fn scan() -> prompt_scan::CrateScan<'static> {
    prompt_scan::CrateScan {
        crate_name: "serve",
        manifest_dir: env!("CARGO_MANIFEST_DIR"),
        self_file: file!(),
        min_walked: TESTS_DIR_RS_FILES_WALKED_TODAY,
        allowed: &[],
    }
}

#[test]
fn no_real_weights_harness_hand_builds_a_gemma4_prompt() {
    scan().run();
}

#[test]
fn the_floor_is_todays_exact_walk_and_not_a_fraction_of_it() {
    let dir = scan().tests_dir();
    let me = std::path::Path::new(file!())
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    let found = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "rs"))
        .filter(|p| p.file_name().unwrap_or_default().to_string_lossy() != me)
        .count();
    eprintln!("[serve] prompt fidelity floor: {found} scannable .rs files in {}, floor pinned at {TESTS_DIR_RS_FILES_WALKED_TODAY}", dir.display());
    assert_eq!(
        found, TESTS_DIR_RS_FILES_WALKED_TODAY,
        "{WHY_THE_FLOOR_IS_PINNED_TO_THE_EXACT_WALK}\n\n{} now holds {found} scannable .rs \
         files against a pinned floor of {TESTS_DIR_RS_FILES_WALKED_TODAY}. If a harness \
         landed, set TESTS_DIR_RS_FILES_WALKED_TODAY to {found}. If one was culled, say so \
         deliberately and set it to {found}; do not leave the floor above or below the walk.",
        dir.display()
    );
}
