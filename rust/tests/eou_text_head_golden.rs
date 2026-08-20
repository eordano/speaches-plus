use speaches_plus::eou::text_head::TextHead;

fn committed_head() -> TextHead {
    TextHead::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/eou_text_head_fnv_v1.bin"
    ))
    .expect("the committed v1 text head must load")
}

#[test]
fn committed_head_reproduces_the_training_time_goldens() {
    let head = committed_head();
    for (text, want) in [
        ("yes.", 0.998432f32),
        ("I think we should probably", 0.001379),
        ("let me check the account number", 0.005086),
        ("That's all I needed, thanks.", 0.040209),
    ] {
        let got = head.prob(text);
        assert!(
            (got - want).abs() < 1e-3,
            "rust FNV features or dot product diverged from the python trainer for {text:?}: \
             got {got}, trained {want}"
        );
    }
}

#[test]
fn committed_head_orders_obvious_cases_correctly() {
    let head = committed_head();
    let done = head.prob("yes.");
    let midword = head.prob("and then we could, um");
    let midclause = head.prob("I think we should probably");
    assert!(
        done > midword && done > midclause,
        "a bare terminal 'yes.' must outrank mid-utterance fragments: done={done} \
         midword={midword} midclause={midclause}"
    );
}
