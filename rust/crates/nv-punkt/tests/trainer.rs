
use nv_punkt::{PunktParameters, PunktTrainer, Segmenter};

fn abbreviation_corpus() -> String {
    let mut s = String::new();
    for i in 0..40 {
        s.push_str("we met kpt. anders at the dock today and talked for a while. ");
        if i % 3 == 0 {
            s.push_str("the weather was fine. ");
        }
    }
    s
}

#[test]
fn training_teaches_an_abbreviation_and_the_segmenter_stops_splitting_on_it() {
    let mut tr = PunktTrainer::new();
    tr.train(&abbreviation_corpus());
    let params = tr.into_params();
    assert!(
        params.abbrev_types.contains("kpt"),
        "40 mid-sentence occurrences of kpt. taught nothing"
    );

    let trained = Segmenter::new(params);
    let text = "we met kpt. anders again.";
    assert_eq!(
        trained.sentence_strings(text).len(),
        1,
        "the learned abbreviation is not honoured at segmentation time"
    );

    let bare = Segmenter::new(PunktParameters::default());
    assert_eq!(
        bare.sentence_strings(text).len(),
        2,
        "the negative control stopped splitting, so the row above proves nothing"
    );
}

#[test]
fn type_counts_fold_case_and_numbers_the_way_the_scores_assume() {
    let mut tr = PunktTrainer::new();
    tr.train("Word word WORD 3.5 7,000 -2 hello");
    assert_eq!(tr.type_count("word"), 3, "types must be case-folded before counting");
    assert_eq!(
        tr.type_count("##number##"),
        3,
        "all numeric shapes must fold into the number type"
    );
    assert_eq!(tr.type_count("hello"), 1);
    assert_eq!(tr.type_count("Word"), 0, "raw-case lookups must miss");
}

#[test]
fn finalize_is_idempotent_so_incremental_training_cannot_double_count() {
    let mut tr = PunktTrainer::new();
    tr.train(&abbreviation_corpus());
    tr.finalize();
    let starters_once = tr.params().sent_starters.clone();
    let collocs_once = tr.params().collocations.clone();
    tr.finalize();
    assert_eq!(tr.params().sent_starters, starters_once);
    assert_eq!(tr.params().collocations, collocs_once);
}

#[test]
fn an_empty_corpus_trains_nothing_and_panics_nowhere() {

    let mut tr = PunktTrainer::new();
    tr.train("");
    let params = tr.into_params();
    assert!(params.sent_starters.is_empty());
    assert!(params.collocations.is_empty());
}
