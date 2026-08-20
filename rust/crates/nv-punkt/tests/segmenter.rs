
use nv_punkt::{tokenize, PunktParameters, Segmenter, Token};

fn english() -> Segmenter {
    Segmenter::english()
}

#[test]
fn every_token_span_slices_back_to_its_own_text_even_through_multibyte_lines() {

    let text = "café costs 3.5 dollars…\n\n\u{201c}Résumé\u{201d} whoa.\nJ. Müller wrote it.";
    for t in tokenize(text) {
        assert_eq!(
            &text[t.span.clone()],
            t.text,
            "token text and its span disagree at {:?}",
            t.span
        );
    }
}

#[test]
fn sentence_ranges_are_ordered_disjoint_in_bounds_and_on_char_boundaries() {
    let text = "He said \u{201c}Stop.\u{201d} Then he left… quickly. Done. café über 3.5 %.";
    let s = english();
    let ranges = s.sentences(text);
    assert!(!ranges.is_empty());
    let mut prev_end = 0usize;
    for r in &ranges {
        assert!(r.start >= prev_end, "ranges overlap or go backwards at {r:?}");
        assert!(r.end <= text.len(), "range {r:?} exceeds the text");
        assert!(text.is_char_boundary(r.start) && text.is_char_boundary(r.end),
            "range {r:?} splits a multibyte char; slicing here panics in the TTS path");
        prev_end = r.end;
    }
    let strings = s.sentence_strings(text);
    assert_eq!(strings.len(), ranges.len());
    for (r, sent) in ranges.iter().zip(&strings) {
        assert_eq!(&text[r.clone()], *sent);
    }
}

#[test]
fn the_three_regex_splitter_traps_stay_whole() {

    let s = english();
    for (text, expect) in [
        ("Dr. Smith arrived early.", 1usize),
        ("It costs 3.5 dollars in total.", 1),
        ("J. Smith wrote the paper.", 1),
        ("One ends here. Two starts now.", 2),
    ] {
        let got = s.sentence_strings(text);
        assert_eq!(
            got.len(),
            expect,
            "{text:?} split as {got:?} -- a naive splitter's answer, not punkt's"
        );
    }
}

#[test]
fn an_ascii_closing_quote_after_the_period_belongs_to_the_finished_sentence() {

    let text = "He said \"Stop.\" Then he left.";
    let strings = english().sentence_strings(text);
    assert_eq!(strings.len(), 2, "got {strings:?}");
    assert!(
        strings[0].ends_with('"'),
        "the closer stayed on the next sentence: {:?}",
        strings
    );
    assert!(strings[1].starts_with("Then"));
}

#[test]
fn a_curly_quoted_sentence_end_now_splits_where_upstream_nltk_still_misses_it() {
    let text = "He said \u{201c}Stop.\u{201d} Then he left.";
    let strings = english().sentence_strings(text);
    assert_eq!(strings.len(), 2, "curly close-quote boundary lost again: {strings:?}");
    assert!(
        strings[0].ends_with('\u{201d}'),
        "realign must attach the curly closer to the finished sentence: {strings:?}"
    );
    assert!(strings[1].starts_with("Then"));
}

#[test]
fn empty_and_whitespace_texts_produce_no_sentences_and_no_panic() {
    let s = english();
    assert!(s.sentences("").is_empty());
    assert!(s.sentences("  \n\n\t ").is_empty());
}

#[test]
fn token_classifiers_agree_with_their_definitions() {
    let t = |s: &str| Token::new(s, 0..s.len(), false, false);
    assert!(t("J.").is_initial());
    assert!(!t("Jr.").is_initial(), "two letters before the period is not an initial");
    assert!(!t("4.").is_initial(), "a digit is not an initial");
    assert!(t("...").is_ellipsis());
    assert!(t(". . .").is_ellipsis());
    assert!(t("\u{2026}").is_ellipsis());
    assert!(!t(".").is_ellipsis(), "a single period is a break, not an ellipsis");
    assert!(t("3.5").is_number());
    assert!(t("-3,500.25").is_number());
    assert!(!t("3.5x").is_number(), "a trailing letter makes it a word");
    assert_eq!(t("Dr.").type_no_period(), "dr");
    assert_eq!(t(".").type_no_period(), ".", "a bare period must not strip to empty");
}

#[test]
fn an_untrained_segmenter_splits_after_dr_which_is_what_the_params_rows_prove_matters() {

    let bare = Segmenter::new(PunktParameters::default());
    let got = bare.sentence_strings("Dr. Smith arrived early.");
    assert_eq!(
        got.len(),
        2,
        "an empty abbreviation list still kept Dr. whole: {got:?}"
    );
}
