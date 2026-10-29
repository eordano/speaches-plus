use crate::json::Json;

const SCRIPTS: &[(&str, &[(u32, u32)])] = &[
    ("greek", &[(0x0370, 0x03FF), (0x1F00, 0x1FFF)]),
    (
        "cyrillic",
        &[(0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)],
    ),
    ("hebrew", &[(0x0590, 0x05FF)]),
    (
        "arabic",
        &[
            (0x0600, 0x06FF),
            (0x0750, 0x077F),
            (0x08A0, 0x08FF),
            (0xFB50, 0xFDFF),
            (0xFE70, 0xFEFF),
        ],
    ),
    ("devanagari", &[(0x0900, 0x097F), (0xA8E0, 0xA8FF)]),
    ("bengali", &[(0x0980, 0x09FF)]),
    ("gurmukhi", &[(0x0A00, 0x0A7F)]),
    ("gujarati", &[(0x0A80, 0x0AFF)]),
    ("oriya", &[(0x0B00, 0x0B7F)]),
    ("tamil", &[(0x0B80, 0x0BFF)]),
    ("telugu", &[(0x0C00, 0x0C7F)]),
    ("kannada", &[(0x0C80, 0x0CFF)]),
    ("malayalam", &[(0x0D00, 0x0D7F)]),
    ("sinhala", &[(0x0D80, 0x0DFF)]),
    ("thai", &[(0x0E00, 0x0E7F)]),
    ("lao", &[(0x0E80, 0x0EFF)]),
    ("tibetan", &[(0x0F00, 0x0FFF)]),
    ("myanmar", &[(0x1000, 0x109F)]),
    ("georgian", &[(0x10A0, 0x10FF)]),
    ("ethiopic", &[(0x1200, 0x137F)]),
    ("khmer", &[(0x1780, 0x17FF)]),
    (
        "hangul",
        &[(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)],
    ),
    (
        "kana",
        &[(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)],
    ),
    (
        "han",
        &[(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)],
    ),
];

const STOPWORDS: &[(&str, &[&str])] = &[
    (
        "en",
        &[
            "the", "and", "is", "are", "was", "were", "to", "of", "in", "for", "with", "that",
            "this", "it", "you", "have", "has", "not", "but", "on", "at", "be", "as", "from",
            "will", "can", "would", "there", "their", "what", "which", "please", "we", "i",
        ],
    ),
    (
        "fr",
        &[
            "le", "la", "les", "des", "une", "est", "pour", "dans", "que", "qui", "avec", "sur",
            "pas", "plus", "nous", "vous", "être", "cette", "mais", "sont", "ont", "aux", "ce",
        ],
    ),
    (
        "de",
        &[
            "der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "nicht", "mit", "für",
            "auf", "von", "zu", "sich", "auch", "werden", "wurde", "haben", "sind", "oder", "aber",
        ],
    ),
    (
        "es",
        &[
            "el", "los", "las", "que", "por", "con", "para", "una", "es", "se", "del", "como",
            "pero", "son", "está", "este", "esta", "todo", "más", "muy", "hay", "sus",
        ],
    ),
    (
        "pt",
        &[
            "os", "as", "que", "em", "um", "uma", "para", "com", "não", "é", "se", "do", "da",
            "dos", "das", "mas", "são", "está", "este", "esta", "muito", "pelo", "pela",
        ],
    ),
    (
        "it",
        &[
            "il", "lo", "gli", "che", "di", "per", "con", "non", "è", "si", "del", "della", "sono",
            "questo", "questa", "anche", "come", "più", "nella", "alla",
        ],
    ),
    (
        "nl",
        &[
            "het", "een", "van", "is", "op", "te", "dat", "niet", "met", "voor", "zijn", "aan",
            "door", "maar", "ook", "worden", "deze", "naar", "wordt",
        ],
    ),
];

const NON_EN_DIACRITICS: &str = "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœđłşţğıåäö";

fn collect_text(state: &Json, depth: usize, out: &mut Vec<String>) {
    if depth > 6 {
        return;
    }
    match state {
        Json::Str(s) => out.push(s.clone()),
        Json::Obj(map) => map
            .iter()
            .for_each(|(_, v)| collect_text(v, depth + 1, out)),
        Json::Arr(list) => list.iter().for_each(|v| collect_text(v, depth + 1, out)),
        _ => {}
    }
}

pub fn state_text(state: &Json) -> String {
    let mut parts = Vec::new();
    collect_text(state, 0, &mut parts);
    parts.join(" ").chars().take(4000).collect()
}

fn script_counts(text: &str) -> Vec<(&'static str, usize)> {
    let mut counts: Vec<(&'static str, usize)> = Vec::new();
    let mut latin = 0;
    for ch in text.chars().filter(|c| c.is_alphabetic()) {
        let cp = ch as u32;
        if cp < 0x0250 || (0x1E00..=0x1EFF).contains(&cp) {
            latin += 1;
        } else if let Some((name, _)) = SCRIPTS
            .iter()
            .find(|(_, ranges)| ranges.iter().any(|(lo, hi)| (*lo..=*hi).contains(&cp)))
        {
            match counts.iter_mut().find(|(n, _)| n == name) {
                Some(entry) => entry.1 += 1,
                None => counts.push((name, 1)),
            }
        }
    }
    counts.push(("latin", latin));
    counts
}

fn guess_latin_language(text: &str) -> Option<&'static str> {
    let lowered = text.to_lowercase();
    let words: Vec<&str> = lowered
        .split(|c: char| !c.is_alphabetic())
        .filter(|w| !w.is_empty())
        .collect();
    if words.len() < 4 {
        return None;
    }
    let scores: Vec<(&str, usize)> = STOPWORDS
        .iter()
        .map(|(lang, list)| (*lang, words.iter().filter(|w| list.contains(w)).count()))
        .collect();
    let diacritics = lowered
        .chars()
        .filter(|c| NON_EN_DIACRITICS.contains(*c))
        .count();
    let rate = diacritics as f64 / lowered.chars().count().max(1) as f64;
    let en = scores[0].1;
    let (best_lang, best) =
        scores[1..].iter().fold(
            ("", 0),
            |acc, (lang, s)| if *s > acc.1 { (*lang, *s) } else { acc },
        );
    let english = if en > 0 { Some("en") } else { None };
    if best == 0 && rate < 0.02 {
        return english;
    }
    if best > 0 && best >= (en + 2).max(2) {
        return Some(best_lang);
    }
    if rate >= 0.04 && best > 0 && best >= en {
        return Some(best_lang);
    }
    english
}

pub struct Detection {
    pub script: String,
    pub is_english: bool,
    pub language: Option<&'static str>,
    pub non_latin_fraction: f64,
    pub json: Json,
}

pub fn analyse(state: &Json) -> Detection {
    let text = state_text(state);
    let counts = script_counts(&text);
    let total: usize = counts.iter().map(|(_, c)| c).sum();
    if total == 0 {
        let json = Json::obj([
            ("script", "unknown".into()),
            ("script_profile", Json::Obj(Vec::new())),
            ("language", Json::Null),
            ("is_english", true.into()),
            ("non_latin_fraction", 0.0.into()),
        ]);
        return Detection {
            script: "unknown".into(),
            is_english: true,
            language: None,
            non_latin_fraction: 0.0,
            json,
        };
    }
    let profile = Json::obj(
        counts
            .iter()
            .filter(|(_, c)| *c > 0)
            .map(|(n, c)| (*n, Json::Float(*c as f64 / total as f64))),
    );
    let script = counts
        .iter()
        .fold(
            ("", 0),
            |acc, (n, c)| if *c > acc.1 { (*n, *c) } else { acc },
        )
        .0;
    let latin = counts.last().map(|(_, c)| *c).unwrap_or(0);
    let non_latin_fraction = ((1.0 - latin as f64 / total as f64) * 1e4).round() / 1e4;
    let (language, is_english) = if script == "latin" {
        let lang = guess_latin_language(&text);
        (lang, matches!(lang, None | Some("en")))
    } else {
        (None, false)
    };
    let json = Json::obj([
        ("script", script.into()),
        ("script_profile", profile),
        ("language", language.into()),
        ("is_english", is_english.into()),
        ("non_latin_fraction", non_latin_fraction.into()),
    ]);
    Detection {
        script: script.to_string(),
        is_english,
        language,
        non_latin_fraction,
        json,
    }
}
