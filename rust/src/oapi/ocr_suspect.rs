#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpectedScript {
    Any,
    Latin,
    Arabic,
    Cjk,
    Cyrillic,
}

impl ExpectedScript {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "any" => Some(ExpectedScript::Any),
            "latin" => Some(ExpectedScript::Latin),
            "arabic" => Some(ExpectedScript::Arabic),
            "cjk" => Some(ExpectedScript::Cjk),
            "cyrillic" => Some(ExpectedScript::Cyrillic),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScriptMix {
    pub letters: u32,
    pub latin: u32,
    pub arabic: u32,
    pub cjk: u32,
    pub cyrillic: u32,
    pub digits: u32,
    pub total_chars: u32,
}

pub fn script_mix(text: &str) -> ScriptMix {
    let mut m = ScriptMix::default();
    for c in text.chars() {
        if c.is_whitespace() {
            continue;
        }
        m.total_chars += 1;
        if c.is_ascii_digit() {
            m.digits += 1;
        }
        if !c.is_alphabetic() {
            continue;
        }
        m.letters += 1;
        let u = c as u32;
        match u {
            0x0041..=0x024F => m.latin += 1,
            0x0400..=0x04FF => m.cyrillic += 1,
            0x0600..=0x06FF | 0x0750..=0x077F | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF => m.arabic += 1,
            0x3040..=0x30FF | 0x4E00..=0x9FFF | 0xAC00..=0xD7AF => m.cjk += 1,
            _ => {}
        }
    }
    m
}

const MIN_LETTERS_FOR_A_SCRIPT_VERDICT: u32 = 40;

const FOREIGN_FRACTION_THAT_FLAGS: f32 = 0.5;

const DIGIT_FRACTION_THAT_FLAGS: f32 = 0.6;

const MIN_CHARS_FOR_A_DIGIT_VERDICT: u32 = 120;

pub const BLANK_INK_LUMA_DELTA_FROM_THE_MODAL_PAPER_TONE: i32 = 24;

pub const BLANK_INK_FRACTION_UNDER_WHICH_A_PAGE_HAS_NOTHING_TO_READ: f64 = 0.0005;

pub const BLANK_MIN_OUTPUT_CHARS_THAT_MAKE_GENERATIVE_TEXT_A_CONFABULATION: usize = 8;

pub fn ink_fraction(luma_hist: &[u32; 256]) -> f64 {
    let total: u64 = luma_hist.iter().map(|&n| n as u64).sum();
    if total == 0 {
        return 0.0;
    }
    let modal_paper = luma_hist
        .iter()
        .enumerate()
        .max_by_key(|&(_, &n)| n)
        .map(|(v, _)| v as i32)
        .unwrap_or(255);
    let ink: u64 = luma_hist
        .iter()
        .enumerate()
        .filter(|&(v, _)| {
            (v as i32 - modal_paper).abs() > BLANK_INK_LUMA_DELTA_FROM_THE_MODAL_PAPER_TONE
        })
        .map(|(_, &n)| n as u64)
        .sum();
    ink as f64 / total as f64
}

pub fn near_blank(luma_hist: &[u32; 256]) -> bool {
    ink_fraction(luma_hist) < BLANK_INK_FRACTION_UNDER_WHICH_A_PAGE_HAS_NOTHING_TO_READ
}

pub fn blank_page_suspect(input_near_blank: bool, text: &str) -> Option<&'static str> {
    (input_near_blank
        && text.chars().filter(|c| !c.is_whitespace()).count()
            >= BLANK_MIN_OUTPUT_CHARS_THAT_MAKE_GENERATIVE_TEXT_A_CONFABULATION)
        .then_some("blank-page")
}

pub fn suspect_reason(text: &str, expected: ExpectedScript) -> Option<&'static str> {
    let m = script_mix(text);
    if m.total_chars >= MIN_CHARS_FOR_A_DIGIT_VERDICT
        && m.digits as f32 / m.total_chars as f32 > DIGIT_FRACTION_THAT_FLAGS
    {
        return Some("digit-flood");
    }
    if expected == ExpectedScript::Any || m.letters < MIN_LETTERS_FOR_A_SCRIPT_VERDICT {
        return None;
    }
    let matching = match expected {
        ExpectedScript::Any => unreachable!(),
        ExpectedScript::Latin => m.latin,
        ExpectedScript::Arabic => m.arabic,
        ExpectedScript::Cjk => m.cjk,
        ExpectedScript::Cyrillic => m.cyrillic,
    };
    let foreign = m.letters.saturating_sub(matching);
    if foreign as f32 / m.letters as f32 > FOREIGN_FRACTION_THAT_FLAGS {
        return Some("script-mismatch");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn script_mismatch_and_digit_floods_are_suspect_but_matching_short_or_mixed_text_is_not() {
        let text = "حقيقة أن نسمي هذا الكتاب من أحكام النبي فقط فإنها ليست فقط من إمامنا بل هي \
                    أيضا من حقائق التي تعكس الحياة وتصرف إلى الواقع فأنا أستاذ في شعوبنا";
        assert_eq!(
            suspect_reason(text, ExpectedScript::Latin),
            Some("script-mismatch"),
            "the archive photos produced exactly this failure: fluent Arabic from Spanish handwriting"
        );
        assert_eq!(
            suspect_reason(text, ExpectedScript::Arabic),
            None,
            "the same output under a matching prior is not suspect"
        );
        assert_eq!(suspect_reason(text, ExpectedScript::Any), None);
        {
            let text = "Tengo una tendencia a tener la idea de un poema y no tener una idea real \
                        sobre las cosas que me rodean cada noche antes de escribir.";
            assert_eq!(suspect_reason(text, ExpectedScript::Latin), None);
            assert_eq!(suspect_reason(text, ExpectedScript::Any), None);
        }
        {
            let text = "<table>1234567891011121314151617181920212223242526272829303132333435363738\
                        39404142434445464748495051525354555657585960616263646566676869707172737475</table>";
            assert_eq!(
                suspect_reason(text, ExpectedScript::Any),
                Some("digit-flood")
            );
        }
        {
            assert_eq!(suspect_reason("你好", ExpectedScript::Latin), None);
            assert_eq!(suspect_reason("ESA", ExpectedScript::Latin), None);
        }
        {
            let text = "El café de la esquina sirve razonablemente bien y el menú tiene un par de \
                        palabras en 日本語 pero el resto es castellano corriente de todos los días.";
            assert_eq!(suspect_reason(text, ExpectedScript::Latin), None);
        }
    }

    #[test]
    fn a_white_page_is_near_blank_and_flags_prose_but_a_printed_paragraph_never_is() {
        let mut hist = [0u32; 256];
        hist[250] = 900_000;
        hist[248] = 60_000;
        hist[252] = 40_000;
        hist[20] = 400;
        assert!(
            near_blank(&hist),
            "ink fraction {} must read as blank",
            ink_fraction(&hist)
        );
        assert_eq!(
            blank_page_suspect(
                true,
                "six styles of hair extensions are shown in the figure"
            ),
            Some("blank-page"),
            "the archive run confabulated exactly this on a white 144 Mpx page"
        );
        assert_eq!(
            blank_page_suspect(true, "(0) = 1 +"),
            None,
            "a tiny artifact under the char floor is not worth a verdict"
        );
        assert_eq!(blank_page_suspect(true, ""), None);
        assert_eq!(
            blank_page_suspect(false, "any amount of text on a page with ink"),
            None
        );
        {
            let mut hist = [0u32; 256];
            hist[245] = 940_000;
            for v in 10..40 {
                hist[v] = 2_000;
            }
            assert!(!near_blank(&hist), "6% ink coverage must not read as blank");
        }
    }

}
