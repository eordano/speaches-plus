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
            0x0600..=0x06FF | 0x0750..=0x077F | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF => {
                m.arabic += 1
            }
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
    fn spanish_text_is_clean_under_a_latin_prior() {
        let text = "Tengo una tendencia a tener la idea de un poema y no tener una idea real \
                    sobre las cosas que me rodean cada noche antes de escribir.";
        assert_eq!(suspect_reason(text, ExpectedScript::Latin), None);
        assert_eq!(suspect_reason(text, ExpectedScript::Any), None);
    }

    #[test]
    fn arabic_output_under_a_latin_prior_is_the_archive_confabulation_case() {
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
    }

    #[test]
    fn digit_table_flood_flags_without_any_prior() {
        let text = "<table>1234567891011121314151617181920212223242526272829303132333435363738\
                    39404142434445464748495051525354555657585960616263646566676869707172737475</table>";
        assert_eq!(suspect_reason(text, ExpectedScript::Any), Some("digit-flood"));
    }

    #[test]
    fn short_outputs_never_get_a_script_verdict() {
        assert_eq!(suspect_reason("你好", ExpectedScript::Latin), None);
        assert_eq!(suspect_reason("ESA", ExpectedScript::Latin), None);
    }

    #[test]
    fn mixed_but_mostly_matching_text_is_not_flagged() {
        let text = "El café de la esquina sirve razonablemente bien y el menú tiene un par de \
                    palabras en 日本語 pero el resto es castellano corriente de todos los días.";
        assert_eq!(suspect_reason(text, ExpectedScript::Latin), None);
    }
}
