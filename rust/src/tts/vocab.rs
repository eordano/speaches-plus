use std::collections::HashMap;
use std::sync::OnceLock;

const KOKORO_PAD: &str = "$";
const KOKORO_PUNCTUATION: &str = ";:,.!?¡¿—…\"«»“” ";
const KOKORO_LETTERS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const KOKORO_LETTERS_IPA: &str = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ";

pub const MAX_PHONEME_LENGTH: usize = crate::defaults::kokoro::MAX_PHONEME_LENGTH;

static VOCAB: OnceLock<HashMap<char, i64>> = OnceLock::new();

fn vocab() -> &'static HashMap<char, i64> {
    VOCAB.get_or_init(|| {
        let mut map: HashMap<char, i64> = HashMap::with_capacity(256);
        let mut idx: i64 = 0;
        for s in [
            KOKORO_PAD,
            KOKORO_PUNCTUATION,
            KOKORO_LETTERS,
            KOKORO_LETTERS_IPA,
        ] {
            for ch in s.chars() {
                map.insert(ch, idx);
                idx += 1;
            }
        }
        map
    })
}

pub fn tokenize(phonemes: &str) -> Vec<i64> {
    let v = vocab();
    let mut out = Vec::with_capacity(phonemes.len());
    for ch in phonemes.chars() {
        if let Some(&id) = v.get(&ch) {
            out.push(id);
        }
    }
    out
}

pub fn clean_phonemes(phonemes: &str) -> String {
    let phonemes = phonemes
        .replace("kəkˈoːɹoʊ", "kˈoʊkəɹoʊ")
        .replace("kəkˈɔːɹəʊ", "kˈəʊkəɹəʊ");

    let mapped: String = phonemes
        .chars()
        .map(|c| match c {
            'ʲ' => 'j',
            'r' => 'ɹ',
            'x' => 'k',
            'ɬ' => 'l',
            other => other,
        })
        .collect();

    let v = vocab();
    let cleaned: String = mapped.chars().filter(|c| v.contains_key(c)).collect();
    cleaned.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_indices_match_python() {
        let phonemes = "ðˈʌ kwˈɪk bɹˈaʊn";
        let tokens = tokenize(phonemes);
        assert!(!tokens.is_empty());
        assert_eq!(*vocab().get(&'$').unwrap(), 0);
        assert_eq!(*vocab().get(&' ').unwrap(), 16);
        assert_eq!(*vocab().get(&'A').unwrap(), 17);
    }

    #[test]
    fn cleans_r_to_alveolar_approximant() {
        let cleaned = clean_phonemes("brown");
        assert!(cleaned.contains('ɹ'));
        assert!(!cleaned.contains('r'));
    }

    #[test]
    fn drops_unknown_runes() {
        let cleaned = clean_phonemes("hello@world");
        assert!(!cleaned.contains('@'));
    }
}
