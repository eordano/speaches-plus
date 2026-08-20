use std::ffi::{c_char, c_int, CStr, CString};
use std::sync::{Mutex, OnceLock};

use anyhow::{anyhow, Result};

unsafe extern "C" {
    fn sp_espeak_init(data_path: *const c_char) -> c_int;
    fn sp_espeak_set_voice(name: *const c_char) -> c_int;
    fn sp_espeak_text_to_ipa(text: *const c_char, out: *mut c_char, out_size: *mut c_int) -> c_int;
    #[allow(dead_code)]
    fn sp_espeak_terminate();
}

struct State {
    inited: bool,
    voice: Option<String>,
}

static STATE: OnceLock<Mutex<State>> = OnceLock::new();

fn state() -> &'static Mutex<State> {
    STATE.get_or_init(|| {
        Mutex::new(State {
            inited: false,
            voice: None,
        })
    })
}

pub fn init(data_path: Option<&str>) -> Result<()> {
    let mut s = state()
        .lock()
        .map_err(|_| anyhow!("phonemizer state poisoned"))?;
    if s.inited {
        return Ok(());
    }
    let cpath = match data_path {
        Some(p) if !p.is_empty() => Some(CString::new(p).unwrap()),
        _ => None,
    };
    let ptr = cpath.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
    let rc = unsafe { sp_espeak_init(ptr) };
    if rc < 0 {
        return Err(anyhow!("espeak_Initialize failed: {rc}"));
    }
    s.inited = true;
    Ok(())
}

pub fn phonemize(text: &str, lang: &str) -> Result<String> {
    let mut s = state()
        .lock()
        .map_err(|_| anyhow!("phonemizer state poisoned"))?;
    if !s.inited {
        return Err(anyhow!("phonemizer not initialised"));
    }
    if s.voice.as_deref() != Some(lang) {
        let cname = CString::new(lang).unwrap();
        let rc = unsafe { sp_espeak_set_voice(cname.as_ptr()) };
        if rc != 0 {
            return Err(anyhow!("espeak_SetVoiceByName({lang:?}) failed: {rc}"));
        }
        s.voice = Some(lang.to_string());
    }

    let ctext = CString::new(text).map_err(|e| anyhow!("text contains NUL: {e}"))?;

    let mut cap = 4096i32;
    loop {
        let mut buf = vec![0u8; cap as usize];
        let mut size = cap;
        let rc = unsafe {
            sp_espeak_text_to_ipa(
                ctext.as_ptr(),
                buf.as_mut_ptr() as *mut c_char,
                &mut size as *mut c_int,
            )
        };
        match rc {
            0 => {
                let n = size.max(0) as usize;
                let cstr = unsafe { CStr::from_ptr(buf.as_ptr() as *const c_char) };
                let bytes = cstr.to_bytes();
                let len = bytes.len().min(n.max(bytes.len()));
                let s = std::str::from_utf8(&buf[..len])
                    .map_err(|e| anyhow!("non-utf8 ipa output: {e}"))?
                    .trim_end_matches('\0')
                    .to_string();
                return Ok(s);
            }
            -2 => {
                cap = (size + 1).max(cap * 2);
                continue;
            }
            other => return Err(anyhow!("text_to_ipa failed: {other}")),
        }
    }
}
