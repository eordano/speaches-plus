use std::collections::HashSet;

pub fn normalize_offer(sdp: &str) -> String {
    let stage_one = unify_ice_credentials(sdp);
    filter_audio_to_opus(&stage_one)
}

fn unify_ice_credentials(sdp: &str) -> String {
    let mut canonical_ufrag: Option<String> = None;
    let mut canonical_pwd: Option<String> = None;
    let mut out = String::with_capacity(sdp.len());

    for line in sdp.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if let Some(rest) = trimmed.strip_prefix("a=ice-ufrag:") {
            match &canonical_ufrag {
                None => {
                    canonical_ufrag = Some(rest.to_string());
                    out.push_str(line);
                }
                Some(canonical) => {
                    out.push_str("a=ice-ufrag:");
                    out.push_str(canonical);
                    out.push_str(line_terminator(line));
                }
            }
        } else if let Some(rest) = trimmed.strip_prefix("a=ice-pwd:") {
            match &canonical_pwd {
                None => {
                    canonical_pwd = Some(rest.to_string());
                    out.push_str(line);
                }
                Some(canonical) => {
                    out.push_str("a=ice-pwd:");
                    out.push_str(canonical);
                    out.push_str(line_terminator(line));
                }
            }
        } else {
            out.push_str(line);
        }
    }
    out
}

fn filter_audio_to_opus(sdp: &str) -> String {
    let lines: Vec<&str> = sdp.split_inclusive('\n').collect();

    let mut sections: Vec<std::collections::HashMap<String, String>> = Vec::new();
    let mut cur_audio: Option<usize> = None;
    for line in &lines {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if let Some(rest) = trimmed.strip_prefix("m=") {
            if rest.starts_with("audio ") {
                sections.push(std::collections::HashMap::new());
                cur_audio = Some(sections.len() - 1);
            } else {
                cur_audio = None;
            }
            continue;
        }
        if let (Some(idx), Some(rest)) = (cur_audio, trimmed.strip_prefix("a=rtpmap:")) {
            if let Some((pt, codec_info)) = rest.split_once(' ') {
                let codec = codec_info
                    .split('/')
                    .next()
                    .unwrap_or("")
                    .to_ascii_lowercase();
                sections[idx].insert(pt.to_string(), codec);
            }
        }
    }

    let keep_codecs: HashSet<&str> = ["opus", "telephone-event", "cn"].iter().copied().collect();
    let kept_per_section: Vec<HashSet<String>> = sections
        .iter()
        .map(|map| {
            map.iter()
                .filter_map(|(pt, codec)| {
                    if keep_codecs.contains(codec.as_str()) {
                        Some(pt.clone())
                    } else {
                        None
                    }
                })
                .collect()
        })
        .collect();

    let mut out = String::with_capacity(sdp.len());
    let mut cur_audio: Option<usize> = None;
    let mut audio_count_so_far: usize = 0;
    for line in &lines {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        let term = line_terminator(line);

        if let Some(rest) = trimmed.strip_prefix("m=") {
            if rest.starts_with("audio ") {
                let idx = audio_count_so_far;
                audio_count_so_far += 1;
                cur_audio = Some(idx);
                let kept = &kept_per_section[idx];

                if kept.is_empty() {
                    out.push_str(line);
                    continue;
                }

                let parts: Vec<&str> = rest.split_whitespace().collect();
                if parts.len() < 4 {
                    out.push_str(line);
                    continue;
                }
                let prefix = &parts[..3];
                let payloads = &parts[3..];
                let filtered: Vec<&str> = payloads
                    .iter()
                    .copied()
                    .filter(|pt| kept.contains(*pt))
                    .collect();
                if filtered.is_empty() {
                    out.push_str(line);
                    continue;
                }
                out.push_str("m=");
                out.push_str(prefix[0]);
                out.push(' ');
                out.push_str(prefix[1]);
                out.push(' ');
                out.push_str(prefix[2]);
                for pt in &filtered {
                    out.push(' ');
                    out.push_str(pt);
                }
                out.push_str(term);
                continue;
            } else {
                cur_audio = None;
                out.push_str(line);
                continue;
            }
        }

        if let Some(idx) = cur_audio {
            let kept = &kept_per_section[idx];
            if let Some(pt) = leading_payload_type(trimmed, "a=rtpmap:")
                .or_else(|| leading_payload_type(trimmed, "a=fmtp:"))
                .or_else(|| leading_payload_type(trimmed, "a=rtcp-fb:"))
            {
                if !kept.is_empty() && !kept.contains(&pt) {
                    continue;
                }
            }
        }

        out.push_str(line);
    }
    out
}

fn leading_payload_type(trimmed: &str, attr_prefix: &str) -> Option<String> {
    let rest = trimmed.strip_prefix(attr_prefix)?;

    let pt_token = rest.split_whitespace().next()?;
    if pt_token.chars().all(|c| c.is_ascii_digit()) {
        Some(pt_token.to_string())
    } else {
        None
    }
}

fn line_terminator(line: &str) -> &'static str {
    if line.ends_with("\r\n") {
        "\r\n"
    } else if line.ends_with('\n') {
        "\n"
    } else {
        ""
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unifies_ice_creds_across_bundle() {
        let input = "v=0\r\n\
                     a=group:BUNDLE 0 1\r\n\
                     m=audio 1 UDP/TLS/RTP/SAVPF 96\r\n\
                     a=mid:0\r\n\
                     a=ice-ufrag:AAA1\r\n\
                     a=ice-pwd:pwd-aaaa\r\n\
                     a=rtpmap:96 opus/48000/2\r\n\
                     m=application 1 DTLS/SCTP 5000\r\n\
                     a=mid:1\r\n\
                     a=ice-ufrag:BBB2\r\n\
                     a=ice-pwd:pwd-bbbb\r\n";
        let out = normalize_offer(input);
        let ufrags: Vec<&str> = out
            .lines()
            .filter(|l| l.starts_with("a=ice-ufrag"))
            .collect();
        let pwds: Vec<&str> = out.lines().filter(|l| l.starts_with("a=ice-pwd")).collect();
        assert_eq!(ufrags, vec!["a=ice-ufrag:AAA1", "a=ice-ufrag:AAA1"]);
        assert_eq!(pwds, vec!["a=ice-pwd:pwd-aaaa", "a=ice-pwd:pwd-aaaa"]);
    }

    #[test]
    fn strips_pcmu_when_offer_lists_it_first() {
        let input = "v=0\r\n\
                     m=audio 9 UDP/TLS/RTP/SAVPF 0 96 8\r\n\
                     a=mid:0\r\n\
                     a=rtpmap:0 PCMU/8000\r\n\
                     a=rtpmap:96 opus/48000/2\r\n\
                     a=fmtp:96 minptime=10;useinbandfec=1\r\n\
                     a=rtpmap:8 PCMA/8000\r\n";
        let out = normalize_offer(input);

        let m_line = out
            .lines()
            .find(|l| l.starts_with("m=audio"))
            .expect("m=audio line present");
        let payloads: Vec<&str> = m_line.split_whitespace().skip(3).collect();
        assert_eq!(payloads, vec!["96"], "expected opus-only m-line: {m_line}");

        assert!(!out.contains("PCMU"), "PCMU should be stripped: {out}");
        assert!(!out.contains("PCMA"), "PCMA should be stripped: {out}");

        assert!(out.contains("a=rtpmap:96 opus/48000/2"));
        assert!(out.contains("a=fmtp:96 minptime=10;useinbandfec=1"));
    }

    #[test]
    fn preserves_telephone_event_and_cn() {
        let input = "v=0\r\n\
                     m=audio 9 UDP/TLS/RTP/SAVPF 96 0 101 13\r\n\
                     a=mid:0\r\n\
                     a=rtpmap:96 opus/48000/2\r\n\
                     a=rtpmap:0 PCMU/8000\r\n\
                     a=rtpmap:101 telephone-event/8000\r\n\
                     a=rtpmap:13 CN/8000\r\n";
        let out = normalize_offer(input);
        let m_line = out
            .lines()
            .find(|l| l.starts_with("m=audio"))
            .expect("m=audio line present");
        let payloads: Vec<&str> = m_line.split_whitespace().skip(3).collect();

        assert_eq!(payloads, vec!["96", "101", "13"], "got: {m_line}");
    }

    #[test]
    fn leaves_video_section_alone() {
        let input = "v=0\r\n\
                     m=video 9 UDP/TLS/RTP/SAVPF 96 97\r\n\
                     a=mid:0\r\n\
                     a=rtpmap:96 VP8/90000\r\n\
                     a=rtpmap:97 H264/90000\r\n";
        let out = normalize_offer(input);
        assert!(out.contains("m=video 9 UDP/TLS/RTP/SAVPF 96 97"));
        assert!(out.contains("a=rtpmap:96 VP8/90000"));
        assert!(out.contains("a=rtpmap:97 H264/90000"));
    }

    #[test]
    fn passthrough_when_no_opus_present() {
        let input = "v=0\r\n\
                     m=audio 9 UDP/TLS/RTP/SAVPF 0 8\r\n\
                     a=mid:0\r\n\
                     a=rtpmap:0 PCMU/8000\r\n\
                     a=rtpmap:8 PCMA/8000\r\n";
        let out = normalize_offer(input);
        assert!(out.contains("m=audio 9 UDP/TLS/RTP/SAVPF 0 8"));
        assert!(out.contains("a=rtpmap:0 PCMU/8000"));
    }
}
