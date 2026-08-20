use anyhow::{anyhow, bail, Context};

use super::resample::downmix_and_resample_f32;
use super::types::TARGET_SAMPLE_RATE;

const EBML_MAGIC: [u8; 4] = [0x1A, 0x45, 0xDF, 0xA3];

const ID_SEGMENT: u64 = 0x1853_8067;
const ID_TRACKS: u64 = 0x1654_AE6B;
const ID_TRACK_ENTRY: u64 = 0xAE;
const ID_TRACK_NUMBER: u64 = 0xD7;
const ID_TRACK_TYPE: u64 = 0x83;
const ID_CODEC_ID: u64 = 0x86;
const ID_CODEC_PRIVATE: u64 = 0x63A2;
const ID_AUDIO: u64 = 0xE1;
const ID_CHANNELS: u64 = 0x9F;
const ID_CLUSTER: u64 = 0x1F43_B675;
const ID_SIMPLE_BLOCK: u64 = 0xA3;
const ID_BLOCK_GROUP: u64 = 0xA0;
const ID_BLOCK: u64 = 0xA1;

const TRACK_TYPE_AUDIO: u64 = 2;
const OPUS_DECODE_RATE: u32 = 48_000;

const ID_SEEK_HEAD: u64 = 0x114D_9B74;
const ID_INFO: u64 = 0x1549_A966;
const ID_CUES: u64 = 0x1C53_BB6B;
const ID_CHAPTERS: u64 = 0x1043_A770;
const ID_TAGS: u64 = 0x1254_C367;
const ID_ATTACHMENTS: u64 = 0x1941_A469;

pub fn is_webm(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && bytes[0..4] == EBML_MAGIC
}

fn is_level1_id(id: u64) -> bool {
    matches!(
        id,
        ID_SEEK_HEAD
            | ID_INFO
            | ID_TRACKS
            | ID_CLUSTER
            | ID_CUES
            | ID_CHAPTERS
            | ID_TAGS
            | ID_ATTACHMENTS
    )
}

fn vint_len(first: u8) -> usize {
    for i in 0..8 {
        if first & (0x80 >> i) != 0 {
            return i + 1;
        }
    }
    0
}

fn read_id(buf: &[u8], pos: &mut usize) -> Option<u64> {
    let first = *buf.get(*pos)?;
    let len = vint_len(first);
    if len == 0 || *pos + len > buf.len() {
        return None;
    }
    let mut v: u64 = 0;
    for i in 0..len {
        v = (v << 8) | buf[*pos + i] as u64;
    }
    *pos += len;
    Some(v)
}

fn read_size(buf: &[u8], pos: &mut usize) -> Option<(u64, bool)> {
    let first = *buf.get(*pos)?;
    let len = vint_len(first);
    if len == 0 || *pos + len > buf.len() {
        return None;
    }
    let mask: u8 = if len >= 8 { 0 } else { 0xFF >> len };
    let mut v: u64 = (first & mask) as u64;
    for i in 1..len {
        v = (v << 8) | buf[*pos + i] as u64;
    }
    *pos += len;
    let unknown = v == (1u64 << (7 * len)) - 1;
    Some((v, unknown))
}

fn read_uint(data: &[u8]) -> u64 {
    let mut v = 0u64;
    for &b in data {
        v = (v << 8) | b as u64;
    }
    v
}

struct OpusTrack {
    number: u64,
    channels: u32,
    pre_skip: usize,
}

struct WebmParser<'a> {
    buf: &'a [u8],
    track: Option<OpusTrack>,
    packets: Vec<&'a [u8]>,
}

impl<'a> WebmParser<'a> {
    fn parse_segment(&mut self, start: usize, end: usize) -> anyhow::Result<()> {
        let mut pos = start;
        while pos < end {
            let id = match read_id(self.buf, &mut pos) {
                Some(id) => id,
                None => break,
            };
            let (size, unknown) = match read_size(self.buf, &mut pos) {
                Some(s) => s,
                None => break,
            };
            let body_end = if unknown {
                end
            } else {
                (pos + size as usize).min(end)
            };
            match id {
                ID_TRACKS => {
                    self.parse_tracks(pos, body_end)?;
                    pos = body_end;
                }
                ID_CLUSTER => {
                    pos = self.parse_cluster(pos, body_end, unknown);
                }
                _ => pos = body_end,
            }
        }
        Ok(())
    }

    fn parse_tracks(&mut self, start: usize, end: usize) -> anyhow::Result<()> {
        let mut pos = start;
        while pos < end {
            let id = match read_id(self.buf, &mut pos) {
                Some(id) => id,
                None => break,
            };
            let (size, unknown) = match read_size(self.buf, &mut pos) {
                Some(s) => s,
                None => break,
            };
            let body_end = if unknown {
                end
            } else {
                (pos + size as usize).min(end)
            };
            if id == ID_TRACK_ENTRY {
                self.parse_track_entry(pos, body_end);
            }
            pos = body_end;
        }
        Ok(())
    }

    fn parse_track_entry(&mut self, start: usize, end: usize) {
        let mut pos = start;
        let mut number: u64 = 0;
        let mut track_type: u64 = 0;
        let mut is_opus = false;
        let mut channels: u32 = 1;
        let mut pre_skip: usize = 0;
        while pos < end {
            let id = match read_id(self.buf, &mut pos) {
                Some(id) => id,
                None => break,
            };
            let (size, unknown) = match read_size(self.buf, &mut pos) {
                Some(s) => s,
                None => break,
            };
            let body_end = if unknown {
                end
            } else {
                (pos + size as usize).min(end)
            };
            let data = &self.buf[pos..body_end];
            match id {
                ID_TRACK_NUMBER => number = read_uint(data),
                ID_TRACK_TYPE => track_type = read_uint(data),
                ID_CODEC_ID => is_opus = data == b"A_OPUS",
                ID_CODEC_PRIVATE => {
                    if data.len() >= 19 && &data[0..8] == b"OpusHead" {
                        channels = data[9] as u32;
                        pre_skip = u16::from_le_bytes([data[10], data[11]]) as usize;
                    }
                }
                ID_AUDIO => {
                    if let Some(c) = parse_audio_channels(data) {
                        channels = c;
                    }
                }
                _ => {}
            }
            pos = body_end;
        }
        if track_type == TRACK_TYPE_AUDIO && is_opus && self.track.is_none() {
            self.track = Some(OpusTrack {
                number,
                channels: channels.max(1),
                pre_skip,
            });
        }
    }

    fn parse_cluster(&mut self, start: usize, end: usize, cluster_unknown: bool) -> usize {
        let mut pos = start;
        while pos < end {
            let id_start = pos;
            let id = match read_id(self.buf, &mut pos) {
                Some(id) => id,
                None => return end,
            };

            if cluster_unknown && is_level1_id(id) {
                return id_start;
            }
            let (size, unknown) = match read_size(self.buf, &mut pos) {
                Some(s) => s,
                None => return end,
            };
            let body_end = if unknown {
                end
            } else {
                (pos + size as usize).min(end)
            };
            match id {
                ID_SIMPLE_BLOCK => self.parse_block(pos, body_end),
                ID_BLOCK_GROUP => self.parse_block_group(pos, body_end),
                _ => {}
            }
            pos = body_end;
        }
        end
    }

    fn parse_block_group(&mut self, start: usize, end: usize) {
        let mut pos = start;
        while pos < end {
            let id = match read_id(self.buf, &mut pos) {
                Some(id) => id,
                None => break,
            };
            let (size, unknown) = match read_size(self.buf, &mut pos) {
                Some(s) => s,
                None => break,
            };
            let body_end = if unknown {
                end
            } else {
                (pos + size as usize).min(end)
            };
            if id == ID_BLOCK {
                self.parse_block(pos, body_end);
            }
            pos = body_end;
        }
    }

    fn parse_block(&mut self, start: usize, end: usize) {
        let track_no = match &self.track {
            Some(t) => t.number,
            None => return,
        };
        let mut pos = start;
        let block_track = match read_size(self.buf, &mut pos) {
            Some((v, _)) => v,
            None => return,
        };
        if block_track != track_no {
            return;
        }
        if pos + 3 > end {
            return;
        }
        pos += 2;
        let flags = self.buf[pos];
        pos += 1;
        let lacing = (flags >> 1) & 0x03;
        match lacing {
            0 => {
                if pos < end {
                    self.packets.push(&self.buf[pos..end]);
                }
            }
            _ => self.parse_laced(pos, end, lacing),
        }
    }

    fn parse_laced(&mut self, mut pos: usize, end: usize, lacing: u8) {
        if pos >= end {
            return;
        }
        let frames = self.buf[pos] as usize + 1;
        pos += 1;
        let mut sizes: Vec<usize> = Vec::with_capacity(frames);
        match lacing {
            2 => {
                if pos > end {
                    return;
                }
                let remaining = end - pos;
                if frames == 0 || !remaining.is_multiple_of(frames) {
                    return;
                }
                let each = remaining / frames;
                for _ in 0..frames {
                    sizes.push(each);
                }
            }
            1 => {
                for _ in 0..frames - 1 {
                    let mut s = 0usize;
                    loop {
                        if pos >= end {
                            return;
                        }
                        let b = self.buf[pos];
                        pos += 1;
                        s += b as usize;
                        if b != 0xFF {
                            break;
                        }
                    }
                    sizes.push(s);
                }
            }
            3 => {
                let (first, _) = match read_size(self.buf, &mut pos) {
                    Some(v) => v,
                    None => return,
                };
                let mut prev = first as i64;
                sizes.push(first as usize);
                for _ in 0..frames.saturating_sub(2) {
                    let len = vint_len(*self.buf.get(pos).unwrap_or(&0));
                    if len == 0 || pos + len > end {
                        return;
                    }
                    let lace_mask: u8 = if len >= 8 { 0 } else { 0xFF >> len };
                    let mut raw: i64 = (self.buf[pos] & lace_mask) as i64;
                    for i in 1..len {
                        raw = (raw << 8) | self.buf[pos + i] as i64;
                    }
                    pos += len;
                    let bias = (1i64 << (7 * len - 1)) - 1;
                    let delta = raw - bias;
                    prev += delta;
                    if prev < 0 {
                        return;
                    }
                    sizes.push(prev as usize);
                }
            }
            _ => return,
        }
        for &s in &sizes {
            if pos + s > end {
                return;
            }
            self.packets.push(&self.buf[pos..pos + s]);
            pos += s;
        }
        if pos < end {
            self.packets.push(&self.buf[pos..end]);
        }
    }
}

fn parse_audio_channels(data: &[u8]) -> Option<u32> {
    let mut pos = 0usize;
    while pos < data.len() {
        let id = read_id(data, &mut pos)?;
        let (size, _) = read_size(data, &mut pos)?;
        let body_end = (pos + size as usize).min(data.len());
        if id == ID_CHANNELS {
            return Some(read_uint(&data[pos..body_end]).max(1) as u32);
        }
        pos = body_end;
    }
    None
}

pub fn decode_webm_opus_to_16k_mono(bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
    let mut pos = 0usize;
    let mut segment: Option<(usize, usize)> = None;
    while pos < bytes.len() {
        let id = read_id(bytes, &mut pos).ok_or_else(|| anyhow!("webm: bad EBML id"))?;
        let (size, unknown) =
            read_size(bytes, &mut pos).ok_or_else(|| anyhow!("webm: bad EBML size"))?;
        let body_end = if unknown {
            bytes.len()
        } else {
            (pos + size as usize).min(bytes.len())
        };
        if id == ID_SEGMENT {
            segment = Some((pos, body_end));
            break;
        }
        pos = body_end;
    }
    let (seg_start, seg_end) = segment.context("webm: no Segment element")?;

    let mut parser = WebmParser {
        buf: bytes,
        track: None,
        packets: Vec::new(),
    };
    parser.parse_segment(seg_start, seg_end)?;

    let track = parser
        .track
        .context("webm: no Opus audio track (only A_OPUS is supported)")?;
    if parser.packets.is_empty() {
        bail!("webm: no audio packets found");
    }

    let out_channels = if track.channels > 1 { 2usize } else { 1usize };
    let mut decoder = opus::Decoder::new(
        OPUS_DECODE_RATE,
        if out_channels == 1 {
            opus::Channels::Mono
        } else {
            opus::Channels::Stereo
        },
    )
    .context("opus decoder create")?;

    let max_frame = 5760 * out_channels;
    let mut pcm_buf = vec![0f32; max_frame];
    let mut all_samples: Vec<f32> = Vec::new();
    for packet in &parser.packets {
        match decoder.decode_float(packet, &mut pcm_buf, false) {
            Ok(n) => all_samples.extend_from_slice(&pcm_buf[..n * out_channels]),
            Err(_) => continue,
        }
    }

    if all_samples.len() > track.pre_skip * out_channels {
        all_samples = all_samples[(track.pre_skip * out_channels)..].to_vec();
    }

    let mono = if out_channels == 1 {
        all_samples
    } else {
        let frames = all_samples.len() / out_channels;
        let mut m = Vec::with_capacity(frames);
        for i in 0..frames {
            m.push((all_samples[i * 2] + all_samples[i * 2 + 1]) * 0.5);
        }
        m
    };

    if OPUS_DECODE_RATE == TARGET_SAMPLE_RATE {
        return Ok(mono);
    }
    Ok(downmix_and_resample_f32(
        &mono,
        1,
        OPUS_DECODE_RATE as usize,
        TARGET_SAMPLE_RATE as usize,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn elem(id: &[u8], body: &[u8]) -> Vec<u8> {
        let mut v = Vec::from(id);
        v.push(0x10);
        let n = body.len() as u32;
        v.extend_from_slice(&n.to_be_bytes()[1..]);
        v.extend_from_slice(body);
        v
    }

    fn opus_head(channels: u8) -> Vec<u8> {
        let mut h = Vec::from(&b"OpusHead"[..]);
        h.push(1);
        h.push(channels);
        h.extend_from_slice(&0u16.to_le_bytes());
        h.extend_from_slice(&48_000u32.to_le_bytes());
        h.extend_from_slice(&0u16.to_le_bytes());
        h.push(0);
        h
    }

    fn simple_block(track: u8, packet: &[u8]) -> Vec<u8> {
        let mut b = vec![0x80 | track];
        b.extend_from_slice(&0i16.to_be_bytes());
        b.push(0x00);
        b.extend_from_slice(packet);
        b
    }

    #[test]
    fn webm_opus_roundtrip_decodes_to_16k_mono() {
        let mut enc =
            opus::Encoder::new(48_000, opus::Channels::Mono, opus::Application::Audio).unwrap();
        let frame = 960usize;
        let mut packets: Vec<Vec<u8>> = Vec::new();
        for f in 0..25 {
            let pcm: Vec<f32> = (0..frame)
                .map(|i| {
                    let t = (f * frame + i) as f32 / 48_000.0;
                    0.25 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                })
                .collect();
            let mut out = vec![0u8; 4000];
            let n = enc.encode_float(&pcm, &mut out).unwrap();
            out.truncate(n);
            packets.push(out);
        }

        let mut track_entry = Vec::new();
        track_entry.extend(elem(&[ID_TRACK_NUMBER as u8], &[1]));
        track_entry.extend(elem(&[ID_TRACK_TYPE as u8], &[TRACK_TYPE_AUDIO as u8]));
        track_entry.extend(elem(&[ID_CODEC_ID as u8], b"A_OPUS"));
        track_entry.extend(elem(
            &[(ID_CODEC_PRIVATE >> 8) as u8, ID_CODEC_PRIVATE as u8],
            &opus_head(1),
        ));
        let audio = elem(&[ID_CHANNELS as u8], &[1]);
        track_entry.extend(elem(&[ID_AUDIO as u8], &audio));

        let tracks = elem(
            &[0x16, 0x54, 0xAE, 0x6B],
            &elem(&[ID_TRACK_ENTRY as u8], &track_entry),
        );

        let unknown_cluster = |blocks: &[Vec<u8>]| -> Vec<u8> {
            let mut body = Vec::new();
            for p in blocks {
                body.extend(elem(&[ID_SIMPLE_BLOCK as u8], &simple_block(1, p)));
            }
            let mut c = Vec::from(&[0x1F, 0x43, 0xB6, 0x75][..]);
            c.push(0x01);
            c.extend_from_slice(&[0xFF; 7]);
            c.extend_from_slice(&body);
            c
        };
        let (first, second) = packets.split_at(12);

        let mut segment_body = Vec::new();
        segment_body.extend(tracks);
        segment_body.extend(unknown_cluster(first));
        segment_body.extend(unknown_cluster(second));
        let segment = elem(&[0x18, 0x53, 0x80, 0x67], &segment_body);

        let mut webm = elem(&[0x1A, 0x45, 0xDF, 0xA3], b"");
        webm.extend(segment);

        assert!(is_webm(&webm));
        let samples = decode_webm_opus_to_16k_mono(&webm).expect("decode webm opus");

        assert!(
            samples.len() > 6000 && samples.len() < 9000,
            "unexpected sample count: {}",
            samples.len()
        );
        let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
        assert!(rms > 0.05, "decoded audio too quiet: rms={rms}");
    }
}
