use std::collections::HashMap;

use crate::unicharset::{Recoder, Unicharset};
use crate::{Error, Logits};

pub const DEFAULT_BEAM_WIDTH: usize = 8;
const MAX_EXPANSIONS: usize = 16;
const MIN_PROB: f32 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CodeStep {
    pub code: usize,
    pub t: usize,
    pub prob: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedChar {
    pub unichar_id: usize,
    pub text: String,
    pub t: usize,
    pub prob: f32,
}

fn check(logits: &Logits, null_code: usize) -> Result<(), Error> {
    if logits.classes == 0 {
        return Err(Error::Ctc("zero classes".into()));
    }
    if logits.data.len() != logits.timesteps * logits.classes {
        return Err(Error::Ctc(format!(
            "logits buffer {} != {} timesteps x {} classes",
            logits.data.len(),
            logits.timesteps,
            logits.classes
        )));
    }
    if null_code >= logits.classes {
        return Err(Error::Ctc(format!(
            "null code {} outside {} classes",
            null_code, logits.classes
        )));
    }
    Ok(())
}

pub fn best_path(logits: &Logits, null_code: usize) -> Result<Vec<CodeStep>, Error> {
    check(logits, null_code)?;
    let mut res = Vec::new();
    let mut prev = usize::MAX;
    for t in 0..logits.timesteps {
        let row = &logits.data[t * logits.classes..(t + 1) * logits.classes];
        let (code, prob) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, &p)| (i, p))
            .unwrap();
        if code != null_code && code != prev {
            res.push(CodeStep { code, t, prob });
        }
        prev = code;
    }
    Ok(res)
}

fn logsumexp(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

const ROOT: u32 = u32::MAX;

struct Node {
    parent: u32,
    code: u32,
    t: u32,
    prob: f32,
}

#[derive(Clone, Copy)]
struct Beam {
    node: u32,
    p_b: f32,
    p_nb: f32,
}

impl Beam {
    fn total(&self) -> f32 {
        logsumexp(self.p_b, self.p_nb)
    }
}

struct Merger {
    next: Vec<Beam>,
    node_slot: HashMap<u32, u32>,
}

impl Merger {
    fn merge(&mut self, node: u32, blank: bool, lp: f32) {
        let idx = *self.node_slot.entry(node).or_insert_with(|| {
            self.next.push(Beam {
                node,
                p_b: f32::NEG_INFINITY,
                p_nb: f32::NEG_INFINITY,
            });
            (self.next.len() - 1) as u32
        });
        let b = &mut self.next[idx as usize];
        if blank {
            b.p_b = logsumexp(b.p_b, lp);
        } else {
            b.p_nb = logsumexp(b.p_nb, lp);
        }
    }
}

pub fn beam_decode(
    logits: &Logits,
    null_code: usize,
    beam_width: usize,
) -> Result<Vec<CodeStep>, Error> {
    check(logits, null_code)?;
    let width = beam_width.max(1);
    let mut nodes: Vec<Node> = Vec::with_capacity(1024);
    let mut child_map: HashMap<(u32, u32), u32> = HashMap::with_capacity(1024);
    let mut beams: Vec<Beam> = vec![Beam {
        node: ROOT,
        p_b: 0.0,
        p_nb: f32::NEG_INFINITY,
    }];
    let mut merger = Merger {
        next: Vec::with_capacity(width * (MAX_EXPANSIONS + 1)),
        node_slot: HashMap::with_capacity(width * (MAX_EXPANSIONS + 1)),
    };
    let mut cands: Vec<u32> = Vec::with_capacity(logits.classes);
    for t in 0..logits.timesteps {
        let row = &logits.data[t * logits.classes..(t + 1) * logits.classes];
        cands.clear();
        cands.extend(0..logits.classes as u32);
        if cands.len() > MAX_EXPANSIONS {
            cands.select_nth_unstable_by(MAX_EXPANSIONS - 1, |&a, &b| {
                row[b as usize].total_cmp(&row[a as usize])
            });
            cands.truncate(MAX_EXPANSIONS);
        }
        if !cands.contains(&(null_code as u32)) {
            cands.push(null_code as u32);
        }
        merger.next.clear();
        merger.node_slot.clear();
        for bi in 0..beams.len() {
            let beam = beams[bi];
            let total = beam.total();
            let last = if beam.node == ROOT {
                None
            } else {
                Some(nodes[beam.node as usize].code)
            };
            for &c in &cands {
                let p = row[c as usize].max(MIN_PROB);
                let lp = p.ln();
                if c as usize == null_code {
                    merger.merge(beam.node, true, total + lp);
                    continue;
                }
                let child = |nodes: &mut Vec<Node>, child_map: &mut HashMap<(u32, u32), u32>| {
                    *child_map.entry((beam.node, c)).or_insert_with(|| {
                        nodes.push(Node {
                            parent: beam.node,
                            code: c,
                            t: t as u32,
                            prob: p,
                        });
                        (nodes.len() - 1) as u32
                    })
                };
                if last == Some(c) {
                    merger.merge(beam.node, false, beam.p_nb + lp);
                    if beam.p_b > f32::NEG_INFINITY {
                        let ext = child(&mut nodes, &mut child_map);
                        merger.merge(ext, false, beam.p_b + lp);
                    }
                } else {
                    let ext = child(&mut nodes, &mut child_map);
                    merger.merge(ext, false, total + lp);
                }
            }
        }
        std::mem::swap(&mut beams, &mut merger.next);
        beams.sort_by(|a, b| b.total().total_cmp(&a.total()));
        beams.truncate(width);
    }
    beams.sort_by(|a, b| b.total().total_cmp(&a.total()));
    let mut res = Vec::new();
    if let Some(best) = beams.first() {
        let mut cur = best.node;
        while cur != ROOT {
            let n = &nodes[cur as usize];
            res.push(CodeStep {
                code: n.code as usize,
                t: n.t as usize,
                prob: n.prob,
            });
            cur = n.parent;
        }
        res.reverse();
    }
    Ok(res)
}

pub fn codes_to_unichars(
    steps: &[CodeStep],
    recoder: &Recoder,
    unicharset: &Unicharset,
) -> Vec<DecodedChar> {
    let mut res = Vec::new();
    let mut pending: Vec<i32> = Vec::new();
    let mut pending_t = 0usize;
    let mut pending_p = 1.0f32;
    for s in steps {
        if pending.is_empty() {
            pending_t = s.t;
            pending_p = 1.0;
        }
        pending.push(s.code as i32);
        pending_p *= s.prob;
        if let Some(id) = recoder.decode(&pending) {
            if let Some(text) = unicharset.glyph(id) {
                if !text.is_empty() {
                    res.push(DecodedChar {
                        unichar_id: id,
                        text: text.to_string(),
                        t: pending_t,
                        prob: pending_p,
                    });
                }
            }
            pending.clear();
        } else if !recoder.is_prefix(&pending) {
            pending.clear();
        }
    }
    res
}
