use anyhow::Result;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;

use crate::name::Names;
use crate::records::Record;
use crate::score::Scored;

pub const SESSION_READER_TAG_THRESHOLD: f32 = 0.35;

#[derive(Serialize)]
struct FeatureOut {
    id: usize,
    name: String,
    desc: String,
    stance: String,
    conf: u32,
}

#[derive(Serialize)]
struct MsgOut<'a> {
    line: usize,
    kind: &'a str,
    agent: &'a str,
    host: &'a str,
    time: &'a str,
    text: &'a str,
    q: Vec<u8>,
    v: Vec<f32>,
}

#[derive(Serialize)]
struct SessionOut<'a> {
    group: String,
    host: &'a str,
    users: usize,
    msgs: Vec<MsgOut<'a>>,
}

fn kind_str(k: crate::records::Kind) -> &'static str {
    match k {
        crate::records::Kind::User => "user",
        crate::records::Kind::Assistant => "assistant",
        crate::records::Kind::Tool => "tool",
    }
}

pub fn session_reader(
    records: &[Record],
    scored: &Scored,
    names: Option<&Names>,
    title: &str,
    out: &Path,
) -> Result<()> {
    let w = scored.width;
    let name_of = |f: usize| -> FeatureOut {
        let n = names.and_then(|n| n.names.iter().find(|x| x.feature == f));
        FeatureOut {
            id: f,
            name: n
                .and_then(|x| x.name.clone())
                .unwrap_or_else(|| format!("feature {f}")),
            desc: n.and_then(|x| x.description.clone()).unwrap_or_default(),
            stance: n.and_then(|x| x.stance.clone()).unwrap_or_default(),
            conf: n.and_then(|x| x.confidence).unwrap_or(0),
        }
    };
    let features: Vec<FeatureOut> = (0..w).map(name_of).collect();
    let n = records.len();
    let dense: Vec<Vec<f32>> = (0..n).map(|i| scored.dense_row(i)).collect();
    let mut mean = vec![0f64; w];
    for r in &dense {
        for (j, v) in r.iter().enumerate() {
            mean[j] += *v as f64;
        }
    }
    let mut order: Vec<usize> = (0..w).collect();
    order.sort_by(|a, b| mean[*b].partial_cmp(&mean[*a]).unwrap());
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, r) in records.iter().enumerate() {
        groups
            .entry(format!("{}:{}", r.agent, r.session))
            .or_default()
            .push(i);
    }
    let mut sessions: Vec<SessionOut> = groups
        .into_iter()
        .map(|(g, mut ks)| {
            ks.sort_by(|a, b| {
                records[*a]
                    .time
                    .cmp(&records[*b].time)
                    .then(records[*a].line.cmp(&records[*b].line))
            });
            let msgs = ks
                .iter()
                .map(|&i| {
                    let r = &records[i];
                    MsgOut {
                        line: r.line,
                        kind: kind_str(r.kind),
                        agent: &r.agent,
                        host: &r.host,
                        time: &r.time,
                        text: &r.text,
                        q: (0..w)
                            .map(|f| {
                                ((dense[i][f] / scored.scale[f]).clamp(0.0, 1.0) * 255.0).round()
                                    as u8
                            })
                            .collect(),
                        v: dense[i].iter().map(|x| (x * 10.0).round() / 10.0).collect(),
                    }
                })
                .collect::<Vec<_>>();
            let users = msgs.iter().filter(|m| m.kind == "user").count();
            SessionOut {
                host: &records[ks[0]].host,
                group: g,
                users,
                msgs,
            }
        })
        .collect();
    sessions.sort_by_key(|s| std::cmp::Reverse(s.msgs.len()));
    let data = serde_json::json!({"title": title, "width": w, "order": order, "features": features, "sessions": sessions, "threshold": SESSION_READER_TAG_THRESHOLD});
    let page = format!(
        "<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content=\"width=device-width,initial-scale=1\"><title>{}</title><style>{}</style></head><body><header><h1>{}</h1><div class=meta>Every session as a full transcript in time order. Each message carries the features active above the threshold (name and activation as a percentage of that feature's 99.5th-percentile scale). The heatmap above each session is the trajectory: one column per message (u user, a assistant, t tool), one row per feature; click a cell to jump. Click a feature in the panel to show only messages where it fires. Private: contains conversation text. Do not publish.</div><div class=ctl><label>tag threshold <select id=min><option value=35 selected>35% of scale</option><option value=20>20%</option><option value=50>50%</option><option value=70>70%</option></select></label><span id=selinfo class=st></span></div></header><main><div id=nav></div><div id=reader></div><div id=side><div id=pos class=st>hover a message</div><h3>features · by activation</h3><ul id=feat></ul></div></main><script>window.READER={};</script><script>{}</script></body></html>",
        html_escape(title),
        READER_CSS,
        html_escape(title),
        serde_json::to_string(&data)?.replace("</", "<\\/"),
        READER_JS
    );
    std::fs::write(out, page)?;
    Ok(())
}

pub fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

const READER_CSS: &str = r##"
:root{--paper:#f5f6f3;--ink:#212e2b;--muted:#61706a;--line:#dce2dc;--accent:#245c47;--soft:#e8efe9;--user:#8a4b1f;--usersoft:#f6ece3;--tool:#4b5a8a;--toolsoft:#eceef6;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.5 Arial,Helvetica,sans-serif}
header{padding:14px 24px 10px;border-bottom:1px solid var(--line)}h1{font-size:18px;margin:0 0 4px}.meta{color:var(--muted);font-size:12px}
main{display:grid;grid-template-columns:230px minmax(0,1fr) 300px}
#nav{position:sticky;top:0;height:100vh;overflow:auto;border-right:1px solid var(--line);background:#fff;padding:10px;font-size:12px}#nav a{display:block;padding:3px 6px;border-radius:3px;color:var(--ink);text-decoration:none}#nav a:hover{background:var(--soft)}
#reader{padding:10px 26px 80px;min-width:0}
#side{position:sticky;top:0;height:100vh;overflow:auto;border-left:1px solid var(--line);background:#fff;padding:12px 14px;font-size:12px}
.session{margin:26px 0 40px}.session h2{font-size:14px;margin:0 0 6px}.session h2 small{color:var(--muted);font-weight:400;margin-left:8px}
.timeline{display:grid;gap:1px 2px;margin:6px 0 14px;background:#fff;border:1px solid var(--line);padding:6px;width:max-content;max-width:100%;overflow-x:auto}
.timeline .lab{font-size:10px;color:var(--muted);white-space:nowrap;padding-right:6px;line-height:12px;height:12px}.timeline .c{width:14px;height:12px;border-radius:2px}
.timeline .hdr{font-size:9px;color:var(--muted);text-align:center;line-height:12px;height:12px}.timeline .c.cur{outline:2px solid #e08a00}
.msg{display:grid;grid-template-columns:150px minmax(0,1fr);gap:14px;margin:8px 0;padding:8px 10px;border-radius:8px;background:#fff;border:1px solid var(--line)}.msg.user{background:var(--usersoft)}.msg.tool{background:var(--toolsoft)}
.msg.cur{outline:2px solid #e08a00}
.who{font-size:11px;color:var(--muted)}.who b{display:block;font-size:12px}.who b.user{color:var(--user)}.who b.assistant{color:var(--accent)}.who b.tool{color:var(--tool)}
.tags{margin-top:6px;display:flex;flex-direction:column;gap:3px}.tag{font-size:11px;padding:1px 6px;border-radius:10px;background:var(--soft);color:var(--accent);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.msg.user .tag{background:#fff;color:var(--user)}.msg.tool .tag{background:#fff;color:var(--tool)}.tag i{font-style:normal;opacity:.7;font-family:var(--mono);font-size:10px}
.text{font:12.5px/1.5 var(--mono);white-space:pre-wrap;word-break:break-word}
.hide{display:none}.ctl{font-size:12px;color:var(--muted);margin-top:6px}.ctl label{margin-right:12px}
#feat{padding:0;margin:0}#feat li{list-style:none;padding:3px 4px;border-radius:3px;display:grid;grid-template-columns:22px 1fr 40px;gap:6px;align-items:center;cursor:pointer}#feat li.on{background:var(--soft)}#feat li.sel{outline:1px solid var(--accent)}#feat i{font-style:normal;color:var(--muted);font-family:var(--mono)}#feat .v{font-family:var(--mono);color:var(--accent);text-align:right}.st{color:var(--muted)}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;border:1px solid #9aa8a2;margin-left:4px}.dot.hi{background:#3fae5a}.dot.mid{background:#fff}.dot.lo{background:#f2d24b}
h3{font-size:12px;margin:14px 0 6px;color:var(--muted);font-weight:600;letter-spacing:.04em;text-transform:uppercase}
"##;

const READER_JS: &str = r##"
const D=window.READER,F=D.features,W=D.width,ORDER=D.order;let SEL=null,MIN=Math.round(D.threshold*255);
const reader=document.getElementById('reader'),nav=document.getElementById('nav'),feat=document.getElementById('feat');
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function col(q,kind){if(!q)return '#f1f3f0';const t=Math.sqrt(q/255);const base=kind==='user'?[138,75,31]:kind==='tool'?[75,90,138]:[36,92,71];return `rgb(${base.map(b=>Math.round(255-t*(255-b))).join(',')})`;}
function dot(c){return `<b class="dot ${c>=90?'hi':c>=80?'mid':'lo'}" title="confidence ${c}%"></b>`;}
function tags(m){return ORDER.filter(f=>m.q[f]>=MIN).sort((a,b)=>m.v[b]-m.v[a]).map(f=>`<span class=tag title="${esc(F[f].desc||'')}"><i>${f}</i> ${esc(F[f].name)} <i>${Math.round(m.q[f]/2.55)}%</i></span>`).join('')||'<span class="tag st">no strong feature</span>';}
function build(){let h='',n='';
 for(const [si,s] of D.sessions.entries()){const M=s.msgs;
  n+=`<a href="#s${si}">${s.group.slice(0,22)} <span class=st>· ${M.length} · ${s.host}</span></a>`;
  h+=`<div class=session id=s${si}><h2>session ${s.group}<small>${M.length} messages · ${s.users} user · ${s.host}</small></h2>`;
  h+=`<div class=timeline style="grid-template-columns:auto repeat(${M.length},14px)"><div class=hdr></div>${M.map(m=>`<div class=hdr title="message ${m.line} (${m.kind})">${m.kind[0]}</div>`).join('')}`;
  for(const f of ORDER){h+=`<div class=lab title="${esc(F[f].desc||'')}">${esc(F[f].name)}</div>${M.map((m,i)=>`<div class=c data-i=${i} data-s=${si} style="background:${col(m.q[f],m.kind)}" title="${esc(F[f].name)} · ${Math.round(m.q[f]/2.55)}% · message ${m.line}"></div>`).join('')}`;}
  h+='</div>';
  for(const [i,m] of M.entries()){h+=`<div class="msg ${m.kind}" data-s=${si} data-i=${i} id="m${si}_${i}"><div class=who><b class=${m.kind}>${m.kind}</b>#${m.line} · ${m.agent}${m.time?' · '+m.time.slice(0,16):''}<div class=tags>${tags(m)}</div></div><div class=text>${esc(m.text)}</div></div>`;}
  h+='</div>';}
 reader.innerHTML=h;nav.innerHTML=`<div class=st style="padding:4px 6px">${D.sessions.length} sessions, longest first</div>`+n;applySel();}
function applySel(){for(const el of reader.querySelectorAll('.msg')){const m=D.sessions[+el.dataset.s].msgs[+el.dataset.i];el.classList.toggle('hide',SEL!=null&&m.q[SEL]<MIN);}
 for(const li of feat.querySelectorAll('li'))li.classList.toggle('sel',+li.dataset.f===SEL);document.getElementById('selinfo').textContent=SEL==null?'':`showing only messages where "${F[SEL].name}" ≥ ${Math.round(MIN/2.55)}% (click again to clear)`;}
function renderFeat(m){const ids=ORDER.slice();if(m)ids.sort((a,b)=>m.v[b]-m.v[a]);
 feat.innerHTML=ids.map(f=>`<li data-f=${f} class="${m&&m.q[f]>=MIN?'on':''}${f===SEL?' sel':''}" title="${esc(F[f].desc||'')}&#10;stance: ${F[f].stance}"><i>${f}</i><span>${esc(F[f].name)} <span class=st>${F[f].stance||''}</span>${dot(F[f].conf||0)}</span><span class=v>${m?Math.round(m.q[f]/2.55)+'%':''}</span></li>`).join('');}
let cur=null;
reader.addEventListener('mousemove',e=>{const el=e.target.closest('.msg,.timeline .c');if(!el)return;const s=+el.dataset.s,i=+el.dataset.i;if(cur&&cur.s===s&&cur.i===i)return;
 if(cur){for(const x of reader.querySelectorAll(`[data-s="${cur.s}"][data-i="${cur.i}"]`))x.classList.remove('cur');}
 cur={s,i};for(const x of reader.querySelectorAll(`[data-s="${s}"][data-i="${i}"]`))x.classList.add('cur');
 const m=D.sessions[s].msgs[i];document.getElementById('pos').textContent=`message ${m.line} · ${m.kind}`;renderFeat(m);});
reader.addEventListener('click',e=>{const c=e.target.closest('.timeline .c');if(!c)return;document.getElementById(`m${c.dataset.s}_${c.dataset.i}`).scrollIntoView({block:'center',behavior:'smooth'});});
feat.addEventListener('click',e=>{const li=e.target.closest('li');if(!li)return;const f=+li.dataset.f;SEL=SEL===f?null:f;applySel();});
document.getElementById('min').addEventListener('change',e=>{MIN=Math.round(+e.target.value*2.55);build();renderFeat(null);});
build();renderFeat(null);
"##;
