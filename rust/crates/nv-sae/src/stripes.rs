use anyhow::Result;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;

use crate::capture::Span;
use crate::export::html_escape;
use crate::name::Names;
use crate::records::Record;
use crate::score::Scored;

pub const STRIPE_CLUSTER_SAMPLE_TOKENS: usize = 60_000;

#[derive(Serialize)]
struct Feat {
    id: usize,
    row: usize,
    lname: Option<String>,
    conf: Option<u32>,
    ldesc: Option<String>,
    trigger: Option<String>,
    u: f32,
    a: f32,
}

#[derive(Serialize)]
struct Rec {
    k: usize,
    line: usize,
    group: String,
    role: String,
    host: String,
    tokens: Vec<String>,
    top: Vec<Vec<(u32, u8, f32)>>,
}

pub fn cluster_order(rows: &[Vec<f32>], width: usize) -> Vec<usize> {
    let n = rows.len().max(1) as f64;
    let mut mean = vec![0f64; width];
    for r in rows {
        for (j, v) in r.iter().enumerate() {
            mean[j] += *v as f64 / n;
        }
    }
    let mut cov = vec![vec![0f64; width]; width];
    for r in rows {
        let d: Vec<f64> = r
            .iter()
            .enumerate()
            .map(|(j, v)| *v as f64 - mean[j])
            .collect();
        for i in 0..width {
            if d[i] == 0.0 {
                continue;
            }
            for j in i..width {
                cov[i][j] += d[i] * d[j];
            }
        }
    }
    let sd: Vec<f64> = (0..width).map(|i| cov[i][i].sqrt().max(1e-9)).collect();
    let dist = |i: usize, j: usize| -> f64 {
        let (a, b) = if i <= j { (i, j) } else { (j, i) };
        1.0 - cov[a][b] / (sd[a] * sd[b])
    };
    let mut clusters: Vec<Vec<usize>> = (0..width).map(|i| vec![i]).collect();
    while clusters.len() > 1 {
        let (mut bi, mut bj, mut best) = (0, 1, f64::MAX);
        for i in 0..clusters.len() {
            for j in i + 1..clusters.len() {
                let mut s = 0.0;
                for &a in &clusters[i] {
                    for &b in &clusters[j] {
                        s += dist(a, b);
                    }
                }
                let d = s / (clusters[i].len() * clusters[j].len()) as f64;
                if d < best {
                    best = d;
                    bi = i;
                    bj = j;
                }
            }
        }
        let mut right = clusters.remove(bj);
        let mut left = clusters.remove(bi);
        let link = |l: &[usize], r: &[usize]| dist(*l.last().unwrap(), r[0]);
        if link(&left, &right) > link(&left.iter().rev().copied().collect::<Vec<_>>(), &right) {
            left.reverse();
        }
        if link(&left, &right) > link(&left, &right.iter().rev().copied().collect::<Vec<_>>()) {
            right.reverse();
        }
        left.extend(right);
        clusters.push(left);
    }
    clusters.pop().unwrap_or_default()
}

pub fn detok(tokenizer: &tokenizers::Tokenizer, id: u32) -> String {
    tokenizer
        .decode(&[id], false)
        .unwrap_or_else(|_| "?".into())
}

pub fn stripes(
    records: &[Record],
    spans: &[Span],
    scored: &Scored,
    names: Option<&Names>,
    tokenizer: &tokenizers::Tokenizer,
    title: &str,
    out: &Path,
) -> Result<()> {
    let w = scored.width;
    let total = scored.indices.len();
    let sample_step = (total / STRIPE_CLUSTER_SAMPLE_TOKENS).max(1);
    let sample: Vec<Vec<f32>> = (0..total)
        .step_by(sample_step)
        .map(|i| scored.dense_row(i))
        .collect();
    let order = cluster_order(&sample, w);
    let mut row_of = vec![0usize; w];
    for (r, f) in order.iter().enumerate() {
        row_of[*f] = r;
    }
    let mut active_by_role: BTreeMap<String, (f64, Vec<f64>)> = BTreeMap::new();
    let mut recs = Vec::new();
    for sp in spans {
        let r = &records[sp.record];
        let role = format!("{:?}", r.kind).to_lowercase();
        let entry = active_by_role
            .entry(role.clone())
            .or_insert_with(|| (0.0, vec![0.0; w]));
        let mut top = Vec::with_capacity(sp.end - sp.start);
        for t in sp.start..sp.end {
            entry.0 += 1.0;
            let mut row: Vec<(u32, u8, f32)> = scored.indices[t]
                .iter()
                .zip(&scored.values[t])
                .map(|(f, v)| {
                    (
                        *f,
                        ((v / scored.scale[*f as usize]).clamp(0.0, 1.0) * 255.0).round() as u8,
                        (v * 100.0).round() / 100.0,
                    )
                })
                .collect();
            for (f, _, _) in &row {
                entry.1[*f as usize] += 1.0;
            }
            row.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
            top.push(row);
        }
        recs.push(Rec {
            k: sp.record,
            line: r.line,
            group: r.session.clone(),
            role,
            host: r.host.clone(),
            tokens: sp.ids.iter().map(|&i| detok(tokenizer, i)).collect(),
            top,
        });
    }
    let rate = |role: &str, f: usize| -> f32 {
        active_by_role
            .get(role)
            .map(|(n, c)| {
                if *n > 0.0 {
                    (c[f] / n * 10000.0).round() as f32 / 100.0
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0)
    };
    let feats: Vec<Feat> = (0..w)
        .map(|f| {
            let n = names.and_then(|n| n.names.iter().find(|x| x.feature == f));
            Feat {
                id: f,
                row: row_of[f],
                lname: n.and_then(|x| x.name.clone()),
                conf: n.and_then(|x| x.confidence),
                ldesc: n.and_then(|x| x.description.clone()),
                trigger: n.and_then(|x| x.stance.clone()),
                u: rate("user", f),
                a: rate("assistant", f),
            }
        })
        .collect();
    let mut sessions: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, r) in recs.iter().enumerate() {
        sessions.entry(r.group.clone()).or_default().push(i);
    }
    let sessions: Vec<serde_json::Value> = sessions
        .into_iter()
        .map(|(g, ks)| serde_json::json!({"group": g, "records": ks}))
        .collect();
    let named = feats.iter().filter(|f| f.lname.is_some()).count();
    let data = serde_json::json!({"run": title, "arm": "layer-19", "order": order, "features": feats, "records": recs, "sessions": sessions, "counts": {"named": named}});
    let page = format!(
        "<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content=\"width=device-width,initial-scale=1\"><title>{}</title><style>{}</style></head><body><header><h1>{}</h1><div class=meta>Each token is a column: its text on top, the {} features beneath it in one fixed order (average-linkage clustering of co-activation), colour depth = activation against that feature's 99.5th-percentile scale. One line per message; hover any column for the token, hover a pixel row for its feature, and the panel re-sorts by activation. Names come from Laguna; the dot after a name is its confidence (green 90% or more, white 80 to 89, yellow below 80). U/A is the share of user / assistant tokens on which the feature is active. Private: contains conversation text. Do not publish.</div><div class=ctl><label>messages <select id=role><option value=all>all</option><option value=user>user only</option><option value=assistant>assistant only</option><option value=tool>tool only</option></select></label><label>row height <select id=rowh><option value=1>1 px</option><option value=2>2 px</option></select></label></div></header><main><div id=list></div><div id=side><div class=rh>token</div><div id=tok>—</div><div id=pos class=st>hover a column</div><h2>features · active first, by activation</h2><ul id=feat></ul></div></main><script>window.STRIPES={};</script><script>{}</script></body></html>",
        html_escape(title), STRIPES_CSS, html_escape(title), w, serde_json::to_string(&data)?.replace("</", "<\\/"), STRIPES_JS
    );
    std::fs::write(out, page)?;
    Ok(())
}

const STRIPES_CSS: &str = r##"
:root{--paper:#f5f6f3;--ink:#212e2b;--muted:#61706a;--line:#dce2dc;--accent:#245c47;--soft:#e8efe9;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 Arial,Helvetica,sans-serif}
header{padding:16px 24px 10px;border-bottom:1px solid var(--line)}h1{font-size:18px;margin:0 0 4px}h2{font-size:12px;margin:18px 0 6px;color:var(--muted);font-weight:600;letter-spacing:.04em;text-transform:uppercase}
.meta{color:var(--muted);font-size:12px}.ctl{font-size:12px;color:var(--muted);margin-top:6px}.ctl label{margin-right:14px}
main{display:grid;grid-template-columns:minmax(0,1fr) 320px}
#list{padding:8px 24px 80px;min-width:0}#side{position:sticky;top:0;height:100vh;overflow:auto;border-left:1px solid var(--line);background:#fff;padding:12px 14px;font-size:12px}
.session{margin:22px 0 4px;padding-top:8px;border-top:1px solid var(--line);font-size:12px;color:var(--muted)}
.rec{margin:8px 0 14px}.rh{font-size:11px;color:var(--muted);margin:0 0 3px}.rh b{color:var(--ink)}
canvas{display:block;background:#fff;border:1px solid var(--line);margin-bottom:6px}
.scroll{overflow-x:auto;overflow-y:hidden;max-width:100%}.hide{display:none}
.role{font-weight:600}.role.user{color:#8a4b1f}.role.assistant{color:var(--accent)}.role.tool{color:#4b5a8a}
#feat{list-style:none;padding:0;margin:0}#feat li{padding:2px 4px;border-radius:3px;display:grid;grid-template-columns:30px minmax(0,1fr) 12px 92px 58px;gap:0 6px;align-items:center}#feat li.on{background:var(--soft)}#feat li.cur{outline:1px solid var(--accent)}
#feat li i{font-style:normal;color:var(--muted);font-family:var(--mono)}#feat li .v{font-family:var(--mono);color:var(--accent);text-align:right;white-space:nowrap}#feat li .v small{display:block;font-size:10px}#feat li .ua{text-align:right;font-family:var(--mono);color:var(--muted);font-size:10px;white-space:nowrap}
.st{color:var(--muted)}.tr{color:var(--accent)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;border:1px solid #9aa8a2;flex:none;align-self:center}.dot.hi{background:#3fae5a;border-color:#2f8a46}.dot.mid{background:#fff}.dot.lo{background:#f2d24b;border-color:#c9a800}
#tok{font-family:var(--mono);font-size:13px;white-space:pre-wrap;word-break:break-all;min-height:18px}
"##;

const STRIPES_JS: &str = r##"
const D=window.STRIPES,F=D.features,ORDER=D.order,N=F.length;
const list=document.getElementById('list');
const FONT='12px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace',TH=18,GAP=3,PAD=2,MINW=6,MAXW=30000;
let ROWH=1;
const mc=document.createElement('canvas').getContext('2d');mc.font=FONT;
function label(t){return t.replace(/\n/g,'⏎').replace(/\t/g,'→').replace(/ /g,'␣');}
function fname(f){const x=F[f];return x.lname||('feature '+f);}
function fconf(f){const c=F[f].conf;if(c==null)return '';const k=c>=90?'hi':c>=80?'mid':'lo';return `<b class="dot ${k}" title="confidence ${c}%"></b>`;}
function ftitle(f){const x=F[f];return [x.ldesc,x.trigger?'stance: '+x.trigger:null].filter(Boolean).join('\n');}
function layout(rec){let ws=rec.tokens.map(t=>Math.max(MINW,Math.ceil(mc.measureText(label(t)).width)+2*PAD));const total=ws.reduce((s,w)=>s+w,0);
 if(total>MAXW){const k=MAXW/total;ws=ws.map(w=>Math.max(2,Math.floor(w*k)));}
 const cur=[];let x=0;for(let i=0;i<ws.length;i++){cur.push({i,x,w:ws[i]});x+=ws[i];}return cur;}
function draw(rec,cv){const ln=layout(rec);const bandH=TH+N*ROWH+GAP;const W=ln.length?ln[ln.length-1].x+ln[ln.length-1].w:10;cv.width=W;cv.height=bandH;const g=cv.getContext('2d');g.font=FONT;g.textBaseline='middle';cv._ln=ln;
 for(const c of ln){const t=rec.tokens[c.i];g.fillStyle=t.trim()?'#212e2b':'#9aa8a2';g.fillText(label(t),c.x+PAD,TH/2);
  for(const [f,q] of rec.top[c.i]){const v=Math.sqrt(q/255);g.fillStyle=`rgb(${Math.round(255-v*(255-36))},${Math.round(255-v*(255-92))},${Math.round(255-v*(255-71))})`;g.fillRect(c.x,TH+F[f].row*ROWH,c.w,ROWH);}
  g.fillStyle='rgba(0,0,0,.06)';g.fillRect(c.x+c.w-1,TH,1,N*ROWH);}
 g.fillStyle='#dce2dc';g.fillRect(0,TH-1,cv.width,1);}
function build(){let h='';for(const s of D.sessions){h+=`<div class=session>session ${s.group} · ${s.records.length} records</div>`;
 for(const k of s.records){const r=D.records[k];h+=`<div class="rec role-${r.role}" id=r${k}><div class=rh><b>${r.line}</b> <span class="role ${r.role}">${r.role}</span> · ${r.host} · ${r.tokens.length} tokens</div><div class=scroll><canvas data-k=${k}></canvas></div></div>`;}}
 list.innerHTML=h;applyRole();
 const io=new IntersectionObserver(es=>{for(const e of es)if(e.isIntersecting&&!e.target._drawn){e.target._drawn=1;draw(D.records[+e.target.dataset.k],e.target);}},{rootMargin:'800px'});
 for(const cv of list.querySelectorAll('canvas')){cv.height=TH+N*ROWH+GAP;cv.width=600;io.observe(cv);}}
function applyRole(){const v=document.getElementById('role').value;for(const el of list.querySelectorAll('.rec'))el.classList.toggle('hide',v!=='all'&&!el.classList.contains('role-'+v));}
document.getElementById('role').addEventListener('change',applyRole);
build();
const feat=document.getElementById('feat');
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function renderFeat(active){const act=new Map(active.map(([f,q,v])=>[f,[q,v]]));const ids=[...Array(N).keys()];
 ids.sort((a,b)=>(act.has(b)-act.has(a))||(act.has(a)?act.get(b)[1]-act.get(a)[1]:F[a].row-F[b].row));
 feat.innerHTML=ids.map(f=>`<li data-f=${f}${act.has(f)?' class=on':''} title="${esc(ftitle(f)).replace(/"/g,'&quot;')}"><i>${f}</i><span class="${F[f].lname?'tr':'st'}">${esc(fname(f))}</span><span class=st>${fconf(f)}</span><span class=ua title="share of tokens where active: user / assistant">U${F[f].u}% A${F[f].a}%</span>${act.has(f)?`<span class=v>${Math.round(act.get(f)[0]*100/255)}% <small class=st>${act.get(f)[1]}</small></span>`:''}</li>`).join('');}
renderFeat([]);
let mark=null;
list.addEventListener('mousemove',e=>{const cv=e.target.closest('canvas');if(!cv||!cv._ln)return;const rect=cv.getBoundingClientRect();const x=e.clientX-rect.left,y=e.clientY-rect.top;
 const c=cv._ln.find(c=>x>=c.x&&x<c.x+c.w);if(!c)return;const rec=D.records[+cv.dataset.k];const row=y>=TH?Math.floor((y-TH)/ROWH):-1;
 document.getElementById('tok').textContent=JSON.stringify(rec.tokens[c.i]);
 document.getElementById('pos').textContent=`token ${c.i+1} / ${rec.tokens.length} · record ${rec.line} · ${rec.role}@${rec.host}`+(row>=0&&row<N?` · row ${row}: ${fname(ORDER[row])}`:'');
 renderFeat(rec.top[c.i]);
 if(row>=0&&row<N){const li2=feat.querySelector(`li[data-f="${ORDER[row]}"]`);if(li2)li2.classList.add('cur');}
 if(mark)mark.cv.getContext('2d').putImageData(mark.img,mark.x,0);
 const g=cv.getContext('2d');mark={cv,x:c.x,img:g.getImageData(c.x,0,c.w,cv.height)};g.fillStyle='rgba(255,170,60,.28)';g.fillRect(c.x,0,c.w,cv.height);});
document.getElementById('rowh').addEventListener('change',e=>{ROWH=+e.target.value;build();});
"##;
