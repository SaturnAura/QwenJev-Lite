"""User-facing web demo: a page with the three question types.

    python serve.py                 # http://127.0.0.1:8300
    python serve.py --port 8301 --model-dir models/qwenjev-multitask-v2

The page has one tab per question type - 判断 (yes/no), 单选 (choice), 档位 (score).
Everything is answered in a single forward pass per submission, and the result box shows
the distribution, the answer, the confidence and the timing. The JSON API behind it is
the same one the tests use (``POST /v1/decide``), so what the page shows is what the
benchmark measures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from qwenjev.api import create_app
from qwenjev.config import QwenJevConfig, default_model_path

PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>QwenJev-lite 决策台</title>
<style>
 body{font-family:system-ui,Segoe UI,Helvetica,Arial,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#1b1b1b}
 h1{font-size:20px;margin:0 0 4px} .sub{color:#666;font-size:13px;margin-bottom:18px}
 textarea,input{width:100%;box-sizing:border-box;padding:8px;border:1px solid #ccc;border-radius:6px;font-size:14px}
 textarea{min-height:96px} label{display:block;margin:12px 0 4px;font-weight:600;font-size:13px}
 .tabs{display:flex;gap:6px;margin:14px 0}
 .tabs button{padding:7px 14px;border:1px solid #ccc;background:#f6f6f6;border-radius:6px;cursor:pointer;font-size:14px}
 .tabs button.on{background:#111;color:#fff;border-color:#111}
 button.go{margin-top:14px;padding:10px 20px;border:0;background:#0b6;color:#fff;border-radius:6px;font-size:15px;cursor:pointer}
 pre{background:#f4f6f8;border:1px solid #e2e6ea;border-radius:6px;padding:12px;font-size:13px;white-space:pre-wrap}
 .bar{display:inline-block;height:12px;background:#0b6;vertical-align:middle;border-radius:2px}
</style></head><body>
<h1>QwenJev-lite 决策台</h1>
<div class="sub">一次前向回答全部问题，直接输出概率分布，不生成文本。readout：<b id="readout">…</b></div>

<label>文本（state）</label>
<textarea id="state">My payouts have failed three times. The bank says everything is fine. The customer has been waiting nine days and is threatening a chargeback.</textarea>

<div class="tabs">
  <button class="on" data-mode="bool">判断 yes/no</button>
  <button data-mode="choice">单选 choice</button>
  <button data-mode="score">档位 score</button>
</div>

<label id="q-label">要判断的结论（会作为断言提问）</label>
<input id="question" value="This message requires urgent human attention.">

<div id="options-wrap" style="display:none">
  <label>选项（每行一个，可写 <code>key=描述</code>）</label>
  <textarea id="options">payments=Payout failures and payment processing
account=Login and account access
other=Something else</textarea>
</div>

<button class="go" onclick="run()">运行</button>
<div id="out"></div>

<script>
let mode = 'bool';
document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => {
  document.querySelectorAll('.tabs button').forEach(x => x.classList.remove('on'));
  b.classList.add('on'); mode = b.dataset.mode;
  const isBool = mode === 'bool';
  document.getElementById('options-wrap').style.display = isBool ? 'none' : 'block';
  document.getElementById('q-label').textContent = isBool
    ? '要判断的结论（会作为断言提问）'
    : (mode === 'choice' ? '问题（选一个选项）' : '问题（给出有序档位，从低到高）');
  document.getElementById('question').value = isBool
    ? 'This message requires urgent human attention.'
    : (mode === 'choice' ? 'Which team should handle this ticket?' : 'How urgent is this ticket?');
  document.getElementById('options').value = mode === 'choice'
    ? 'payments=Payout failures and payment processing\\naccount=Login and account access\\nother=Something else'
    : 'No time pressure\\nNeeds attention this week\\nBlocking issue or hard deadline';
});

function buildQuestion() {
  const q = document.getElementById('question').value.trim();
  const lines = document.getElementById('options').value.split('\\n').map(s => s.trim()).filter(Boolean);
  if (mode === 'bool') {
    return {answer: {type:'bool', instructions:q, claim:q}};
  }
  if (mode === 'choice') {
    const criteria = {};
    for (const line of lines) { const [k, ...rest] = line.split('='); criteria[k.trim()] = (rest.join('=') || k).trim(); }
    return {answer: {type:'choice', instructions:q, criteria}};
  }
  const criteria = {};
  lines.forEach((level, i) => criteria['L' + i] = level);
  return {answer: {type:'score', instructions:q, criteria}};
}

async function run() {
  const out = document.getElementById('out');
  out.innerHTML = '<pre>running…</pre>';
  const body = {state: document.getElementById('state').value, questions: buildQuestion()};
  const t0 = performance.now();
  const r = await fetch('/v1/decide', {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(body)});
  const data = await r.json();
  const ms = performance.now() - t0;
  if (!r.ok) { out.innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>'; return; }
  const d = data.results.answer;
  let html = '';
  for (const [k, p] of Object.entries(d.probabilities)) {
    const w = Math.round(p * 260);
    html += `<div>${k.padEnd(10)} ${p.toFixed(3)} <span class="bar" style="width:${w}px"></span></div>`;
  }
  html += `<div style="margin-top:8px">answer=<b>${d.answer}</b> confidence=${d.confidence.toFixed(3)}`;
  if (d.score !== undefined) html += ` score=${d.score}`;
  html += `</div>`;
  html += `<div style="color:#666;font-size:12px;margin-top:6px">${data.usage.input_tokens} input tokens · ${data.usage.questions} question(s) · 浏览器端 ${ms.toFixed(0)} ms</div>`;
  out.innerHTML = '<pre>' + html + '</pre>';
}
</script></body></html>
"""


def build(config: QwenJevConfig) -> FastAPI:
    app = create_app(config=config)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    return app


def main() -> int:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--model", default=None, help="backbone path or Hub repo id (default $QWENJEV_MODEL)")
    parser.add_argument("--model-dir", default="models/qwenjev-multitask-v2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrained-readout", action="store_true",
                        help="use the readout the backbone came with instead of readout.pt")
    args = parser.parse_args()
    args.model = args.model or default_model_path()

    checkpoint = Path(args.model_dir) / "readout.pt"
    use_trained = checkpoint.is_file() and not args.pretrained_readout
    config = QwenJevConfig(
        model_path=args.model,
        device=args.device,
        readout="slot_head" if use_trained else "reserved_label",
        readout_checkpoint=str(checkpoint) if use_trained else None,
    )
    print(f"http://{args.host}:{args.port}  (readout={'trained ' + str(checkpoint) if use_trained else 'pretrained'})")
    uvicorn.run(build(config), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
