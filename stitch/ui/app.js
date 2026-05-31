// ═══════════════════════════════════════════════════════════════════════════
//  ContextFree-X  —  Stitch UI  —  app.js
//  Connects the HTML frontend to the FastAPI bridge on :8502
// ═══════════════════════════════════════════════════════════════════════════

const API_BASE = "http://localhost:8502";

// ── Navigation ──────────────────────────────────────────────────────────────
function navigate(page) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.sb-nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + page).classList.add('active');
  document.querySelector(`[data-page="${page}"]`).classList.add('active');
}

// ── Toast notifications ──────────────────────────────────────────────────────
function showToast(message, type = 'info') {
  const existing = document.getElementById('toast');
  if (existing) existing.remove();
  const colors = { success: '#065f46', error: '#ba1a1a', info: '#0058be' };
  const bg     = { success: '#ecfdf5', error: '#fff1f2', info: '#eff6ff' };
  const toast  = document.createElement('div');
  toast.id = 'toast';
  toast.style.cssText = `
    position:fixed;bottom:24px;right:24px;z-index:9999;
    background:${bg[type]||bg.info};color:${colors[type]||colors.info};
    border:1px solid ${colors[type]||colors.info}33;
    border-radius:12px;padding:12px 20px;max-width:360px;
    font-family:'Manrope',sans-serif;font-size:0.85rem;font-weight:700;
    box-shadow:0 4px 20px rgba(0,0,0,0.12);
    animation:slideIn 0.25s ease;
  `;
  toast.textContent = message;
  const style = document.createElement('style');
  style.textContent = '@keyframes slideIn{from{transform:translateY(20px);opacity:0}to{transform:translateY(0);opacity:1}}';
  document.head.appendChild(style);
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

function copyText(btn) {
  const card = btn.closest('.output-card, #bp-result, #ml-output');
  const texts = card?.querySelectorAll('.output-text, .tree-node');
  const content = texts ? Array.from(texts).map(t => t.textContent).join('\n') : '';
  navigator.clipboard.writeText(content).then(() => showToast('Copied to clipboard!', 'success'));
}

// ── Generic drag helpers ─────────────────────────────────────────────────────
function dragOver(e, zoneId) {
  e.preventDefault();
  document.getElementById(zoneId)?.classList.add('drag-over');
}
function dragLeave(zoneId) {
  document.getElementById(zoneId)?.classList.remove('drag-over');
}

// ═══════════════════════════════════════════════════════════════════════════
//  PAGE: SUMMARIZE
// ═══════════════════════════════════════════════════════════════════════════

const INPUT_TABS = ['text', 'pdf', 'url', 'youtube', 'arxiv'];
let currentInputType = 'text';
let activePdfFile = null;

function switchInputTab(type, btn) {
  btn.closest('.tab-group').querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  INPUT_TABS.forEach(t => {
    const el = document.getElementById('input-panel-' + t);
    if (el) el.style.display = (t === type) ? 'block' : 'none';
  });
  currentInputType = type;
  updateStatusBar();
}

function switchTab(groupId, btn) {
  const group = document.getElementById(groupId);
  group?.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
}

// Word counter
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('input-text')?.addEventListener('input', function () {
    const words = this.value.trim() ? this.value.trim().split(/\s+/).length : 0;
    document.getElementById('word-count').textContent = `${words} words`;
    updateStatusBar();
  });
  document.getElementById('url-input')?.addEventListener('blur', fetchUrlPreview);
  document.getElementById('yt-input')?.addEventListener('input', previewYouTube);
  loadModelOptions();
});

// Status bar chips
function updateStatusBar() {
  const sourceEl = document.getElementById('status-source');
  if (sourceEl) {
    const labels = { text:'TEXT', pdf:'PDF', url:'URL', youtube:'YOUTUBE', arxiv:'ARXIV' };
    sourceEl.textContent = `SOURCE: ${labels[currentInputType] || 'TEXT'}`;
  }
}

// ── PDF handlers ─────────────────────────────────────────────────────────────
function dropPdf(e) {
  e.preventDefault();
  dragLeave('pdf-dropzone');
  const file = e.dataTransfer.files[0];
  if (file && file.type === 'application/pdf') pdfFileReady(file);
  else showToast('Please drop a PDF file.', 'error');
}
function pdfFileSelected(input) {
  if (input.files[0]) pdfFileReady(input.files[0]);
}
function pdfFileReady(file) {
  activePdfFile = file;
  document.getElementById('pdf-dropzone').style.display = 'none';
  const info = document.getElementById('pdf-file-info');
  info.style.display = 'flex';
  document.getElementById('pdf-file-name').textContent =
    file.name + ' · ' + (file.size / 1024).toFixed(1) + ' KB';
  showToast('PDF ready: ' + file.name, 'success');
}
function clearPdf() {
  activePdfFile = null;
  document.getElementById('pdf-dropzone').style.display = 'block';
  document.getElementById('pdf-file-info').style.display = 'none';
  document.getElementById('pdf-file-input').value = '';
}

// ── URL preview ──────────────────────────────────────────────────────────────
async function fetchUrlPreview() {
  const url = document.getElementById('url-input').value.trim();
  if (!url || !url.startsWith('http')) return;
  const preview     = document.getElementById('url-preview');
  const previewText = document.getElementById('url-preview-text');
  preview.style.display = 'block';
  previewText.textContent = 'Fetching article…';
  try {
    const res = await fetch(`${API_BASE}/api/fetch/url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, model: 'bart-cnn' }),
      signal: AbortSignal.timeout(15000),
    });
    if (res.ok) {
      const d = await res.json();
      previewText.textContent =
        `"${d.title || url}" — ${d.word_count.toLocaleString()} words extracted`;
    } else {
      previewText.textContent = 'Preview unavailable — will attempt on summarize.';
    }
  } catch {
    previewText.textContent = 'Could not reach article — will attempt on summarize.';
  }
}

// ── YouTube preview ───────────────────────────────────────────────────────────
function previewYouTube() {
  const url   = document.getElementById('yt-input').value.trim();
  const match = url.match(/(?:v=|youtu\.be\/)([a-zA-Z0-9_-]{11})/);
  const preview = document.getElementById('yt-preview');
  if (match) {
    const vid = match[1];
    document.getElementById('yt-thumb').src  = `https://img.youtube.com/vi/${vid}/mqdefault.jpg`;
    document.getElementById('yt-title').textContent = `Video detected (ID: ${vid})`;
    preview.style.display = 'block';
  } else {
    preview.style.display = 'none';
  }
}

// ── Populate model <select> from API ─────────────────────────────────────────
async function loadModelOptions() {
  try {
    const res = await fetch(`${API_BASE}/api/models`, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) return;
    const models = await res.json();
    const nonMT5 = models.filter(m => m.key !== 'mt5');
    ['#page-summarize .select-input',
     '#compare-model-select',
     '#metrics-model-select',
     '#batch-model-select',
    ].forEach(sel => {
      const el = document.querySelector(sel);
      if (!el) return;
      const isMulti = el.hasAttribute('multiple');
      const prev = isMulti ? Array.from(el.selectedOptions).map(o => o.value) : [el.value];
      el.innerHTML = nonMT5.map(m =>
        `<option value="${m.key}"${prev.includes(m.key) ? ' selected' : ''}>${m.label}</option>`
      ).join('');
    });
    // Update status bar
    const modelEl = document.getElementById('status-model');
    if (modelEl && nonMT5.length) {
      modelEl.textContent = `MODEL: ${nonMT5[0].label.toUpperCase()}`;
      document.querySelector('#page-summarize .select-input')?.addEventListener('change', e => {
        modelEl.textContent = `MODEL: ${e.target.options[e.target.selectedIndex].text.toUpperCase()}`;
      });
    }
  } catch { /* API not running yet, use HTML defaults */ }
}

// ─────────────────────────────────────────────────────────────────────────────
//  SUMMARIZE  (all input types)
// ─────────────────────────────────────────────────────────────────────────────
async function runSummarize() {
  const model  = document.querySelector('#page-summarize .select-input')?.value || 'bart-cnn';
  const btn    = document.querySelector('#page-summarize .btn-primary');
  const origHtml = btn.innerHTML;
  btn.innerHTML  = '<span class="spinner"></span> Running…';
  btn.disabled   = true;

  document.getElementById('output-placeholder').style.display = 'flex';
  document.getElementById('output-body').style.display        = 'none';

  try {
    let data;

    // ── PDF ──
    if (currentInputType === 'pdf') {
      if (!activePdfFile) { showToast('Please upload a PDF first.', 'error'); return; }
      const form  = new FormData();
      form.append('file',  activePdfFile);
      form.append('model', model);
      // pick section if research-paper metadata arrived on earlier call
      form.append('section', window._pdfSection || 'full_text');
      const res = await apiFetch('/api/summarize/pdf', { method: 'POST', body: form });
      data = await res.json();
      // Show section selector if research paper
      if (data.is_research_paper && data.sections_available) {
        showPdfSectionPicker(data.sections_available, model);
      }

    // ── URL ──
    } else if (currentInputType === 'url') {
      const url = document.getElementById('url-input').value.trim();
      if (!url) { showToast('Please enter a URL.', 'error'); return; }
      const res = await apiFetch('/api/summarize/url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, model }),
      });
      data = await res.json();

    // ── YouTube ──
    } else if (currentInputType === 'youtube') {
      const url = document.getElementById('yt-input').value.trim();
      if (!url) { showToast('Please enter a YouTube URL.', 'error'); return; }
      const res = await apiFetch('/api/summarize/youtube', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, model }),
      });
      data = await res.json();
      if (data.transcript_words)
        showToast(`Transcript: ${data.transcript_words.toLocaleString()} words · ${Math.floor((data.duration_seconds||0)/60)}m`, 'info');

    // ── arXiv ──
    } else if (currentInputType === 'arxiv') {
      const query = document.getElementById('arxiv-input')?.value.trim();
      if (!query) { showToast('Please enter an arXiv ID or URL.', 'error'); return; }
      const source = document.getElementById('arxiv-source')?.value || 'abstract';
      const res = await apiFetch('/api/summarize/arxiv', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, model, source }),
      });
      data = await res.json();
      if (data.title) showToast(`Paper: "${data.title.substring(0,60)}…"`, 'info');

    // ── Plain text ──
    } else {
      const text = document.getElementById('input-text').value.trim();
      if (!text) { showToast('Please enter some text.', 'error'); return; }
      const res = await apiFetch('/api/summarize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, model }),
      });
      data = await res.json();
    }

    displaySummary(data.summary, model, data.metrics, data.word_count);

  } catch (e) {
    showToast('Error: ' + e.message, 'error');
    document.getElementById('output-placeholder').style.display = 'flex';
  } finally {
    btn.innerHTML = origHtml;
    btn.disabled  = false;
  }
}

function displaySummary(summary, model, metrics, wordCount) {
  document.getElementById('output-placeholder').style.display = 'none';
  const body = document.getElementById('output-body');
  body.style.display = 'block';
  const outputCard = document.getElementById('output-card');
  outputCard.classList.add('has-content');
  outputCard.style.cssText += 'align-items:flex-start;justify-content:flex-start;';
  document.getElementById('output-text').textContent = summary;

  // Metrics strip — map API keys to friendly labels
  const strip  = document.getElementById('metrics-strip');
  const LABELS = {
    'ROUGE-1':'R1','ROUGE-2':'R2','ROUGE-L':'RL',
    'BERTScore_F1':'BERTScore','BLEU':'BLEU','METEOR':'METEOR',
    'abstractiveness':'Abstr.','compression':'Compr.','composite':'Composite',
    // camelCase aliases
    rouge1:'R1', rouge2:'R2', rougeL:'RL', bertscore:'BERTScore',
  };
  if (metrics) {
    strip.innerHTML = Object.entries(metrics)
      .filter(([, v]) => typeof v === 'number')
      .map(([k, v]) =>
        `<span class="meta-chip"><strong style="color:#0058be;">${LABELS[k]||k}</strong> ${typeof v==='number'?v.toFixed(3):v}</span>`
      ).join('');
  }
  if (wordCount) {
    strip.innerHTML += `<span class="meta-chip" style="margin-left:auto;"><strong style="color:#065f46;">Words</strong> ${wordCount}</span>`;
  }
}

function showPdfSectionPicker(sections, model) {
  const panel = document.getElementById('input-panel-pdf');
  if (!panel || document.getElementById('pdf-section-picker')) return;
  const div = document.createElement('div');
  div.id = 'pdf-section-picker';
  div.style.cssText = 'margin-top:12px;display:flex;gap:8px;flex-wrap:wrap;';
  const labels = { full_text:'Full Text', abstract:'Abstract', introduction:'Introduction', conclusion:'Conclusion' };
  sections.filter(s => s !== 'full_text' || true).forEach(s => {
    const btn = document.createElement('button');
    btn.className = 'tab' + (s === 'full_text' ? ' active' : '');
    btn.textContent = labels[s] || s;
    btn.onclick = () => {
      div.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      window._pdfSection = s;
    };
    div.appendChild(btn);
  });
  panel.appendChild(div);
  window._pdfSection = 'full_text';
}

// ═══════════════════════════════════════════════════════════════════════════
//  PAGE: COMPARE MODELS
// ═══════════════════════════════════════════════════════════════════════════

async function runCompare() {
  const text = document.getElementById('compare-input')?.value.trim();
  if (!text) { showToast('Please enter text to compare.', 'error'); return; }

  // Collect selected models from multi-select if present, else use defaults
  const sel    = document.getElementById('compare-model-select');
  const models = sel
    ? Array.from(sel.selectedOptions).map(o => o.value)
    : [];

  const grid = document.getElementById('model-grid');
  grid.innerHTML = '<div style="padding:40px;text-align:center;width:100%;"><span class="spinner" style="width:32px;height:32px;border-width:3px;"></span><p style="margin-top:12px;color:#727785;">Running models in parallel…</p></div>';

  try {
    const refInput = document.getElementById('compare-ref')?.value.trim();
    const res  = await apiFetch('/api/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, models, reference: refInput || null }),
    });
    const data = await res.json();

    // Recommendation banner
    const recEl = document.getElementById('compare-recommendation');
    if (recEl) recEl.textContent = data.recommendation || '';

    // Render model cards
    grid.innerHTML = Object.entries(data.results).map(([mk, r]) => {
      const isBest  = mk === data.best_model;
      const m       = r.metrics;
      const badges  = [
        ['Composite', m.composite],
        ['BERTScore', m['BERTScore_F1']],
        ['ROUGE-1',   m['ROUGE-1']],
        ['Abstr.',    m.abstractiveness],
        ['Compr.',    m.compression],
      ].filter(([,v]) => v != null).map(([label, v]) =>
        `<span class="meta-chip"><strong style="color:${r.color};">${label}</strong> ${(+v).toFixed(3)}</span>`
      ).join('');
      return `
        <div class="model-card${isBest ? ' best-model' : ''}">
          <div class="model-card-accent" style="background:${r.color}"></div>
          ${isBest ? '<span class="best-badge">🏆 Best Model</span>' : ''}
          <div class="model-card-name">${r.label}</div>
          <p class="model-card-text">${r.summary}</p>
          <div style="font-size:0.78rem;color:#727785;margin-bottom:8px;">${r.words} words</div>
          <div style="display:flex;flex-wrap:wrap;gap:6px;">${badges}</div>
        </div>`;
    }).join('');

  } catch (e) {
    grid.innerHTML = `<p style="color:#ba1a1a;padding:20px;">Error: ${e.message}</p>`;
    showToast('Compare failed: ' + e.message, 'error');
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  PAGE: METRICS DASHBOARD
// ═══════════════════════════════════════════════════════════════════════════

async function runMetrics() {
  const text = document.getElementById('metrics-input')?.value.trim();
  if (!text) { showToast('Please enter text to evaluate.', 'error'); return; }

  const sel    = document.getElementById('metrics-model-select');
  const models = sel ? Array.from(sel.selectedOptions).map(o => o.value) : [];
  const ref    = document.getElementById('metrics-ref')?.value.trim() || null;
  const btn    = document.getElementById('metrics-run-btn');
  if (btn) { btn.disabled = true; btn.textContent = 'Evaluating…'; }

  const container = document.getElementById('metrics-results');
  if (container) {
    container.style.display = 'block';
    container.innerHTML = '<div style="padding:40px;text-align:center;"><span class="spinner" style="width:32px;height:32px;border-width:3px;"></span><p style="margin-top:12px;color:#727785;">Computing metrics for all models…</p></div>';
  }

  try {
    const res  = await apiFetch('/api/metrics', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, models, reference: ref }),
    });
    const data = await res.json();
    renderMetricsResults(data);
  } catch (e) {
    if (container) container.innerHTML = `<p style="color:#ba1a1a;padding:20px;">Error: ${e.message}</p>`;
    showToast('Metrics failed: ' + e.message, 'error');
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'Run Full Evaluation'; }
  }
}

function renderMetricsResults(data) {
  const container = document.getElementById('metrics-results');
  if (!container) return;

  const { scores, summaries, novel_metrics, ranking, recommendation, model_meta } = data;
  const METRIC_KEYS = ['ROUGE-1','ROUGE-2','ROUGE-L','BLEU','BERTScore_F1','METEOR','composite'];

  // Recommendation
  const recHtml = recommendation
    ? `<div class="info-banner" style="background:#eff6ff;border-left:4px solid #0058be;padding:12px 16px;border-radius:8px;margin-bottom:20px;font-size:0.88rem;color:#1e3a5f;">${recommendation}</div>`
    : '';

  // Ranking list
  const rankHtml = ranking.length
    ? `<div style="margin-bottom:20px;"><div class="card-label">Composite Ranking</div>
        ${ranking.map(([mk, score], i) => `
          <div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid #e8eaf0;">
            <span style="font-weight:800;color:${model_meta[mk]?.color||'#000'};width:28px;">#${i+1}</span>
            <span style="flex:1;font-weight:700;">${model_meta[mk]?.label||mk}</span>
            <span class="meta-chip"><strong style="color:${model_meta[mk]?.color||'#000'};">Composite</strong> ${(+score).toFixed(4)}</span>
          </div>`).join('')}
       </div>`
    : '';

  // Scores table
  const tableRows = Object.entries(scores).map(([mk, s]) => {
    const meta = model_meta[mk] || { label: mk, color: '#000' };
    const cells = METRIC_KEYS.map(k =>
      `<td style="padding:8px;text-align:right;">${s[k] != null ? (+s[k]).toFixed(4) : '—'}</td>`
    ).join('');
    const novel = novel_metrics[mk] || {};
    return `<tr>
      <td style="padding:8px;font-weight:700;color:${meta.color}">${meta.label}</td>
      ${cells}
      <td style="padding:8px;text-align:right;">${novel.abstractiveness != null ? (novel.abstractiveness*100).toFixed(1)+'%' : '—'}</td>
      <td style="padding:8px;text-align:right;">${novel.compression != null ? (novel.compression*100).toFixed(1)+'%' : '—'}</td>
    </tr>`;
  }).join('');

  const tableHeaders = ['Model',...METRIC_KEYS,'Abstr.%','Compr.%'].map(h =>
    `<th style="padding:8px;text-align:${h==='Model'?'left':'right'};font-size:0.78rem;color:#727785;font-weight:600;">${h}</th>`
  ).join('');

  // Summaries accordion
  const summariesHtml = Object.entries(summaries).map(([mk, s]) => {
    const meta = model_meta[mk] || { label: mk, color: '#000' };
    return `<div style="border-left:3px solid ${meta.color};padding-left:12px;margin-bottom:16px;">
      <div style="font-weight:700;margin-bottom:4px;">${meta.label}</div>
      <div style="font-size:0.9rem;color:#3d4152;line-height:1.6;">${s}</div>
    </div>`;
  }).join('');

  container.innerHTML = `
    ${recHtml}
    ${rankHtml}
    <div class="card-label" style="margin-bottom:8px;">Score Breakdown</div>
    <div style="overflow-x:auto;margin-bottom:20px;">
      <table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
        <thead><tr style="border-bottom:2px solid #e8eaf0;">${tableHeaders}</tr></thead>
        <tbody>${tableRows}</tbody>
      </table>
    </div>
    <details style="margin-bottom:12px;">
      <summary style="cursor:pointer;font-weight:700;color:#0058be;padding:8px 0;">View generated summaries</summary>
      <div style="margin-top:12px;">${summariesHtml}</div>
    </details>
    <div style="display:flex;gap:8px;flex-wrap:wrap;">
      <button class="btn-secondary" onclick="exportMetricsCSV(${JSON.stringify(JSON.stringify(data))})">⬇️ Export CSV</button>
      <button class="btn-secondary" onclick="exportMetricsJSON(${JSON.stringify(JSON.stringify(data))})">⬇️ Export JSON</button>
    </div>`;
}

function exportMetricsCSV(dataStr) {
  const data = JSON.parse(dataStr);
  const rows = [['Model','ROUGE-1','ROUGE-2','ROUGE-L','BLEU','BERTScore_F1','METEOR','Composite','Abstractiveness','Compression']];
  Object.entries(data.scores).forEach(([mk, s]) => {
    const n = data.novel_metrics[mk] || {};
    rows.push([data.model_meta[mk]?.label||mk, s['ROUGE-1']||'', s['ROUGE-2']||'', s['ROUGE-L']||'',
      s['BLEU']||'', s['BERTScore_F1']||'', s['METEOR']||'', s.composite||'',
      n.abstractiveness||'', n.compression||'']);
  });
  const csv  = rows.map(r => r.join(',')).join('\n');
  const link = Object.assign(document.createElement('a'), {
    href: 'data:text/csv,' + encodeURIComponent(csv),
    download: 'evaluation_results.csv',
  });
  link.click();
}
function exportMetricsJSON(dataStr) {
  const data = JSON.parse(dataStr);
  const link = Object.assign(document.createElement('a'), {
    href: 'data:application/json,' + encodeURIComponent(JSON.stringify(data, null, 2)),
    download: 'evaluation_results.json',
  });
  link.click();
}

// ═══════════════════════════════════════════════════════════════════════════
//  PAGE: BATCH EVALUATE
// ═══════════════════════════════════════════════════════════════════════════

let batchCsvFile = null;

function handleBatchDrop(e) {
  e.preventDefault();
  dragLeave('dropzone');
  const file = e.dataTransfer.files[0];
  if (file) handleBatchFile(file);
}
function batchFileSelected(input) {
  if (input.files[0]) handleBatchFile(input.files[0]);
}
function handleBatchFile(file) {
  batchCsvFile = file;
  const dz = document.getElementById('dropzone');
  dz.innerHTML = `
    <span class="material-symbols-outlined dropzone-icon" style="font-size:36px;color:#065f46;">check_circle</span>
    <p class="dropzone-title" style="color:#065f46;">${file.name}</p>
    <p class="dropzone-sub">${(file.size/1024).toFixed(1)} KB · Ready to process</p>`;
  showToast(`Loaded: ${file.name}`, 'success');
}

async function runBatch() {
  if (!batchCsvFile) { showToast('Please upload a CSV file first.', 'error'); return; }

  const model    = document.getElementById('batch-model-select')?.value || 'bart-cnn';
  const maxRows  = parseInt(document.getElementById('batch-max-rows')?.value || '50');
  const resultsEl = document.getElementById('batch-results');
  const barEl    = document.getElementById('batch-progress');
  const statusEl = document.getElementById('batch-status');

  resultsEl.style.display = 'block';
  barEl.style.width       = '0%';
  statusEl.textContent    = 'Uploading and processing…';

  // Animate progress bar while waiting
  let progress = 0;
  const ticker = setInterval(() => {
    progress = Math.min(progress + 0.5, 90);
    barEl.style.width = progress + '%';
  }, 400);

  try {
    const form = new FormData();
    form.append('file',     batchCsvFile);
    form.append('model',    model);
    form.append('max_rows', maxRows);

    const res  = await apiFetch('/api/batch', { method: 'POST', body: form });
    const data = await res.json();

    clearInterval(ticker);
    barEl.style.width  = '100%';
    barEl.style.background = '#065f46';
    statusEl.textContent = `✅ Done — ${data.rows_processed} rows processed with ${data.model_label}`;

    // Averages
    const avgContainer = document.getElementById('batch-averages');
    if (avgContainer && data.averages) {
      avgContainer.innerHTML = Object.entries(data.averages)
        .filter(([k]) => ['ROUGE-1','ROUGE-2','BERTScore_F1','METEOR','composite'].includes(k))
        .map(([k,v]) =>
          `<div class="stat-card"><div class="stat-value">${(+v).toFixed(4)}</div><div class="stat-label">${k}</div></div>`
        ).join('');
      avgContainer.style.display = 'flex';
    }

    // Results table
    const tableContainer = document.getElementById('batch-table');
    if (tableContainer && data.results) {
      const rows = data.results.slice(0, 20);
      tableContainer.innerHTML = `
        <table style="width:100%;border-collapse:collapse;font-size:0.82rem;">
          <thead><tr style="border-bottom:2px solid #e8eaf0;">
            <th style="padding:8px;text-align:left;">#</th>
            <th style="padding:8px;text-align:left;">Input Preview</th>
            <th style="padding:8px;text-align:left;">Summary</th>
            <th style="padding:8px;">R1</th><th style="padding:8px;">BERTScore</th><th style="padding:8px;">Composite</th>
          </tr></thead>
          <tbody>${rows.map(r => `
            <tr style="border-bottom:1px solid #f0f2f7;">
              <td style="padding:8px;color:#727785;">${r.row}</td>
              <td style="padding:8px;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${r.preview}</td>
              <td style="padding:8px;max-width:250px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${r.summary}</td>
              <td style="padding:8px;text-align:center;">${r['ROUGE-1']!=null?(+r['ROUGE-1']).toFixed(3):'—'}</td>
              <td style="padding:8px;text-align:center;">${r['BERTScore_F1']!=null?(+r['BERTScore_F1']).toFixed(3):'—'}</td>
              <td style="padding:8px;text-align:center;">${r.composite!=null?(+r.composite).toFixed(3):'—'}</td>
            </tr>`).join('')}
          </tbody>
        </table>
        ${data.results.length > 20 ? `<p style="text-align:center;color:#727785;padding:8px;">…and ${data.results.length-20} more rows</p>` : ''}`;
      tableContainer.style.display = 'block';

      // Export button
      const exportBtn = document.getElementById('batch-export');
      if (exportBtn) {
        exportBtn.style.display = 'inline-flex';
        exportBtn.onclick = () => {
          const csv = [
            ['row','preview','summary','ROUGE-1','ROUGE-2','BERTScore_F1','METEOR','composite'].join(','),
            ...data.results.map(r =>
              [r.row, `"${r.preview.replace(/"/g,'""')}"`, `"${r.summary.replace(/"/g,'""')}"`,
               r['ROUGE-1']||'', r['ROUGE-2']||'', r['BERTScore_F1']||'', r['METEOR']||'', r.composite||''].join(',')
            )
          ].join('\n');
          Object.assign(document.createElement('a'), {
            href: 'data:text/csv,' + encodeURIComponent(csv), download: 'batch_results.csv'
          }).click();
        };
      }
    }
  } catch (e) {
    clearInterval(ticker);
    barEl.style.background = '#ba1a1a';
    statusEl.textContent = '❌ Error: ' + e.message;
    showToast('Batch failed: ' + e.message, 'error');
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  PAGE: MULTILINGUAL
// ═══════════════════════════════════════════════════════════════════════════

async function runMultilingual() {
  const text = document.getElementById('ml-input')?.value.trim();
  if (!text) { showToast('Please enter text first.', 'error'); return; }

  const langOverride = document.getElementById('lang-override')?.value || null;
  const out          = document.getElementById('ml-output');
  out.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:200px;"><span class="spinner" style="width:32px;height:32px;border-width:3px;"></span></div>';

  try {
    // First detect language for immediate badge update
    const detRes = await apiFetch('/api/detect-language', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const detData = await detRes.json();
    const langEl  = document.getElementById('lang-label');
    if (langEl) langEl.textContent = `${detData.flag} ${detData.name} (${detData.code})`;

    // Populate language override selector
    const overrideEl = document.getElementById('lang-override');
    if (overrideEl && detData.supported_languages && !overrideEl.options.length) {
      const LANG_META = { en:'🇬🇧 English', ta:'🇮🇳 Tamil', hi:'🇮🇳 Hindi',
        fr:'🇫🇷 French', de:'🇩🇪 German', es:'🇪🇸 Spanish', zh:'🇨🇳 Chinese', ar:'🇸🇦 Arabic' };
      overrideEl.innerHTML = `<option value="">Auto-detect</option>` +
        detData.supported_languages.map(c =>
          `<option value="${c}"${c===detData.code?' selected':''}>${LANG_META[c]||c}</option>`
        ).join('');
    }

    // Now summarize
    const res  = await apiFetch('/api/multilingual', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, lang_override: langOverride }),
    });
    const data = await res.json();

    const bert       = data.bert_scores || {};
    const bertChips  = Object.entries(bert).map(([k,v]) =>
      `<span class="meta-chip"><strong style="color:#0058be;">${k}</strong> ${(+v).toFixed(4)}</span>`
    ).join('');

    const translatedBlock = data.translated_summary ? `
      <div style="margin-top:16px;">
        <div class="card-label" style="margin-bottom:8px;">🇬🇧 English Translation</div>
        <div style="background:var(--surface-low);border-radius:12px;padding:16px;">
          <p class="output-text">${data.translated_summary}</p>
        </div>
      </div>` : '';

    out.innerHTML = `
      <div class="output-header-row">
        <span class="model-chip">
          <span class="material-symbols-outlined" style="font-size:14px;">translate</span>
          ${data.model_label}
        </span>
        <span class="lang-chip">${data.flag} ${data.lang_name}</span>
        <button class="icon-btn" onclick="copyText(this)" title="Copy">
          <span class="material-symbols-outlined" style="font-size:18px;">content_copy</span>
        </button>
      </div>
      <div class="card-label" style="margin-bottom:8px;">${data.flag} ${data.lang_name} Summary</div>
      <div style="background:var(--surface-low);border-radius:12px;padding:16px;margin-bottom:12px;">
        <p class="output-text">${data.summary}</p>
      </div>
      ${translatedBlock}
      ${bert && Object.keys(bert).length ? `<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:12px;">${bertChips}</div>` : ''}`;

  } catch (e) {
    out.innerHTML = `<p style="color:#ba1a1a;padding:20px;">Error: ${e.message}</p>`;
    showToast('Multilingual failed: ' + e.message, 'error');
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  PAGE: BLUEPRINT SUMMARIZER
// ═══════════════════════════════════════════════════════════════════════════

let blueprintFile = null;

function dropBlueprint(e) {
  e.preventDefault();
  dragLeave('bp-dropzone');
  const file = e.dataTransfer.files[0];
  if (file) blueprintFileReady(file);
}
function blueprintFileSelected(input) {
  if (input.files[0]) blueprintFileReady(input.files[0]);
}
function blueprintFileReady(file) {
  blueprintFile = file;
  const dz = document.getElementById('bp-dropzone');
  if (dz) {
    dz.innerHTML = `
      <span class="material-symbols-outlined dropzone-icon" style="font-size:36px;color:#065f46;">check_circle</span>
      <p class="dropzone-title" style="color:#065f46;">${file.name}</p>
      <p class="dropzone-sub">${(file.size/1024).toFixed(1)} KB · Ready to analyze</p>`;
  }
  showToast(`Blueprint ready: ${file.name}`, 'success');
}
function clearBlueprint() {
  blueprintFile = null;
  const dz = document.getElementById('bp-dropzone');
  if (dz) dz.innerHTML = `
    <span class="material-symbols-outlined dropzone-icon">architecture</span>
    <p class="dropzone-title">Drop blueprint here</p>
    <p class="dropzone-sub">JPG · PNG · PDF · DXF — or click to browse</p>`;
  document.getElementById('bp-file-input').value = '';
}

async function runBlueprint() {
  if (!blueprintFile) {
    showToast('Please upload a blueprint file first.', 'error');
    return;
  }

  const btn = document.getElementById('bp-run-btn');
  if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Analyzing…'; }

  const placeholder = document.getElementById('bp-placeholder');
  const result      = document.getElementById('bp-result');
  const out         = document.getElementById('bp-output');

  if (placeholder) placeholder.style.display = 'none';
  if (result) { result.style.display = 'block'; result.innerHTML = '<div style="padding:40px;text-align:center;"><span class="spinner" style="width:32px;height:32px;border-width:3px;"></span><p style="margin-top:12px;color:#727785;">Extracting rooms & calculating areas…</p></div>'; }

  try {
    const method = document.getElementById('bp-method')?.value || 'auto';
    const unit   = document.getElementById('bp-unit')?.value   || 'sq ft';

    const form = new FormData();
    form.append('file',   blueprintFile);
    form.append('method', method);
    form.append('unit',   unit);

    const res  = await apiFetch('/api/blueprint', { method: 'POST', body: form }, 120000);
    const data = await res.json();

    renderBlueprintResult(data, unit);
  } catch (e) {
    if (result) result.innerHTML = `<p style="color:#ba1a1a;padding:20px;">Analysis failed: ${e.message}</p>`;
    showToast('Blueprint analysis failed: ' + e.message, 'error');
  } finally {
    if (btn) { btn.disabled = false; btn.innerHTML = '<span class="material-symbols-outlined" style="font-size:16px;">search</span> Analyze Blueprint'; }
  }
}

function renderBlueprintResult(data, unit) {
  const result = document.getElementById('bp-result');
  if (!result) return;

  const conf    = (data.confidence * 100).toFixed(0);
  const confColor = data.confidence >= 0.8 ? '#065f46' : data.confidence >= 0.6 ? '#924700' : '#ba1a1a';

  const rooms = data.room_breakdown || [];
  const areaCol = unit === 'sq m' ? 'area_sqm' : 'area_sqft';
  const roomRows = rooms.map(r => `
    <tr style="border-bottom:1px solid #f0f2f7;">
      <td style="padding:7px 8px;">${r.floor||'F1'}</td>
      <td style="padding:7px 8px;font-weight:600;">${r.room||'Room'}</td>
      <td style="padding:7px 8px;text-align:right;">${r.width_ft||'—'}</td>
      <td style="padding:7px 8px;text-align:right;">${r.length_ft||'—'}</td>
      <td style="padding:7px 8px;text-align:right;font-weight:700;">${r[areaCol]!=null?r[areaCol]+' '+unit:'—'}</td>
      <td style="padding:7px 8px;color:#727785;font-size:0.8rem;">${(Array.isArray(r.features) ? r.features : (typeof r.features === 'string' ? r.features.split(',') : [])).join(', ')||'—'}</td>
    </tr>`).join('');

  const features = data.special_features || {};
  const featureTags = [
    features.garage   ? '🚗 Garage'  : '',
    features.basement ? '⬇️ Basement' : '',
    features.garden   ? '🌿 Garden'  : '',
    ...(features.extras||[]).map(f => `• ${f}`),
  ].filter(Boolean).map(t => `<span class="meta-chip">${escapeHtml(t)}</span>`).join('');

  result.innerHTML = `
    <div class="output-header-row" style="margin-bottom:16px;">
      <span class="model-chip"><span class="material-symbols-outlined" style="font-size:14px;">architecture</span> ${data.engine_used}</span>
      <span class="meta-chip" style="color:${confColor};">Confidence ${conf}%</span>
      <span class="meta-chip">Coverage ${(data.coverage*100).toFixed(0)}%</span>
      <span class="meta-chip">Quality ${(data.quality*100).toFixed(0)}%</span>
    </div>

    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px;margin-bottom:20px;">
      ${[
        ['Total Area', data.total_area != null ? data.total_area+' '+unit : '—'],
        ['Bedrooms',   data.num_bedrooms ?? '—'],
        ['Bathrooms',  data.num_bathrooms ?? '—'],
        ['Floors',     data.num_floors ?? '—'],
        ['Rooms',      data.rooms_detected ?? '—'],
      ].map(([l,v]) => `<div class="stat-card"><div class="stat-value">${v}</div><div class="stat-label">${l}</div></div>`).join('')}
    </div>

    <div class="card-label" style="margin-bottom:8px;">🗂️ Room Breakdown</div>
    <div style="overflow-x:auto;margin-bottom:16px;">
      <table style="width:100%;border-collapse:collapse;font-size:0.83rem;">
        <thead><tr style="border-bottom:2px solid #e8eaf0;">
          <th style="padding:7px 8px;text-align:left;font-size:0.75rem;color:#727785;">Floor</th>
          <th style="padding:7px 8px;text-align:left;font-size:0.75rem;color:#727785;">Room</th>
          <th style="padding:7px 8px;text-align:right;font-size:0.75rem;color:#727785;">Width(ft)</th>
          <th style="padding:7px 8px;text-align:right;font-size:0.75rem;color:#727785;">Length(ft)</th>
          <th style="padding:7px 8px;text-align:right;font-size:0.75rem;color:#727785;">Area (${unit})</th>
          <th style="padding:7px 8px;text-align:left;font-size:0.75rem;color:#727785;">Features</th>
        </tr></thead>
        <tbody>${roomRows || '<tr><td colspan="6" style="padding:20px;text-align:center;color:#727785;">No rooms detected</td></tr>'}</tbody>
      </table>
    </div>

    ${featureTags ? `<div style="margin-bottom:16px;"><div class="card-label" style="margin-bottom:8px;">✨ Special Features</div><div style="display:flex;flex-wrap:wrap;gap:6px;">${featureTags}</div></div>` : ''}

    <div class="card-label" style="margin-bottom:8px;">📝 AI Summary</div>
    <div style="background:var(--surface-low);border-radius:12px;padding:16px;margin-bottom:16px;">
      <p class="output-text">${escapeHtml(data.short_summary)}</p>
    </div>
    <details>
      <summary style="cursor:pointer;font-weight:700;color:#0058be;padding:4px 0;">Full Description</summary>
      <div style="margin-top:8px;font-size:0.88rem;color:#3d4152;line-height:1.7;">${escapeHtml(data.full_description)}</div>
    </details>

    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:16px;">
      <button class="btn-secondary" onclick='downloadBlueprintJSON(${JSON.stringify(JSON.stringify(data))})'>⬇️ JSON</button>
    </div>`;
}

function downloadBlueprintJSON(dataStr) {
  const data = JSON.parse(dataStr);
  Object.assign(document.createElement('a'), {
    href: 'data:application/json,' + encodeURIComponent(JSON.stringify(data, null, 2)),
    download: 'blueprint_analysis.json',
  }).click();
}

// ── Legacy Blueprint (text mode) — kept for backward compat ─────────────────
// The Stitch UI also supports text-mode blueprint (summarize page's Blueprint section)
async function runLegacyBlueprint() {
  const text = document.getElementById('bp-text')?.value.trim();
  if (!text) { showToast('Please enter a document first.', 'error'); return; }
  const res = await apiFetch('/api/summarize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, model: 'bart-cnn', length_mode: 'detailed' }),
  });
  const data = await res.json();
  const result = document.getElementById('bp-result');
  if (result) {
    result.style.display = 'block';
    result.innerHTML = `<div class="output-header-row"><span class="model-chip">Blueprint Mode</span></div>
      <p class="output-text">${data.summary}</p>`;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  HISTORY
// ═══════════════════════════════════════════════════════════════════════════

async function loadHistory() {
  try {
    const res  = await fetch(`${API_BASE}/api/history?limit=50`);
    const data = await res.json();
    const container = document.getElementById('history-list');
    if (!container) return;
    if (!data.history || !data.history.length) {
      container.innerHTML = '<p style="color:#727785;padding:20px;">No history yet.</p>';
      return;
    }
    container.innerHTML = [...data.history].reverse().map(h => `
      <div style="border-bottom:1px solid #e8eaf0;padding:12px 0;">
        <div style="font-size:0.78rem;color:#727785;margin-bottom:4px;">${h.timestamp||''} · ${h.model||''}</div>
        <div style="font-size:0.88rem;color:#191b23;line-height:1.5;">${escapeHtml(String(h.summary||'').substring(0,300))}…</div>
      </div>`).join('');
  } catch { /* history optional */ }
}

// ═══════════════════════════════════════════════════════════════════════════
//  SHARED UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

function escapeHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#039;');
}

async function apiFetch(path, options = {}, timeoutMs = 90000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(API_BASE + path, { ...options, signal: controller.signal });
    if (!res.ok) {
      let msg = `Server error ${res.status}`;
      try { const err = await res.json(); msg = err.detail || JSON.stringify(err); } catch {}
      throw new Error(msg);
    }
    return res;
  } finally {
    clearTimeout(timer);
  }
}
