const state = {
  catalog: null,
  overview: null,
  route: "overview",
  overviewFeatures: "baseline",
  experimentMetric: "ap",
  experimentSplit: "all",
  experimentLead: "all",
  experimentSort: "ap",
  forecast: {
    lead: 24,
    split: "region_split",
    features: "baseline",
    family: "xgboost",
    inputTab: "example",
    exampleKind: "flare",
    observations: null,
    source: null,
    knownLabel: null,
    prediction: null,
    comparison: null,
    trace: "R_VALUE",
    busy: false,
    error: null,
  },
};

const page = document.getElementById("page");
const statusText = document.getElementById("system-status");
const toast = document.getElementById("toast");
const names = {overview:"Overview", forecast:"Forecast studio", experiments:"Experiments", data:"Data & method", signals:"Model signals"};
const splitName = {region_split:"Unseen regions", time_split:"Later partitions", random_split:"Random diagnostic"};
const modelName = {xgboost:"XGBoost", logistic:"Logistic regression", forest:"Random forest", ann:"Neural network"};
const featureName = {baseline:"Magnetic state", baseline_temporal:"State + temporal"};
const format = (value, digits=3) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
const comma = value => new Intl.NumberFormat("en-US").format(Number(value));
const percent = (value, digits=1) => `${(Number(value)*100).toFixed(digits)}%`;
const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));

function notify(message) {
  toast.textContent = message;
  toast.classList.add("visible");
  clearTimeout(notify.timer);
  notify.timer = setTimeout(() => toast.classList.remove("visible"), 4800);
}

async function api(path, options={}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);
  try {
    const base = String(window.HELION_API_BASE || "").replace(/\/$/, "");
    const response = await fetch(`${base}${path}`, {...options, signal:controller.signal});
    let body;
    try { body = await response.json(); } catch { body = null; }
    if (!response.ok) throw new Error(body?.error?.message || `Request failed (${response.status})`);
    return body;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("The model request timed out. Try again.");
    throw error;
  } finally { clearTimeout(timeout); }
}

function currentModel() {
  if (!state.catalog) return null;
  const f = state.forecast;
  return state.catalog.models.find(m => !m.diagnostic_only && m.lead_hours === f.lead &&
    m.split === f.split && m.feature_set === f.features && m.model === f.family);
}

function setRoute() {
  const hash = location.hash.replace(/^#/, "") || "overview";
  state.route = names[hash] ? hash : "overview";
  document.title = `${names[state.route]} — Helion`;
  document.getElementById("current-page").textContent = names[state.route];
  document.querySelectorAll("[data-route]").forEach(link => {
    const active = link.dataset.route === state.route;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current");
  });
  render();
  window.scrollTo({top:0, behavior:"auto"});
}

function heading(label, title, subtitle, stamp="VERIFIED SAVED ARTIFACTS") {
  return `<div class="page-heading"><div><span class="eyebrow">${label}</span><h1>${title}</h1><p>${subtitle}</p></div><div class="meta-stamp">${stamp}</div></div>`;
}

function metricCard(label, value, detail, tone="") {
  return `<div class="metric-card ${tone}"><span class="metric-label">${label}</span><strong>${value}</strong><small>${detail}</small><div class="meter"></div></div>`;
}

function segment(options, selected, action, variant="") {
  return `<div class="segmented ${variant}" role="group" aria-label="${escapeHtml(action)}">${options.map(([value,label]) =>
    `<button type="button" data-action="${action}" data-value="${escapeHtml(value)}" class="${String(value)===String(selected)?"active":""}" aria-pressed="${String(value)===String(selected)}">${label}</button>`).join("")}</div>`;
}

function pathFor(points) { return points.map((point,i) => `${i?"L":"M"}${point[0].toFixed(1)} ${point[1].toFixed(1)}`).join(" "); }

function leadChart(featureSet="baseline", metric="ap", onlySplit="all") {
  const runs = state.overview.results.filter(r => r.phase === "phase3" && r.features === featureSet && r.model === "xgboost");
  const leads = [6,12,24,48], width=610, height=265, left=45, right=22, top=20, bottom=36;
  const values = runs.map(r => Number(r[metric]));
  const max = Math.max(.5, Math.ceil(Math.max(...values)*10)/10);
  const x = h => left + (leads.indexOf(h)/(leads.length-1))*(width-left-right);
  const y = v => top + (max-v)/max*(height-top-bottom);
  const tick = [0,.25,.5,.75,1].map(i => i*max);
  const grid = tick.map(t => `<line class="chart-grid-line" x1="${left}" y1="${y(t)}" x2="${width-right}" y2="${y(t)}"/><text class="axis-label" x="${left-10}" y="${y(t)+4}" text-anchor="end">${t.toFixed(2)}</text>`).join("");
  const lines = [["region_split", "#d45a2e"],["time_split", "#323b37"]].filter(([split]) => onlySplit === "all" || onlySplit === split).map(([split,color]) => {
    const points = leads.map(h => {
      const run = runs.find(r => r.lead_hours === h && r.split === split);
      return [x(h),y(Number(run[metric])),h,Number(run[metric])];
    });
    return `<path class="chart-line" d="${pathFor(points)}" stroke="${color}"/>` + points.map(([cx,cy,h,value]) =>
      `<circle class="chart-point" cx="${cx}" cy="${cy}" r="6" fill="${color}"><title>${splitName[split]} · ${h}h · ${metric.toUpperCase()} ${format(value)}</title></circle>`).join("");
  }).join("");
  return `<div class="chart-wrap"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${metric.toUpperCase()} by lead for unseen regions and later partitions">${grid}${lines}${leads.map(h => `<text class="axis-label" x="${x(h)}" y="${height-10}" text-anchor="middle">${h}h</text>`).join("")}</svg></div>`;
}

function renderOverview() {
  const info = state.overview;
  const region6 = info.results.find(r => r.phase==="phase3" && r.lead_hours===6 && r.split==="region_split" && r.features==="baseline");
  const region48 = info.results.find(r => r.phase==="phase3" && r.lead_hours===48 && r.split==="region_split" && r.features==="baseline");
  return `<div class="hero"><span class="hero-orbit" aria-hidden="true"></span><div class="hero-content"><span class="eyebrow">SWAN-SF / RESEARCH OBSERVATORY</span><h1>A clearer view of solar flare forecasts.</h1><p>Explore the verified runs, test a saved model, and see what changes when the forecast window moves farther ahead.</p><div class="hero-actions"><a class="button button-primary" href="#forecast">Open forecast studio <span aria-hidden="true">↗</span></a><a class="button button-quiet" href="#experiments">Explore results <span aria-hidden="true">↗</span></a></div></div><span class="hero-tag">M / X PEAK FORECASTS · 24H TARGET WINDOW</span></div>
    <div class="metric-grid">
      ${metricCard("Observation windows",comma(info.dataset.windows),"12-hour windows in the local extraction")}
      ${metricCard("Active regions",comma(info.dataset.regions),"Distinct region IDs")}
      ${metricCard("Verified runs",comma(info.verified_runs),"Saved models and metric checks","gold")}
      ${metricCard("Random overlap",percent(info.splits.random_split.overlap_fraction,2),"Random-test intervals sharing training time","hot")}
    </div>
    <div class="content-grid"><div class="stack">
      <section class="panel panel-pad"><div class="panel-header"><div><div class="panel-kicker">PERFORMANCE / POSITIVE LEADS</div><h2 class="panel-title">How ranking changes with lead</h2><p class="panel-subtitle">Held-out XGBoost average precision across four separate target intervals.</p></div>${segment([["baseline","Magnetic state"],["baseline_temporal","+ Temporal"]],state.overviewFeatures,"overview-features")}</div>${leadChart(state.overviewFeatures,"ap")}<div class="legend"><span><i style="background:#d45a2e"></i>Unseen regions</span><span><i style="background:#323b37"></i>Later partitions</span></div><div class="panel-foot">The event interval moves with the selected lead. AP comparisons also reflect changes in target prevalence.</div></section>
      <section class="panel panel-pad"><div class="panel-header"><div><div class="panel-kicker">MODEL EVIDENCE</div><h2 class="panel-title">What the model actually predicts</h2></div><span class="pill ember">PEAK, NOT ONSET</span></div><div class="timeline"><div class="timeline-segment observed"><strong>12h observed</strong><small>Magnetic readings</small></div><div class="timeline-segment wait"><strong>Lead H</strong><small>0–48 hours</small></div><div class="timeline-segment target"><strong>24h target window</strong><small>Recorded M/X flare peak</small></div></div><div class="timeline-labels"><span>Observation starts</span><span>Forecast issued</span><span>Target closes</span></div><div class="panel-foot">A 6h lead means a peak between 6 and 30 hours after issue time. It does not mean six hours before flare onset.</div></section>
    </div><div class="stack">
      <section class="panel panel-pad"><div class="panel-header"><div><div class="panel-kicker">UNSEEN REGIONS</div><h2 class="panel-title">Earlier is harder here</h2></div><span class="pill">BASELINE XGBOOST</span></div><div style="display:flex;align-items:baseline;gap:15px;margin:18px 0 12px"><span class="serif" style="font-size:55px;letter-spacing:-.07em;color:#af492a">${format(region6.ap)}</span><span style="color:#aab0a8;font-size:20px">→</span><span class="serif" style="font-size:55px;letter-spacing:-.07em">${format(region48.ap)}</span></div><p class="note">Average precision from a 6-hour minimum lead to a 48-hour minimum lead. These are related, distinct peak-window tasks.</p><a class="link-button" href="#experiments">See all 33 runs ↗</a></section>
      <div class="callout"><span class="eyebrow">A NOTE ON TRUST</span><h3>Similar windows can flatter a model.</h3><p>49,758 of 49,815 random-test windows share observation time with training. Region and chronological tests have zero measured train-test overlap in the audit.</p><a class="link-button" href="#data">See the split design ↗</a></div>
      <section class="panel panel-pad"><div class="panel-header"><div><div class="panel-kicker">INPUT CONTRACT</div><h2 class="panel-title">One window. Two views.</h2></div></div><div class="stat-strip"><div><small>MAGNETIC INPUTS</small><strong>24</strong><small>Final values</small></div><div><small>TEMPORAL</small><strong>120</strong><small>Lags & rolling</small></div><div><small>CADENCE</small><strong>12m</strong><small>60 readings</small></div></div><div class="panel-foot">Flare annotations and future X-ray fields stay out of predictive features.</div></section>
    </div></div>`;
}

function selectionBlock(title, hint, content) {
  return `<div class="control-block"><div class="control-heading"><strong>${title}</strong><small>${hint}</small></div>${content}</div>`;
}

function traceSVG(observations, key) {
  const values = observations.map(row => row[key] === null || row[key] === "" || row[key] === undefined ? NaN : Number(row[key]));
  const finite = values.filter(Number.isFinite);
  if (!finite.length) return `<p class="empty-state">No finite ${escapeHtml(key)} readings in this window.</p>`;
  const min=Math.min(...finite), max=Math.max(...finite), spread=(max-min)||1;
  const width=550,height=170,left=15,right=13,top=15,bottom=27;
  const points = values.map((value,i) => Number.isFinite(value) ? [left+i/(values.length-1)*(width-left-right),top+(max-value)/spread*(height-top-bottom)] : null);
  let paths=[], current=[];
  for (const point of points) { if (point) current.push(point); else if (current.length) { paths.push(current); current=[]; } }
  if (current.length) paths.push(current);
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(key)} values over the 12-hour observation window" style="width:100%;height:auto"><line x1="${left}" y1="${height-bottom}" x2="${width-right}" y2="${height-bottom}" stroke="#e6e5df"/>${paths.map(part => `<path d="${pathFor(part)}" fill="none" stroke="#d86231" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>`).join("")}<text class="axis-label" x="${left}" y="${height-5}">12h before issue</text><text class="axis-label" x="${width-right}" y="${height-5}" text-anchor="end">Issue time</text></svg>`;
}

function renderPrediction() {
  const result=state.forecast.prediction;
  if (!result) return `<div class="result-placeholder"><span class="orb-small" aria-hidden="true"></span><h3>Your reading, in context.</h3><p>Select a verified example or upload a 60-reading CSV, then run a saved model to see its estimate.</p></div>`;
  const high=result.positive;
  return `<div class="panel-kicker">MODEL SCORE FOR M/X PEAK</div><div class="probability">${percent(result.probability,1)}</div><div class="probability-track"><span style="width:${Math.max(1,result.probability*100)}%"></span><i style="left:${Math.min(99,result.threshold*100)}%"></i></div><div class="track-labels"><span>0%</span><span>Saved threshold ${percent(result.threshold,1)}</span><span>100%</span></div><div class="verdict ${high?"":"quiet"}"><span class="verdict-mark">${high?"↑":"✓"}</span><strong>${high?"Above the saved threshold":"Below the saved threshold"}</strong></div><div class="result-details"><div class="result-detail"><span>Forecast issued</span><span>${escapeHtml(formatDate(result.issue_time))}</span></div><div class="result-detail"><span>Peak window begins</span><span>${escapeHtml(formatDate(result.target_start))}</span></div><div class="result-detail"><span>Peak window ends</span><span>${escapeHtml(formatDate(result.target_end))}</span></div><div class="result-detail"><span>Readings present</span><span>${percent(result.input_completeness,1)}</span></div>${state.forecast.knownLabel!==null?`<div class="result-detail"><span>Known historical label</span><span>${state.forecast.knownLabel?"M/X peak recorded":"No M/X peak in interval"}</span></div>`:""}</div>${result.warning?`<div class="result-warning">${escapeHtml(result.warning)}</div>`:""}<div class="result-warning">This retrospective model score is not calibrated for operational alerts.</div>${renderComparison()}`;
}

function renderComparison() {
  const f = state.forecast;
  if (f.family !== "xgboost") return "";
  if (!f.comparison) return `<button type="button" class="button button-outline compare-button" data-action="compare" ${f.busy?"disabled":""}>Compare all five leads <span aria-hidden="true">↗</span></button>`;
  return `<div class="comparison"><div class="panel-kicker">SAME INPUT / FIVE TARGET WINDOWS</div>${f.comparison.predictions.map(item=>`<div class="comparison-row"><span>${item.lead_hours}h</span><div class="comparison-track"><span style="width:${Math.max(1,item.probability*100)}%"></span></div><strong>${percent(item.probability,1)}</strong></div>`).join("")}<p>Each score uses a different target interval and is not a calibrated alert probability.</p></div>`;
}

function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("en-GB", {day:"2-digit",month:"short",year:"numeric",hour:"2-digit",minute:"2-digit",hour12:false}).format(date);
}

function renderForecast() {
  const f=state.forecast, model=currentModel();
  return `${heading("FORECAST STUDIO","Run a saved forecast.","Use a held-out historical window or supply your own 60-reading magnetic series. The same preprocessing used in training runs here.","SAVED MODEL INFERENCE")}
  <div class="forecast-layout"><div class="stack"><section class="panel panel-pad"><div class="panel-header"><div><div class="panel-kicker">CONFIGURATION</div><h2 class="panel-title">Set the forecast question</h2></div><span class="pill green">${model?"MODEL READY":"MODEL UNAVAILABLE"}</span></div>
    ${selectionBlock("Minimum lead", "TO RECORDED PEAK", segment([[0,"0h"],[6,"6h"],[12,"12h"],[24,"24h"],[48,"48h"]],f.lead,"forecast-lead"))}
    ${selectionBlock("Generalization protocol", "TRAINING SPLIT", segment([["region_split","Unseen regions"],["time_split","Later partitions"]],f.split,"forecast-split")+`<p class="control-help">The random split is reserved for the overlap diagnostic.</p>`)}
    ${selectionBlock("Feature representation", `${f.features==="baseline"?24:144} MODEL INPUTS`, segment([["baseline","Final magnetic state"],["baseline_temporal","State + temporal"]],f.features,"forecast-features"))}
    ${selectionBlock("Model family", f.lead===0?"H=0 COMPARISON":"POSITIVE LEADS USE XGBOOST", f.lead===0?`<select class="select-field" id="family-select" aria-label="Model family">${[["xgboost","XGBoost"],["logistic","Logistic regression"],["forest","Random forest"],["ann","Neural network"]].map(([value,label])=>`<option value="${value}" ${f.family===value?"selected":""}>${label}</option>`).join("")}</select>`:`<div class="pill">XGBoost · fixed lead experiment</div>`)}
    <div class="input-tabs" role="tablist"><button role="tab" data-action="input-tab" data-value="example" class="${f.inputTab==="example"?"active":""}" aria-selected="${f.inputTab==="example"}">Verified example</button><button role="tab" data-action="input-tab" data-value="upload" class="${f.inputTab==="upload"?"active":""}" aria-selected="${f.inputTab==="upload"}">Upload CSV / TSV</button></div>
    ${f.inputTab==="example"?`<div class="sample-picker"><button type="button" class="sample-option ${f.exampleKind==="flare"?"selected":""}" data-action="example-kind" data-value="flare"><span class="sample-icon">✦</span><strong>Recorded flare</strong><small>Held-out positive window</small></button><button type="button" class="sample-option ${f.exampleKind==="quiet"?"selected":""}" data-action="example-kind" data-value="quiet"><span class="sample-icon">○</span><strong>Quiet interval</strong><small>Held-out negative window</small></button></div><p class="source-note">${f.source?`Region ${escapeHtml(f.source.region_id)} · issue ${escapeHtml(formatDate(f.source.issue_time))} · ${f.observations.length} readings`:(f.busy?"Loading verified example…":"Select an example to load its readings.")}</p>`:`<label class="dropzone" id="dropzone"><span style="font-size:23px;color:#ca6134">⇧</span><strong>Drop a CSV or TSV here</strong><small>60 rows · Timestamp + magnetic columns · 12-minute cadence</small><input id="file-input" type="file" accept=".csv,.tsv,.txt,text/csv,text/tab-separated-values"></label><p class="upload-filename">${f.source?.type==="upload"?escapeHtml(f.source.filename):""}</p><p class="source-note">Extra flare or X-ray columns are ignored. Values are processed with the selected run's saved normalization.</p>`}
    ${f.error?`<div class="error-card" style="margin-top:15px"><strong>Input needs attention</strong>${escapeHtml(f.error)}</div>`:""}
    <div class="form-actions"><button type="button" class="button button-dark" data-action="predict" ${!f.observations||!model||f.busy?"disabled":""}>${f.busy?"Processing…":"Run forecast"}<span aria-hidden="true">↗</span></button><button type="button" class="button button-outline" data-action="reset-forecast">Reset</button></div><p class="source-note">Selected run: ${model?`${escapeHtml(modelName[model.model])} · ${escapeHtml(featureName[model.feature_set])} · ${model.test_n.toLocaleString()} held-out windows`:"Unavailable"}</p>
  </section></div><div class="stack"><section class="panel result-card"><div class="panel-pad"><div class="panel-header"><div><div class="panel-kicker">SAVED MODEL INFERENCE</div><h2 class="panel-title">Forecast readout</h2></div><span class="pill ember">${f.lead}H LEAD</span></div>${renderPrediction()}</div></section>
  ${f.observations?`<section class="panel panel-pad trace-card"><div class="panel-header"><div><div class="panel-kicker">INPUT WINDOW</div><h2 class="panel-title">Magnetic trace</h2></div></div><div class="trace-selector">${["R_VALUE","TOTUSJH","ABSNJZH","USFLUX"].map(key=>`<button type="button" data-action="trace" data-value="${key}" class="${f.trace===key?"active":""}">${key}</button>`).join("")}</div>${traceSVG(f.observations,f.trace)}<div class="panel-foot">Only readings up to forecast issue time are shown and used by the model.</div></section>`:`<section class="panel panel-pad"><div class="panel-kicker">READ BEFORE YOU RUN</div><h2 class="panel-title">The target is a peak window.</h2><p class="note">For a 24h lead, the model asks whether a recorded M/X flare peak falls between 24 and 48 hours after the last observation. It does not predict flare onset.</p></section>`}</div></div>`;
}

function renderExperiments() {
  const metric=state.experimentMetric;
  const filtered=state.overview.results.filter(r => (state.experimentSplit==="all"||r.split===state.experimentSplit) && (state.experimentLead==="all"||String(r.lead_hours)===state.experimentLead));
  const sorted=[...filtered].sort((a,b)=>Number(b[state.experimentSort])-Number(a[state.experimentSort]));
  return `${heading("VERIFIED RUNS","What the tests found.","Explore the complete 33-run matrix. Filter by lead and split, switch metrics, and keep the random diagnostic in context.")}
  <div class="metric-grid">${metricCard("Saved models",33,"17 at H=0 · 16 positive-lead runs")}${metricCard("Primary metric","AP","Average precision for rare events","gold")}${metricCard("Comparisons",8,"Paired temporal AP intervals")}${metricCard("Clear temporal gains",0,"All eight 95% intervals include zero","hot")}</div>
  <section class="panel panel-pad" style="margin-bottom:16px"><div class="panel-header"><div><div class="panel-kicker">POSITIVE LEADS / XGBOOST</div><h2 class="panel-title">Ranking and decision skill</h2><p class="panel-subtitle">The plotted score changes when you change metric. Each point is a held-out test result.</p></div>${segment([["ap","AP"],["tss","TSS"],["f1","F1"]],metric,"experiment-metric")}</div>${leadChart("baseline",metric)}<div class="legend"><span><i style="background:#d45a2e"></i>Unseen regions</span><span><i style="background:#323b37"></i>Later partitions</span></div></section>
  <section class="panel"><div class="panel-pad"><div class="panel-header"><div><div class="panel-kicker">FULL EXPERIMENT MATRIX</div><h2 class="panel-title">All held-out runs</h2></div><span class="pill">${sorted.length} RESULTS</span></div><div class="filters"><div class="filter-group"><label for="filter-split">Split</label><select id="filter-split"><option value="all">All splits</option><option value="region_split" ${state.experimentSplit==="region_split"?"selected":""}>Unseen regions</option><option value="time_split" ${state.experimentSplit==="time_split"?"selected":""}>Later partitions</option><option value="random_split" ${state.experimentSplit==="random_split"?"selected":""}>Random diagnostic</option></select></div><div class="filter-group"><label for="filter-lead">Minimum lead</label><select id="filter-lead"><option value="all">All leads</option>${[0,6,12,24,48].map(h=>`<option value="${h}" ${state.experimentLead===String(h)?"selected":""}>${h} hours</option>`).join("")}</select></div><div class="filter-spacer"></div><div class="result-count">Sorted by ${state.experimentSort.toUpperCase()} ↓</div></div></div><div class="table-wrap"><table class="results-table"><thead><tr><th>Lead</th><th>Split</th><th>Family</th><th>Features</th><th><button class="sort" data-action="sort" data-value="ap">AP ↕</button></th><th><button class="sort" data-action="sort" data-value="precision">Precision ↕</button></th><th><button class="sort" data-action="sort" data-value="recall">Recall ↕</button></th><th><button class="sort" data-action="sort" data-value="f1">F1 ↕</button></th><th><button class="sort" data-action="sort" data-value="tss">TSS ↕</button></th><th>Test n</th></tr></thead><tbody>${sorted.map(r=>`<tr><td class="num">${r.lead_hours}h</td><td>${escapeHtml(splitName[r.split])}${r.split==="random_split"?" <span class='pill ember'>DIAGNOSTIC</span>":""}</td><td>${escapeHtml(modelName[r.model])}</td><td>${escapeHtml(featureName[r.features])}</td><td class="num highlight">${format(r.ap)}</td><td class="num">${format(r.precision)}</td><td class="num">${format(r.recall)}</td><td class="num">${format(r.f1)}</td><td class="num">${format(r.tss)}</td><td class="num">${comma(r.n_test)}</td></tr>`).join("")}</tbody></table></div><div class="panel-pad"><div class="panel-foot">All values are held-out test metrics. Each classification threshold was chosen on validation rows. Random-split performance is a leakage diagnostic.</div></div></section>`;
}

function renderData() {
  const info=state.overview;
  return `${heading("DATA & METHOD","The contract behind every score.","The same indexed windows, labels, split assignments, and training-only preprocessing feed every saved run.")}
  <section class="panel panel-pad" style="margin-bottom:16px"><div class="panel-header"><div><div class="panel-kicker">ISSUE TIME / TARGET TIME</div><h2 class="panel-title">A forecast has three distinct intervals.</h2></div><span class="pill ember">PEAK ANNOTATION</span></div><div class="timeline"><div class="timeline-segment observed"><strong>Observe 12 hours</strong><small>60 magnetic readings</small></div><div class="timeline-segment wait"><strong>Wait H hours</strong><small>0, 6, 12, 24, 48</small></div><div class="timeline-segment target"><strong>Evaluate next 24 hours</strong><small>M/X peak in (issue+H, issue+H+24h]</small></div></div><p class="narrative">Targets are reconstructed from full active-region series. A window is excluded when its future interval lacks coverage, rather than marked quiet. At H=0, all 299,175 eligible reconstructed labels matched the supplied binary label.</p></section>
  <div class="content-grid"><div class="stack"><section class="panel panel-pad"><div class="panel-header"><div><div class="panel-kicker">FROM ARCHIVE TO MODEL</div><h2 class="panel-title">A reproducible path</h2></div></div><div class="method-steps"><div class="method-step"><span class="step-no">01</span><h3>Collect & index</h3><p>Five instance partitions and the full region series are joined by window and region identity.</p></div><div class="method-step"><span class="step-no">02</span><h3>Build features</h3><p>24 final magnetic values; optionally 120 lags, rates, and rolling summaries.</p></div><div class="method-step"><span class="step-no">03</span><h3>Split & normalize</h3><p>Training-only medians and z-scores are saved separately for each split and feature array.</p></div><div class="method-step"><span class="step-no">04</span><h3>Train</h3><p>Four H=0 model families. Positive leads use XGBoost with a shared eligible cohort.</p></div><div class="method-step"><span class="step-no">05</span><h3>Validate</h3><p>Early stopping and binary thresholds use validation rows, never test rows.</p></div><div class="method-step"><span class="step-no">06</span><h3>Verify</h3><p>All 33 model files passed hash, row identity, metric, and reload checks.</p></div></div></section>
    <section class="panel panel-pad"><div class="panel-header"><div><div class="panel-kicker">LABEL COVERAGE</div><h2 class="panel-title">The farther the lead, the fewer eligible windows.</h2></div></div><div class="mini-bars">${info.leads.map(item=>`<div class="mini-bar-row"><strong>${item.hours}h lead</strong><div class="mini-bar-track"><div class="mini-bar-fill" style="width:${item.valid/info.dataset.windows*100}%"></div></div><span class="mini-bar-value">${comma(item.valid)}</span></div>`).join("")}</div><div class="panel-foot">At 48h, ${comma(info.leads.find(x=>x.hours===48).censored)} original windows lack a valid target under the coverage rule.</div></section>
  </div><div class="stack"><section class="panel panel-pad"><div class="panel-header"><div><div class="panel-kicker">SPLIT AUDIT</div><h2 class="panel-title">Overlap is measurable.</h2></div></div>${[["random_split","Random"],["region_split","Region"],["time_split","Chronological"]].map(([key,label])=>`<div class="split-row"><strong>${label}</strong><div class="split-track"><span style="width:${Math.max(.35,info.splits[key].overlap_fraction*100)}%;${key==="random_split"?"":"background:#769885"}"></span></div><em>${percent(info.splits[key].overlap_fraction,2)}</em></div>`).join("")}<div class="panel-foot">Fraction of original test windows with observation time also present in training. The random split is used only as a diagnostic.</div></section><div class="callout"><span class="eyebrow">FEATURE HYGIENE</span><h3>Only the observed magnetic field enters.</h3><p>Flare annotations, X-ray values, and future measurements are excluded from predictors. Missing values use training-fitted medians.</p></div><section class="panel panel-pad"><div class="panel-kicker">CLASS BALANCE</div><h2 class="panel-title">Rare positives, by design.</h2><div class="stat-strip" style="margin-top:18px"><div><small>ALL WINDOWS</small><strong>331k</strong><small>${comma(info.dataset.windows)} exact</small></div><div><small>M/X POSITIVE</small><strong>6,234</strong><small>Supplied label</small></div><div><small>PREVALENCE</small><strong>1.88%</strong><small>Original task</small></div></div><div class="panel-foot">Average precision is reported alongside prevalence, precision, recall, and false-alarm counts.</div></section></div></div>`;
}

function renderSignals() {
  const features=state.overview.top_features;
  const max=Math.max(...features.map(f=>f.mean_absolute_tree_shap));
  const pairs=state.overview.paired_comparisons;
  return `${heading("MODEL SIGNALS","What moved the model.","Interpretation is tied to the H=24 chronological temporal XGBoost run. The bars describe fitted-model associations, not physical causes.")}
  <div class="content-grid"><div class="stack"><section class="panel panel-pad"><div class="panel-header"><div><div class="panel-kicker">TREE SHAP / SELECTED RUN</div><h2 class="panel-title">Strongest sampled feature associations</h2></div><span class="pill">H=24 · TIME SPLIT</span></div>${features.map(f=>`<div class="signal-row"><strong class="mono">${escapeHtml(f.feature)}</strong><div class="mini-bar-track"><div class="mini-bar-fill" style="width:${f.mean_absolute_tree_shap/max*100}%"></div></div><span>${format(f.mean_absolute_tree_shap,2)}</span></div>`).join("")}<div class="panel-foot">Mean absolute TreeSHAP contribution for the saved interpretation sample. Correlated inputs can share or exchange attribution.</div></section>
    <section class="panel panel-pad"><div class="panel-header"><div><div class="panel-kicker">PAIRED AP COMPARISONS</div><h2 class="panel-title">Temporal features: no clear gain</h2></div></div><p class="narrative" style="margin:0 0 15px">Adding 120 temporal summaries produced mixed point estimates. Every 95% region-bootstrap interval for the eight lead–split comparisons includes zero.</p><div class="table-wrap"><table class="results-table" style="min-width:540px"><thead><tr><th>Lead</th><th>Split</th><th>AP difference</th><th>95% region interval</th></tr></thead><tbody>${pairs.map(p=>`<tr><td>${p.lead_hours}h</td><td>${escapeHtml(splitName[p.split])}</td><td class="num ${p.difference_ap>0?"highlight":""}">${p.difference_ap>=0?"+":""}${format(p.difference_ap)}</td><td class="num">[${format(p.ci95_region_bootstrap[0])}, ${format(p.ci95_region_bootstrap[1])}]</td></tr>`).join("")}</tbody></table></div><div class="panel-foot">Difference is temporal minus baseline AP on the same test windows; 500 bootstrap samples resample active regions.</div></section>
  </div><div class="stack"><div class="callout"><span class="eyebrow">INTERPRETATION BOUNDARY</span><h3>A useful association is not a causal precursor.</h3><p>High attribution means this trained model relied on a feature in the selected sample. It does not show that changing that magnetic measurement would change flare activity.</p></div><section class="panel panel-pad"><div class="panel-header"><div><div class="panel-kicker">READING THE SIGNALS</div><h2 class="panel-title">Three recurring measurements</h2></div></div>${[["R_VALUE","3-hour rolling mean and 1-hour lag"],["TOTUSJH","3-hour rolling mean and 1-hour lag"],["ABSNJZH","3-hour rolling standard deviation"]].map(([key,description],i)=>`<div style="display:flex;gap:15px;padding:14px 0;${i?"border-top:1px solid #edede6":""}"><div><strong class="mono" style="font-size:12px">${key}</strong><p class="note" style="margin:5px 0 0">${description}</p></div></div>`).join("")}<div class="panel-foot">The H=24 chronological test has 881 positive windows from only 20 positive active regions. Pattern plots sample one window per region.</div></section></div></div>`;
}

function render() {
  if (!state.catalog || !state.overview) return;
  page.innerHTML = ({overview:renderOverview, forecast:renderForecast, experiments:renderExperiments, data:renderData, signals:renderSignals})[state.route]();
  if (state.route === "forecast" && state.forecast.inputTab === "example" && !state.forecast.observations && !state.forecast.busy && !state.forecast.error) loadExample();
}

function resetForecastInput(clearUpload=true) {
  const f=state.forecast;
  if (f.source?.type === "upload" && !clearUpload) { f.prediction=null; f.comparison=null; f.error=null; return; }
  f.observations=null; f.source=null; f.knownLabel=null; f.prediction=null; f.comparison=null; f.error=null;
}

async function loadExample() {
  const f=state.forecast, model=currentModel();
  if (!model || f.busy) return;
  f.busy=true; f.error=null; render();
  try {
    const sample=await api(`/api/example?model_id=${encodeURIComponent(model.id)}&kind=${f.exampleKind}`);
    if (state.forecast.inputTab !== "example" || currentModel()?.id !== model.id) return;
    f.observations=sample.observations;
    f.source={type:"example",region_id:sample.region_id,issue_time:sample.issue_time};
    f.knownLabel=sample.known_label; f.prediction=null; f.comparison=null;
  } catch(error) { f.error=error.message; notify(error.message); }
  finally { f.busy=false; render(); }
}

function parseDelimited(text) {
  const first=text.replace(/^\ufeff/, "").split(/\r?\n/,1)[0];
  const delimiter=(first.match(/\t/g)||[]).length > (first.match(/,/g)||[]).length ? "\t" : ",";
  const rows=[]; let row=[], field="", quoted=false;
  const source=text.replace(/^\ufeff/, "");
  for(let i=0;i<source.length;i++) {
    const ch=source[i];
    if(ch==='"') { if(quoted && source[i+1]==='"') {field+='"';i++;} else quoted=!quoted; }
    else if(ch===delimiter && !quoted) {row.push(field);field="";}
    else if((ch==="\n"||ch==="\r") && !quoted) {
      if(ch==="\r" && source[i+1]==="\n") i++;
      row.push(field); field="";
      if(row.some(cell=>cell.trim()!=="")) rows.push(row);
      row=[];
    } else field+=ch;
  }
  if(quoted) throw new Error("The CSV has an unclosed quoted field.");
  row.push(field); if(row.some(cell=>cell.trim()!=="")) rows.push(row);
  if(rows.length<2) throw new Error("The file needs a header and 60 data rows.");
  const headers=rows.shift().map(h=>h.trim());
  const stamp=headers.findIndex(h=>h.toLowerCase()==="timestamp");
  if(stamp<0) throw new Error("Add a Timestamp column to the file.");
  if(rows.length!==60) throw new Error(`Expected exactly 60 readings; found ${rows.length}.`);
  const magnetic=state.catalog.magnetic_columns;
  const mapped=rows.map((cells,i)=>{
    if(cells.length!==headers.length) throw new Error(`Reading ${i+1} has ${cells.length} fields; the header has ${headers.length}.`);
    const obj={Timestamp:cells[stamp]};
    for(const name of magnetic) {
      const index=headers.findIndex(h=>h.toUpperCase()===name);
      if(index>=0) obj[name]=cells[index].trim() || null;
    }
    return obj;
  });
  if(!magnetic.some(name=>headers.includes(name))) throw new Error("The file has no expected magnetic columns.");
  return mapped;
}

async function readUpload(file) {
  if (!file) return;
  if (file.size > 1_000_000) { notify("Choose a CSV or TSV smaller than 1 MB."); return; }
  try {
    const text=await file.text();
    const observations=parseDelimited(text);
    state.forecast.observations=observations;
    state.forecast.source={type:"upload",filename:file.name};
    state.forecast.knownLabel=null;
    state.forecast.prediction=null;
    state.forecast.comparison=null;
    state.forecast.error=null;
    notify(`${file.name} loaded: 60 readings ready to validate.`);
    render();
  } catch(error) { state.forecast.error=error.message; notify(error.message); render(); }
}

async function runForecast() {
  const f=state.forecast, model=currentModel();
  if (!model || !f.observations || f.busy) return;
  f.busy=true; f.error=null; render();
  try {
    const result=await api("/api/predict", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({model_id:model.id,observations:f.observations})});
    f.prediction=result; f.comparison=null;
    notify("Forecast complete. The saved model and preprocessing were used.");
  } catch(error) {f.error=error.message; notify(error.message);}
  finally { f.busy=false; render(); }
}

async function compareLeads() {
  const f=state.forecast;
  if (!f.observations || f.busy || f.family !== "xgboost") return;
  const modelIds=[0,6,12,24,48].map(lead=>state.catalog.models.find(m=>!m.diagnostic_only && m.lead_hours===lead && m.split===f.split && m.feature_set===f.features && m.model==="xgboost")?.id);
  if (modelIds.some(id=>!id)) return;
  f.busy=true; f.error=null; render();
  try { f.comparison=await api("/api/compare", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({model_ids:modelIds,observations:f.observations})}); }
  catch(error) { f.error=error.message; notify(error.message); }
  finally { f.busy=false; render(); }
}

page.addEventListener("click", async event => {
  const button=event.target.closest("[data-action]");
  if(!button) return;
  const action=button.dataset.action, value=button.dataset.value;
  if(action==="overview-features") { state.overviewFeatures=value; render(); }
  else if(action==="experiment-metric") { state.experimentMetric=value; render(); }
  else if(action==="sort") { state.experimentSort=value; render(); }
  else if(action==="forecast-lead") { state.forecast.lead=Number(value); if(Number(value)>0) state.forecast.family="xgboost"; resetForecastInput(state.forecast.inputTab!=="upload"); render(); }
  else if(action==="forecast-split") { state.forecast.split=value; resetForecastInput(state.forecast.inputTab!=="upload"); render(); }
  else if(action==="forecast-features") { state.forecast.features=value; resetForecastInput(state.forecast.inputTab!=="upload"); render(); }
  else if(action==="input-tab") { state.forecast.inputTab=value; resetForecastInput(); render(); }
  else if(action==="example-kind") { state.forecast.exampleKind=value; resetForecastInput(); render(); }
  else if(action==="trace") { state.forecast.trace=value; render(); }
  else if(action==="predict") { await runForecast(); }
  else if(action==="compare") { await compareLeads(); }
  else if(action==="reset-forecast") { resetForecastInput(); render(); }
});
document.addEventListener("click", event => { if (event.target.closest("[data-action='retry-boot']")) boot(); });

page.addEventListener("change", event => {
  if(event.target.id==="family-select") {state.forecast.family=event.target.value; resetForecastInput(state.forecast.inputTab!=="upload"); render();}
  if(event.target.id==="filter-split") {state.experimentSplit=event.target.value; render();}
  if(event.target.id==="filter-lead") {state.experimentLead=event.target.value; render();}
  if(event.target.id==="file-input") readUpload(event.target.files?.[0]);
});

page.addEventListener("dragover", event => { const zone=event.target.closest("#dropzone"); if(zone) {event.preventDefault();zone.classList.add("dragover");} });
page.addEventListener("dragleave", event => { const zone=event.target.closest("#dropzone"); if(zone) zone.classList.remove("dragover"); });
page.addEventListener("drop", event => { const zone=event.target.closest("#dropzone"); if(zone) {event.preventDefault();zone.classList.remove("dragover");readUpload(event.dataTransfer.files?.[0]);} });
window.addEventListener("hashchange",setRoute);

async function boot() {
  document.getElementById("clock").textContent=new Intl.DateTimeFormat("en-GB",{day:"2-digit",month:"short",year:"numeric"}).format(new Date()).toUpperCase();
  try {
    const [catalog, overview]=await Promise.all([api("/api/catalog"),api("/api/overview")]);
    state.catalog=catalog; state.overview=overview;
    statusText.textContent=`${catalog.models.length} saved models available`;
    setRoute();
  } catch(error) {
    document.querySelector(".live-dot").classList.add("offline");
    statusText.textContent="Model service unavailable";
    page.innerHTML=`<div class="error-card"><strong>Could not load the observatory</strong><p>${escapeHtml(error.message)}</p><p>Check the API service and its model bundle, then try again.</p><button type="button" class="button button-dark" data-action="retry-boot">Retry connection</button></div>`;
  }
}

boot();
