const queryEl = document.getElementById("query");
const contextEl = document.getElementById("context");
const ratioEl = document.getElementById("ratio");
const lastNEl = document.getElementById("lastN");
const keepEntitiesEl = document.getElementById("keepEntities");
const keepHeadingsEl = document.getElementById("keepHeadings");
const statusEl = document.getElementById("status");
const qualityBadgeEl = document.getElementById("qualityBadge");
const savingsBadgeEl = document.getElementById("savingsBadge");
const compressedTextEl = document.getElementById("compressedText");
const baselineGridEl = document.getElementById("baselineGrid");
const clusterMapEl = document.getElementById("clusterMap");
const clusterHintEl = document.getElementById("clusterHint");
const originalSpansEl = document.getElementById("originalSpans");
const tokenCoTextEl = document.getElementById("tokenCoText");
const tokenCoTokensEl = document.getElementById("tokenCoTokens");
const tokenCoMetaEl = document.getElementById("tokenCoMeta");
const inputTokensEl = document.getElementById("inputTokens");
const budgetValueEl = document.getElementById("budgetValue");
const coverageValueEl = document.getElementById("coverageValue");
const apiKeyEl = document.getElementById("apiKey");
const toggleKeyEl = document.getElementById("toggleKey");
const tokenCoOutputBlockEl = document.getElementById("tokenCoOutputBlock");
const scenarioGridEl = document.getElementById("scenarioGrid");
const heroSavingsEl = document.getElementById("heroSavings");
const heroCoverageEl = document.getElementById("heroCoverage");
const heroBudgetEl = document.getElementById("heroBudget");
const heroSpansEl = document.getElementById("heroSpans");
const statSpansEl = document.getElementById("statSpans");
const statGuardrailsEl = document.getElementById("statGuardrails");
const statHeadingsEl = document.getElementById("statHeadings");
const signalListEl = document.getElementById("signalList");
const statBudgetEl = document.getElementById("statBudget");
const statCoverageEl = document.getElementById("statCoverage");
const statSelectedEl = document.getElementById("statSelected");
const statParaphraseEl = document.getElementById("statParaphrase");
const statOutputTokensEl = document.getElementById("statOutputTokens");
const useSignalEl = document.getElementById("useSignal");
const signalBoostEl = document.getElementById("signalBoost");
const noveltyBoostEl = document.getElementById("noveltyBoost");
const signalBoostValueEl = document.getElementById("signalBoostValue");
const noveltyBoostValueEl = document.getElementById("noveltyBoostValue");
const paraphraseModeEl = document.getElementById("paraphraseMode");
const cosmosTokensEl = document.getElementById("cosmosTokens");
const cosmosMetaEl = document.getElementById("cosmosMeta");
const comparisonBannerEl = document.getElementById("comparisonBanner");
const comparisonTextEl = document.getElementById("comparisonText");
const impactChartEl = document.getElementById("impactChart");

const FALLBACK_SCENARIOS = [
  {
    category: "rag_overload",
    query: "What are the key risks and required controls for the payment rollout?",
    text: `Background: Payment API launch for EU and US merchants next quarter. Compliance and reliability are the top constraints.

Risk: duplicate charge bugs seen in prior beta. Impact: double billing, refunds, support overload. Control: require idempotency keys, run replay tests on critical flows, and block deploys if error rate >0.5%.

Risk: PCI scope creep. Some engineers are logging full PAN data in debug traces. Control: scrub logs, ship automated detectors, rotate secrets weekly, and gate any PCI adjacent change behind security review.

Risk: regional downtime. EU region had 14 minutes of downtime last month. Control: add health probes per region, include failover runbooks, and simulate region failover every sprint.

Risk: slow dispute handling. Support team needs a checklist for high-risk transactions. Control: route transactions over $5,000 to manual review and require supervisor sign-off.

Note: Marketing wants to highlight speed. Constraint: never sacrifice auditability. Keep evidence for every decision in the risk register.`,
  },
  {
    category: "meeting_transcript",
    query: "Summarize the decisions and owners from the meeting",
    text: `Moderator: Today's goals are to decide MVP scope and lock owners. We keep mentioning the same blockers so let's capture them cleanly.

Alice: The data connector repeats the same schema mapping note three times in docs; we only need the concise version. She will trim the copy and publish a sandbox endpoint by Friday.

Ben: Latency is the user-visible issue. He proposes caching embeddings for repeated spans and wants a quick synthetic benchmark to prove savings.

Chandra: Concerned about deleting rare facts. Suggests boosting spans with dates and counts so we do not lose SLAs. She will own an ablation chart with quality vs compression ratio.

Decision: MVP includes greedy facility location, novelty boost, entity protection, baseline comparisons, and one click demo page. Owners: Alice (docs), Ben (benchmark), Chandra (analysis), Dana (frontend polish).

Action: share demo recording and deploy to staging with a feature flag for compression.`,
  },
  {
    category: "policy_doc",
    query: "What onboarding requirements must a contractor follow?",
    text: `Section: Identity and access
Contractors must use single sign-on with hardware keys. Shared accounts are prohibited. Access requests require manager approval and expire after 90 days.

Section: Devices
Only managed laptops may access production data. Machines need full disk encryption and monthly patching. USB storage is blocked by default.

Section: Data handling
Sensitive data must stay within approved storage buckets. Never email secrets. Audit logs must be retained for one year. Incident reports are due within 24 hours.

Section: Offboarding
All accounts are revoked on end date. Hardware must be returned within seven days. Badge access is disabled immediately. Managers confirm data handoff.`,
  },
];

const CATEGORY_LABELS = {
  rag_overload: "RAG overload",
  meeting_transcript: "Meeting transcript",
  policy_doc: "Policy doc",
};

let scenarioDeck = [];
let selectedScenarioIndex = 0;

function countTokens(text) {
  const tokens = text.match(/[A-Za-z0-9']+/g) || [];
  return tokens.length;
}

function shorten(text, max = 80) {
  if (!text) return "";
  if (text.length <= max) return text;
  return `${text.slice(0, max - 3)}...`;
}

function formatCategory(category) {
  if (!category) return "Scenario";
  if (CATEGORY_LABELS[category]) return CATEGORY_LABELS[category];
  return category.replace(/_/g, " ");
}

function formatBaselineName(name) {
  if (!name) return "Baseline";
  return name
    .split("_")
    .map((chunk) => chunk.charAt(0).toUpperCase() + chunk.slice(1))
    .join(" ");
}

function normalizeScenario(raw, index) {
  const query = raw.query || "";
  const text = raw.text || "";
  const title = query ? shorten(query, 56) : shorten(text.split("\n")[0] || "Scenario", 56);
  const summary = shorten(text.replace(/\s+/g, " ").trim(), 92);
  return {
    id: raw.id ?? index,
    category: raw.category || "scenario",
    query,
    text,
    title,
    summary,
    tokens: countTokens(text),
  };
}

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#c14636" : "#536274";
}

function syncStats() {
  const orig = countTokens(contextEl.value);
  const aggressiveness = Number(ratioEl.value);
  inputTokensEl.textContent = `${orig} tokens`;
  qualityBadgeEl.textContent = aggressiveness.toFixed(2);
  signalBoostValueEl.textContent = Number(signalBoostEl.value).toFixed(2);
  noveltyBoostValueEl.textContent = Number(noveltyBoostEl.value).toFixed(2);
}

function updateMetricBar(metrics = {}, budget = null, spansKept = null, spansTotal = null, inputTokens = null) {
  if (!metrics) return;
  if (inputTokens !== null) {
    inputTokensEl.textContent = `${inputTokens} tokens`;
  }
  cosmosTokensEl.textContent = `${metrics.compressed_tokens ?? "-"} tokens`;
  cosmosMetaEl.textContent =
    metrics.savings_percent !== undefined
      ? `Saved ${metrics.savings_percent}% • Coverage ${metrics.coverage_score ?? "-"}`
      : "Saved - • Coverage -";
  const spanText = spansKept !== null && spansTotal !== null ? ` • spans ${spansKept}/${spansTotal}` : "";
  budgetValueEl.textContent = `Budget: ${budget ?? "-"}${spanText}`;
  coverageValueEl.textContent = `Coverage: ${metrics.coverage_score ?? "-"}`;
  savingsBadgeEl.textContent =
    metrics.savings_percent !== undefined
      ? `Saved ${metrics.savings_percent}% · ${metrics.compression_ratio ?? ""}x`
      : "Savings";
}

function updateHero(metrics, budget, spansKept, spansTotal) {
  heroSavingsEl.textContent = metrics?.savings_percent !== undefined ? `${metrics.savings_percent}%` : "-";
  heroCoverageEl.textContent = metrics?.coverage_score !== undefined ? metrics.coverage_score : "-";
  heroBudgetEl.textContent = budget ?? "-";
  heroSpansEl.textContent = spansKept !== null && spansTotal !== null ? `${spansKept}/${spansTotal}` : "-";
}

function updateProcessStats(spans, metrics, toggles, budget, spansKept, spansTotal) {
  const total = spans?.length || 0;
  const mustKeep = spans?.filter((s) => s.must_keep).length || 0;
  const headings = spans?.filter((s) => s.is_heading).length || 0;
  statSpansEl.textContent = `${total} spans`;
  statGuardrailsEl.textContent = `must-keep ${mustKeep}`;
  statHeadingsEl.textContent = `headings ${headings}`;
  statBudgetEl.textContent = `Budget: ${budget ?? "-"}`;
  statCoverageEl.textContent = `Coverage: ${metrics?.coverage_score ?? "-"}`;
  statSelectedEl.textContent = spansKept !== null && spansTotal !== null ? `Kept: ${spansKept}/${spansTotal}` : "Kept: -";
  statParaphraseEl.textContent = `Mode: ${toggles?.paraphrase_mode ?? "none"}`;
  statOutputTokensEl.textContent = `Output: ${metrics?.compressed_tokens ?? "-"}`;
}

function renderSignalList(spans) {
  signalListEl.innerHTML = "";
  if (!spans?.length) {
    const empty = document.createElement("div");
    empty.className = "signal-item muted";
    empty.textContent = "Run compression to see top signals.";
    signalListEl.appendChild(empty);
    return;
  }
  const sorted = [...spans]
    .filter((span) => span.weight !== undefined)
    .sort((a, b) => (b.weight || 0) - (a.weight || 0))
    .slice(0, 4);
  sorted.forEach((span) => {
    const div = document.createElement("div");
    div.className = "signal-item";
    const head = document.createElement("div");
    head.className = "signal-head";
    head.innerHTML = `<span>weight ${span.weight?.toFixed?.(2) ?? "-"}</span><span>${span.token_count ?? "-"} tokens</span>`;
    const text = document.createElement("div");
    text.className = "signal-text";
    const tags = [];
    if (span.must_keep) tags.push("must-keep");
    if (span.is_heading) tags.push("heading");
    if (span.is_question) tags.push("query");
    const tagText = tags.length ? `(${tags.join(", ")}) ` : "";
    text.textContent = `${tagText}${shorten(span.text, 120)}`;
    div.appendChild(head);
    div.appendChild(text);
    signalListEl.appendChild(div);
  });
}

function renderSpans(spans) {
  originalSpansEl.innerHTML = "";
  if (!spans?.length) {
    originalSpansEl.textContent = "No spans yet. Run compression.";
    return;
  }
  spans.forEach((span) => {
    const div = document.createElement("div");
    div.className = `span ${span.selected ? "selected" : ""}`;
    const tags = document.createElement("div");
    tags.className = "tags";
    if (span.must_keep) {
      const t = document.createElement("span");
      t.className = "tag";
      t.textContent = "must-keep";
      tags.appendChild(t);
    }
    if (span.is_heading) {
      const t = document.createElement("span");
      t.className = "tag";
      t.textContent = "heading";
      tags.appendChild(t);
    }
    if (span.is_question) {
      const t = document.createElement("span");
      t.className = "tag";
      t.textContent = "query";
      tags.appendChild(t);
    }
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.innerHTML = `<span>#${span.id}</span><span>${span.token_count}t</span><span>wt ${span.weight?.toFixed?.(2) ?? "-"}</span>${span.cluster !== null ? `<span>c${span.cluster}</span>` : ""}`;
    const text = document.createElement("div");
    text.className = "text";
    text.textContent = span.text;
    div.appendChild(tags);
    div.appendChild(meta);
    div.appendChild(text);
    originalSpansEl.appendChild(div);
  });
}

function renderBaselines(baselines) {
  baselineGridEl.innerHTML = "";
  (baselines || []).forEach((baseline) => {
    const card = document.createElement("div");
    card.className = "baseline-card";
    card.innerHTML = `<h5>${baseline.name}</h5>
      <div class="stat">tokens: ${baseline.metrics.compressed_tokens}</div>
      <div class="stat">savings: ${baseline.metrics.savings_percent}%</div>
      ${baseline.quality ? `<div class="stat">quality: ${baseline.quality}</div>` : ""}`;
    baselineGridEl.appendChild(card);
  });
}

function renderClusters(clusters, spans) {
  clusterMapEl.innerHTML = "";
  clusterHintEl.textContent = `${clusters.length} groups`;
  if (!clusters.length) {
    clusterMapEl.textContent = "Run compression to see redundancy clusters.";
    return;
  }
  clusters.forEach((cluster) => {
    const div = document.createElement("div");
    div.className = "cluster";
    const title = document.createElement("div");
    title.className = "title";
    title.textContent = `Cluster ${cluster.cluster}`;
    div.appendChild(title);
    const spanBadges = document.createElement("div");
    cluster.spans.forEach((sid) => {
      const spanText = spans.find((s) => s.id === sid)?.text || "";
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = `#${sid} ${shorten(spanText, 48)}`;
      spanBadges.appendChild(badge);
    });
    div.appendChild(spanBadges);
    clusterMapEl.appendChild(div);
  });
}

function extractTokencTokens(tokenc) {
  if (!tokenc) return null;
  return tokenc.output_tokens ?? tokenc.metrics?.compressed_tokens ?? null;
}

function extractTokencSavings(tokenc) {
  if (!tokenc) return null;
  const savings = tokenc.compression_percentage ?? tokenc.metrics?.savings_percent;
  return savings !== undefined && savings !== null ? Number(savings) : null;
}

function renderImpactChart(payload) {
  if (!impactChartEl) return;
  impactChartEl.innerHTML = "";
  if (!payload?.inputTokens || !payload?.cosmosTokens) {
    impactChartEl.textContent = "Run compression to see chart.";
    return;
  }

  const items = [];
  items.push({ label: "Input", tokens: payload.inputTokens, kind: "input", savings: null });
  items.push({ label: "COSMOS", tokens: payload.cosmosTokens, kind: "cosmos" });

  if (payload.tokencTokens) {
    items.push({ label: "TokenCo", tokens: payload.tokencTokens, kind: "tokenc" });
  }

  (payload.baselines || []).forEach((baseline) => {
    if (!baseline?.metrics?.compressed_tokens) return;
    items.push({
      label: formatBaselineName(baseline.name),
      tokens: baseline.metrics.compressed_tokens,
      kind: "baseline",
    });
  });

  const maxTokens = Math.max(...items.map((item) => item.tokens || 0), 1);

  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "impact-row";

    const label = document.createElement("div");
    label.className = "impact-label";
    label.textContent = item.label;

    const bar = document.createElement("div");
    bar.className = "impact-bar";
    const fill = document.createElement("div");
    const width = Math.max(6, (item.tokens / maxTokens) * 100);
    fill.className = `impact-fill ${item.kind}`;
    fill.style.width = `${width}%`;
    bar.appendChild(fill);

    const value = document.createElement("div");
    value.className = "impact-value";
    const savings =
      item.savings !== null && item.savings !== undefined
        ? item.savings
        : Math.max(0, (1 - item.tokens / payload.inputTokens) * 100);
    const savingsText = item.label === "Input" ? "" : ` • ${savings.toFixed(1)}%`;
    value.textContent = `${item.tokens}t${savingsText}`;

    row.appendChild(label);
    row.appendChild(bar);
    row.appendChild(value);
    impactChartEl.appendChild(row);
  });
}

function buildToggles() {
  return {
    keep_numbers_entities: keepEntitiesEl.checked,
    keep_headings: keepHeadingsEl.checked,
    keep_code_blocks: true,
    keep_role_markers: true,
    use_signal_scores: useSignalEl.checked,
    signal_boost: Number(signalBoostEl.value),
    novelty_boost: Number(noveltyBoostEl.value),
    paraphrase_mode: paraphraseModeEl.value,
  };
}

function updateComparisonBanner(cosmosMetrics, tokenc) {
  if (!comparisonBannerEl || !comparisonTextEl) return;
  if (!tokenc?.available) {
    comparisonBannerEl.hidden = true;
    return;
  }
  const tokencSavings = extractTokencSavings(tokenc);
  const cosmosSavings = cosmosMetrics?.savings_percent;
  if (tokencSavings !== null && cosmosSavings !== undefined) {
    const delta = cosmosSavings - tokencSavings;
    if (delta >= 0) {
      comparisonTextEl.textContent = `COSMOS saves ${delta.toFixed(2)}% more tokens than TokenCo.`;
    } else {
      comparisonTextEl.textContent = `TokenCo saves ${Math.abs(delta).toFixed(2)}% more tokens than COSMOS.`;
    }
    comparisonBannerEl.hidden = false;
    return;
  }
  comparisonTextEl.textContent = "Comparison available.";
  comparisonBannerEl.hidden = false;
}

function applyCosmosResults(data, toggles, tokenc = null) {
  renderSpans(data.spans || []);
  compressedTextEl.textContent = data.compressed_text || "No output yet.";
  renderBaselines(data.baselines);
  renderClusters(data.clusters || [], data.spans || []);
  updateMetricBar(
    data.metrics,
    data.budget,
    data.span_counts?.selected ?? null,
    data.span_counts?.total ?? null,
    data.input_tokens ?? countTokens(contextEl.value)
  );
  updateHero(data.metrics, data.budget, data.span_counts?.selected ?? null, data.span_counts?.total ?? null);
  updateProcessStats(
    data.spans || [],
    data.metrics,
    toggles,
    data.budget,
    data.span_counts?.selected ?? null,
    data.span_counts?.total ?? null
  );
  renderSignalList(data.spans || []);
  updateComparisonBanner(data.metrics, tokenc);
  renderImpactChart({
    inputTokens: data.input_tokens ?? countTokens(contextEl.value),
    cosmosTokens: data.metrics?.compressed_tokens ?? null,
    tokencTokens: tokenc?.available ? extractTokencTokens(tokenc) : null,
    baselines: data.baselines || [],
  });
}

function setButtonsDisabled(disabled) {
  document.getElementById("demoButton").disabled = disabled;
  document.getElementById("compareButton").disabled = disabled;
}

async function compressPrompt() {
  setStatus("Compressing with COSMOS...");
  tokenCoOutputBlockEl.hidden = true;
  comparisonBannerEl.hidden = true;
  setButtonsDisabled(true);
  const toggles = buildToggles();
  const payload = {
    text: contextEl.value,
    query: queryEl.value,
    target_ratio: 1 - Number(ratioEl.value),
    keep_last_n: Number(lastNEl.value),
    run_baselines: true,
    toggles,
  };
  try {
    const res = await fetch("/compress", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    applyCosmosResults(data, toggles, null);
    setStatus(`Done. Saved ${data.metrics?.savings_percent ?? "n/a"}% tokens.`);
  } catch (err) {
    console.error(err);
    setStatus("Compression failed", true);
  } finally {
    setButtonsDisabled(false);
  }
}

async function compareTokenc() {
  setStatus("Running comparison...");
  tokenCoOutputBlockEl.hidden = true;
  comparisonBannerEl.hidden = true;
  setButtonsDisabled(true);
  const toggles = buildToggles();
  const payload = {
    text: contextEl.value,
    query: queryEl.value,
    target_ratio: 1 - Number(ratioEl.value),
    aggressiveness: Number(ratioEl.value),
    api_key: apiKeyEl.value || null,
    toggles,
  };
  try {
    const res = await fetch("/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    const cosmos = data.cosmos;
    const tokenc = data.tokenc?.available ? data.tokenc : null;
    applyCosmosResults(cosmos, toggles, tokenc);
    if (tokenc) {
      tokenCoOutputBlockEl.hidden = false;
      tokenCoTokensEl.textContent = `${extractTokencTokens(tokenc) ?? "-"} tokens`;
      const savings = extractTokencSavings(tokenc);
      tokenCoMetaEl.textContent =
        savings !== null ? `Saved ${savings.toFixed(2)}%` : "Savings: n/a";
      tokenCoTextEl.textContent = tokenc.text || "No output received.";
    }
    setStatus(tokenc ? "Compared vs TokenCo" : "COSMOS done; TokenCo unavailable");
  } catch (err) {
    console.error(err);
    setStatus("Comparison failed", true);
  } finally {
    setButtonsDisabled(false);
  }
}

function renderScenarioDeck(scenarios) {
  scenarioGridEl.innerHTML = "";
  scenarios.forEach((scenario, index) => {
    const card = document.createElement("div");
    card.className = "scenario-card";
    card.setAttribute("role", "button");
    card.setAttribute("tabindex", "0");
    if (index === selectedScenarioIndex) {
      card.classList.add("active");
    }
    card.innerHTML = `
      <div class="scenario-tag">${formatCategory(scenario.category)}</div>
      <div class="scenario-title">${scenario.title}</div>
      <div class="scenario-meta">
        <span>${scenario.tokens} tokens</span>
        <span>${shorten(scenario.summary, 36)}</span>
      </div>
    `;
    card.addEventListener("click", () => selectScenario(index));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectScenario(index);
      }
    });
    scenarioGridEl.appendChild(card);
  });
}

function updateScenarioActive() {
  const cards = scenarioGridEl.querySelectorAll(".scenario-card");
  cards.forEach((card, index) => {
    if (index === selectedScenarioIndex) {
      card.classList.add("active");
    } else {
      card.classList.remove("active");
    }
  });
}

function selectScenario(index) {
  const scenario = scenarioDeck[index];
  if (!scenario) return;
  selectedScenarioIndex = index;
  queryEl.value = scenario.query || "";
  contextEl.value = scenario.text || "";
  syncStats();
  updateScenarioActive();
  setStatus(`Loaded scenario: ${scenario.title}`);
}

async function initScenarioDeck() {
  scenarioDeck = FALLBACK_SCENARIOS.map(normalizeScenario);
  renderScenarioDeck(scenarioDeck);
  try {
    const res = await fetch("/examples");
    if (res.ok) {
      const data = await res.json();
      if (data.examples?.length) {
        scenarioDeck = data.examples.map(normalizeScenario);
        renderScenarioDeck(scenarioDeck);
      }
    }
  } catch (err) {
    console.warn("Failed to load examples", err);
  }
  if (!contextEl.value && scenarioDeck.length) {
    selectScenario(0);
  }
}

function updateSignalControls() {
  const enabled = useSignalEl.checked;
  signalBoostEl.disabled = !enabled;
  signalBoostValueEl.textContent = Number(signalBoostEl.value).toFixed(2);
}

function updateNoveltyControls() {
  noveltyBoostValueEl.textContent = Number(noveltyBoostEl.value).toFixed(2);
}

toggleKeyEl.addEventListener("click", () => {
  const isPassword = apiKeyEl.type === "password";
  apiKeyEl.type = isPassword ? "text" : "password";
  toggleKeyEl.textContent = isPassword ? "Hide" : "Show";
  toggleKeyEl.setAttribute("aria-pressed", String(isPassword));
});

document.getElementById("demoButton").addEventListener("click", compressPrompt);
document.getElementById("compareButton").addEventListener("click", compareTokenc);

ratioEl.addEventListener("input", syncStats);
contextEl.addEventListener("input", syncStats);
signalBoostEl.addEventListener("input", updateSignalControls);
noveltyBoostEl.addEventListener("input", updateNoveltyControls);
useSignalEl.addEventListener("change", updateSignalControls);

syncStats();
updateSignalControls();
updateNoveltyControls();
initScenarioDeck();
