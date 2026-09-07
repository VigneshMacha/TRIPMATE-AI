document.addEventListener("DOMContentLoaded", () => {
  let activeThreadId = null;
  let lastOutput = "";
  let runToken = 0;
  let toastTimer = null;

  const $ = (id) => document.getElementById(id);
  const form = $("travelForm");
  const promptInput = $("userPrompt");
  const submitBtn = $("submitBtn");
  const submitSpinner = $("submitSpinner");
  const submitLabel = $("submitLabel");
  const charCount = $("charCount");
  const outputBadge = $("outputStateBadge");
  const itineraryBox = $("itineraryContainer");
  const hitlPanel = $("hitlPanel");
  const researchPanel = $("researchPanel");
  const feedbackInput = $("reviewFeedback");
  const approveBtn = $("approveBtn");
  const rejectBtn = $("rejectBtn");
  const planMeta = $("planMeta");

  function showToast(message) {
    const toast = $("toast");
    toast.textContent = message;
    toast.classList.add("visible");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("visible"), 2800);
  }

  function setState(text, state = "idle") {
    outputBadge.textContent = text;
    outputBadge.className = `state-pill ${state}`;
  }

  function setBusy(busy, label = "Build my trip") {
    submitBtn.disabled = busy;
    submitSpinner.classList.toggle("hidden", !busy);
    submitLabel.textContent = busy ? label : "Build my trip";
  }

  function updateCharCount() {
    charCount.textContent = `${promptInput.value.length} / 8000`;
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>\"]/g, (char) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"
    }[char]));
  }

  function inlineFormat(text) {
    let value = escapeHtml(text);
    value = value.replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    value = value.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    value = value.replace(/__(.+?)__/g, "<strong>$1</strong>");
    value = value.replace(/`(.+?)`/g, "<code>$1</code>");
    value = value.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    return value;
  }

  function cleanLine(line) {
    return String(line || "")
      .replace(/^\s*```(?:markdown|md|text)?\s*$/i, "")
      .replace(/^\s*```\s*$/i, "")
      .replace(/\\n/g, "\n")
      .trim();
  }

  function isHorizontalRule(line) {
    return /^\s*([-*_]\s*){3,}$/.test(line);
  }

  function headingInfo(line) {
    const clean = line.replace(/^\s*#{1,6}\s*/, "").trim();
    const numbered = clean.match(/^\d+[.)]\s+(.+)$/);
    if (numbered) return { level: 2, title: numbered[1].trim() };
    if (/^day\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b/i.test(clean)) {
      const match = clean.match(/^(day\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten))\s*[:\-–—]?\s*(.*)$/i);
      return { level: 2, title: match ? `${match[1]}${match[2] ? ` — ${match[2]}` : ""}` : clean };
    }
    if (/^[A-Z][A-Z0-9 &'’–—-]{4,}$/.test(clean) && clean.length < 70) return { level: 2, title: clean };
    if (/^#{1,6}\s*/.test(line)) return { level: 2, title: clean };
    return null;
  }

  function isTableLine(line) {
    return /^\s*\|.*\|\s*$/.test(line) || /^\s*[^|]+\|[^|]+\|/.test(line);
  }

  function isTableSeparator(line) {
    const cells = line.replace(/^\||\|$/g, "").split("|").map(x => x.trim());
    return cells.length >= 2 && cells.every(cell => /^:?-{3,}:?$/.test(cell));
  }

  function splitTableRow(line) {
    let value = line.trim();
    if (value.startsWith("|")) value = value.slice(1);
    if (value.endsWith("|")) value = value.slice(0, -1);
    return value.split("|").map(cell => cell.trim());
  }

  function parseBlocks(text) {
    const lines = String(text || "")
      .replace(/\r/g, "")
      .replace(/\\n/g, "\n")
      .split("\n")
      .map(cleanLine)
      .filter(Boolean)
      .filter(line => !isHorizontalRule(line));

    const blocks = [];
    let i = 0;

    while (i < lines.length) {
      const line = lines[i];

      // Markdown tables become real tables instead of literal pipe text.
      if (isTableLine(line) && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
        const headers = splitTableRow(line);
        const rows = [];
        i += 2;
        while (i < lines.length && isTableLine(lines[i]) && !headingInfo(lines[i])) {
          rows.push(splitTableRow(lines[i]));
          i++;
        }
        blocks.push({ type: "table", headers, rows });
        continue;
      }

      const heading = headingInfo(line);
      if (heading) {
        blocks.push({ type: "heading", title: heading.title });
        i++;
        continue;
      }

      if (/^[-*•]\s+/.test(line) || /^\d+[.)]\s+/.test(line)) {
        const items = [];
        while (i < lines.length && (/^[-*•]\s+/.test(lines[i]) || /^\d+[.)]\s+/.test(lines[i]))) {
          items.push(lines[i].replace(/^[-*•]\s+/, "").replace(/^\d+[.)]\s+/, ""));
          i++;
        }
        blocks.push({ type: "list", items });
        continue;
      }

      // Ignore common markdown title decoration / duplicate title lines.
      if (/^tripmate\s*[-–—:]/i.test(line)) {
        i++;
        continue;
      }

      const paragraph = [line];
      i++;
      while (i < lines.length && !headingInfo(lines[i]) && !isTableLine(lines[i]) && !/^[-*•]\s+/.test(lines[i]) && !/^\d+[.)]\s+/.test(lines[i])) {
        paragraph.push(lines[i]);
        i++;
      }
      blocks.push({ type: "paragraph", text: paragraph.join(" ") });
    }
    return blocks;
  }

  function renderTable(headers, rows) {
    const safeHeaders = headers.length ? headers : ["Details"];
    const normalizedRows = rows.map(row => {
      const copy = row.slice(0, safeHeaders.length);
      while (copy.length < safeHeaders.length) copy.push("");
      return copy;
    });
    return `<div class="plan-table-wrap"><table class="plan-table"><thead><tr>${safeHeaders.map(h => `<th>${inlineFormat(h)}</th>`).join("")}</tr></thead><tbody>${normalizedRows.map(row => `<tr>${row.map(cell => `<td>${inlineFormat(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  }

  function renderItinerary(text, constraints = {}) {
    lastOutput = String(text || "");
    const blocks = parseBlocks(lastOutput);
    if (!blocks.length) {
      itineraryBox.className = "plan-canvas error-output";
      itineraryBox.textContent = "No itinerary was returned.";
      return;
    }

    const title = constraints.destination
      ? `${constraints.duration || "Tailored"} in ${constraints.destination}`
      : "Your tailored trip";
    const subtitle = constraints.travel_style
      ? `${constraints.travel_style} · Built around your preferences`
      : "A practical draft built from the research available right now.";

    let html = `<article class="itinerary">
      <header class="itinerary-head">
        <div>
          <p class="itinerary-kicker">YOUR TRIP / DRAFT</p>
          <h3 class="itinerary-title">${escapeHtml(title)}</h3>
          <p class="itinerary-subtitle">${escapeHtml(subtitle)}</p>
        </div>
        <button class="print-button" type="button" onclick="window.print()">Print</button>
      </header>`;

    if (constraints.origin || constraints.budget || constraints.duration || constraints.travel_style) {
      html += `<div class="plan-summary">
        ${constraints.origin ? `<div class="summary-card"><span>From</span><strong>${escapeHtml(constraints.origin)}</strong></div>` : ""}
        ${constraints.duration ? `<div class="summary-card"><span>Duration</span><strong>${escapeHtml(constraints.duration)}</strong></div>` : ""}
        ${constraints.budget ? `<div class="summary-card"><span>Budget</span><strong>${escapeHtml(constraints.budget)}</strong></div>` : ""}
        ${constraints.travel_style ? `<div class="summary-card"><span>Style</span><strong>${escapeHtml(constraints.travel_style)}</strong></div>` : ""}
      </div>`;
    }

    let sectionOpen = false;
    let firstHeadingSkipped = false;

    const closeSection = () => {
      if (sectionOpen) {
        html += `</div></section>`;
        sectionOpen = false;
      }
    };

    blocks.forEach((block) => {
      if (block.type === "heading") {
        const normalized = block.title.replace(/[*_]/g, "").trim();

        // The model often repeats its title before the actual content.
        if (!firstHeadingSkipped && /(?:tripmate|itinerary|trip summary|\bday\s*\d+)/i.test(normalized) && normalized.length < 110) {
          firstHeadingSkipped = true;
          return;
        }

        closeSection();
        const dayMatch = normalized.match(/^day\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[—:\-]?\s*(.*)$/i);
        if (dayMatch) {
          const dayNumber = dayMatch[1];
          const dayTitle = dayMatch[2] || "Plan";
          html += `<section class="day-block"><div class="day-label">DAY ${escapeHtml(dayNumber)}</div><div class="day-content"><h4>${escapeHtml(dayTitle)}</h4>`;
        } else {
          html += `<section class="content-section"><div class="content-section-heading"><span></span><h4>${escapeHtml(normalized)}</h4></div><div class="content-section-body">`;
        }
        sectionOpen = true;
        return;
      }

      if (block.type === "paragraph") {
        if (!sectionOpen) {
          html += `<section class="content-section"><div class="content-section-heading"><span></span><h4>Trip overview</h4></div><div class="content-section-body">`;
          sectionOpen = true;
        }
        html += `<p class="plan-paragraph">${inlineFormat(block.text)}</p>`;
      } else if (block.type === "list") {
        if (!sectionOpen) {
          html += `<section class="content-section"><div class="content-section-heading"><span></span><h4>Plan details</h4></div><div class="content-section-body">`;
          sectionOpen = true;
        }
        html += `<ul class="plan-list">${block.items.map(item => `<li>${inlineFormat(item)}</li>`).join("")}</ul>`;
      } else if (block.type === "table") {
        if (!sectionOpen) {
          html += `<section class="content-section"><div class="content-section-heading"><span></span><h4>Details</h4></div><div class="content-section-body">`;
          sectionOpen = true;
        }
        html += renderTable(block.headers, block.rows);
      }
    });

    closeSection();
    html += `</article>`;

    itineraryBox.className = "plan-canvas";
    itineraryBox.innerHTML = html;
  }

  function renderMeta(constraints = {}) {
    const values = {
      metaDestination: constraints.destination || "Not specified",
      metaDuration: constraints.duration || "Not specified",
      metaBudget: constraints.budget || "Not specified",
      metaStyle: constraints.travel_style || "Not specified",
    };
    Object.entries(values).forEach(([id, value]) => $(id).textContent = value);
    planMeta.classList.remove("hidden");
  }

  function prettifyResearch(value) {
    let text = String(value || "").trim();
    if (!text) return "";
    text = text.replace(/\\n/g, "\n").replace(/\\"/g, '"');
    try {
      const parsed = JSON.parse(text);
      if (parsed && typeof parsed === "object") {
        const useful = parsed.results || parsed.forecast || parsed.current || parsed.data || parsed;
        return JSON.stringify(useful, null, 2);
      }
    } catch (_) {}
    return text;
  }

  function researchExcerpt(value, limit = 850) {
    const text = prettifyResearch(value);
    return text.length > limit ? `${text.slice(0, limit).trim()}…` : text;
  }

  function renderResearch(data) {
    const items = [
      ["Flight evidence", data.flight_results, "Transport"],
      ["Accommodation research", data.hotel_results, "Stays"],
      ["Weather", data.weather_results, "Conditions"],
      ["Research synthesis", data.research_summary, "Sources"],
    ].filter(([, value]) => value && String(value).trim());

    if (!items.length) {
      researchPanel.classList.add("hidden");
      return;
    }

    $("agentCount").textContent = `${items.length} research streams`;
    $("researchGrid").innerHTML = items.map(([title, value, tag], index) => {
      const text = researchExcerpt(value);
      const lines = text.split("\n").filter(Boolean);
      const preview = lines.slice(0, 7).join("\n");
      return `<article class="research-card">
        <div class="research-card-top"><span class="research-tag">${escapeHtml(tag)}</span><span class="research-number">0${index + 1}</span></div>
        <h4>${escapeHtml(title)}</h4>
        <p>${escapeHtml(preview)}</p>
        <details><summary>View evidence</summary><pre>${escapeHtml(text)}</pre></details>
      </article>`;
    }).join("");
    researchPanel.classList.remove("hidden");
  }

  async function readJson(response) {
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.success === false) throw new Error(data.error || `Request failed with HTTP ${response.status}`);
    return data;
  }

  document.querySelectorAll("[data-prompt]").forEach(button => button.addEventListener("click", () => {
    promptInput.value = button.dataset.prompt || "";
    updateCharCount();
    promptInput.focus();
  }));
  promptInput.addEventListener("input", updateCharCount);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = promptInput.value.trim();
    if (!message) { showToast("Start with a few details about the trip."); promptInput.focus(); return; }

    const token = ++runToken;
    hitlPanel.classList.add("hidden");
    researchPanel.classList.add("hidden");
    feedbackInput.value = "";
    setBusy(true, "Researching your trip…");
    setState("Researching", "running");
    itineraryBox.className = "plan-canvas empty-state";
    itineraryBox.innerHTML = `<div class="empty-scenery" aria-hidden="true"><span></span><i></i><b></b></div><p class="canvas-kicker">WORKING IN PARALLEL</p><h3>Finding the pieces<br>that matter.</h3><p>Flights, stays, weather and budget evidence are being gathered together.</p>`;

    try {
      const response = await fetch("/api/travel", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message, thread_id: activeThreadId }) });
      const data = await readJson(response);
      if (token !== runToken) return;
      activeThreadId = data.thread_id || activeThreadId;
      renderMeta(data.trip_constraints || {});
      renderItinerary(data.answer || data.itinerary || "No itinerary was returned.", data.trip_constraints || {});
      renderResearch(data);

      if (data.requires_approval) {
        setState("Ready for your review", "review");
        hitlPanel.classList.remove("hidden");
        setTimeout(() => hitlPanel.scrollIntoView({ behavior: "smooth", block: "center" }), 100);
      } else {
        setState("Plan ready", "success");
      }
    } catch (error) {
      if (token !== runToken) return;
      itineraryBox.className = "plan-canvas error-output";
      itineraryBox.innerHTML = `<div class="error-inner"><span>Something went wrong</span><h3>We couldn't finish the trip.</h3><p>${escapeHtml(error.message)}</p></div>`;
      setState("Could not complete", "error");
      showToast(error.message);
    } finally {
      if (token === runToken) setBusy(false);
    }
  });

  async function submitDecision(approved) {
    if (!activeThreadId) { showToast("There is no active trip to review."); return; }
    const feedback = feedbackInput.value.trim();
    if (!approved && !feedback) { showToast("Tell us what you'd like changed."); feedbackInput.focus(); return; }
    const token = ++runToken;
    approveBtn.disabled = true; rejectBtn.disabled = true;
    setState(approved ? "Finalizing" : "Regenerating", "running");
    hitlPanel.classList.add("hidden");

    try {
      const response = await fetch("/api/travel/approve", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ thread_id: activeThreadId, approved, feedback }) });
      const data = await readJson(response);
      if (token !== runToken) return;
      renderMeta(data.trip_constraints || {});
      renderItinerary(data.answer || data.itinerary || "No response was returned.", data.trip_constraints || {});
      renderResearch(data);
      if (data.requires_approval) {
        setState("Ready for your review", "review");
        hitlPanel.classList.remove("hidden");
        feedbackInput.value = "";
        showToast("New draft ready. Have another look.");
      } else {
        setState("Trip finalized", "success");
        feedbackInput.value = "";
        showToast(approved ? "Your trip is finalized." : "Revision completed.");
      }
    } catch (error) {
      if (token !== runToken) return;
      setState("Review failed", "error");
      hitlPanel.classList.remove("hidden");
      showToast(error.message);
    } finally {
      if (token === runToken) { approveBtn.disabled = false; rejectBtn.disabled = false; }
    }
  }

  approveBtn.addEventListener("click", () => submitDecision(true));
  rejectBtn.addEventListener("click", () => submitDecision(false));

  $("copyOutputBtn").addEventListener("click", async () => {
    if (!lastOutput.trim()) { showToast("There is no plan to copy yet."); return; }
    try { await navigator.clipboard.writeText(lastOutput); showToast("Plan copied to clipboard."); }
    catch { showToast("Clipboard access is unavailable here."); }
  });

  $("resetThreadBtn").addEventListener("click", () => {
    runToken++;
    activeThreadId = null; lastOutput = "";
    promptInput.value = ""; feedbackInput.value = "";
    planMeta.classList.add("hidden"); hitlPanel.classList.add("hidden"); researchPanel.classList.add("hidden");
    setState("Ready", "idle");
    itineraryBox.className = "plan-canvas empty-state";
    itineraryBox.innerHTML = `<div class="empty-scenery" aria-hidden="true"><span></span><i></i><b></b></div><p class="canvas-kicker">READY WHEN YOU ARE</p><h3>Start with a feeling,<br>not a spreadsheet.</h3><p>Give us the rough idea. TripMate will work out the practical pieces around it.</p>`;
    updateCharCount();
    showToast("Fresh trip, fresh start.");
  });

  updateCharCount();
});
