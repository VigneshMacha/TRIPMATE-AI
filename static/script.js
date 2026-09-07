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
    return String(value ?? "").replace(/[&<>\"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[char]));
  }

  function inlineFormat(text) {
    let value = escapeHtml(text);
    value = value.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    value = value.replace(/\*(.+?)\*/g, "<em>$1</em>");
    value = value.replace(/`(.+?)`/g, "<code>$1</code>");
    return value;
  }

  function looksLikeDayHeading(line) {
    return /^(day\s+\d+|day\s+one|day\s+two|day\s+three|day\s+four|day\s+five|day\s+six|day\s+seven)(\s*[:\-–—]|\s*$)/i.test(line.trim());
  }

  function splitHeading(line) {
    const clean = line.replace(/^#{1,6}\s*/, "").replace(/^[-*]\s*/, "").trim();
    const match = clean.match(/^(day\s+\d+|day\s+one|day\s+two|day\s+three|day\s+four|day\s+five|day\s+six|day\s+seven)\s*[:\-–—]?\s*(.*)$/i);
    return match ? { label: match[1], title: match[2] || "Plan" } : null;
  }

  function renderItinerary(text, constraints = {}) {
    lastOutput = String(text || "");
    const rawLines = lastOutput.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
    if (!rawLines.length) {
      itineraryBox.className = "plan-canvas error-output";
      itineraryBox.textContent = "No itinerary was returned.";
      return;
    }

    const dayBlocks = [];
    let current = null;
    const intro = [];

    rawLines.forEach((line) => {
      const heading = splitHeading(line);
      if (heading && (looksLikeDayHeading(line) || /^#{1,6}\s*day\s+/i.test(line))) {
        if (current) dayBlocks.push(current);
        current = { label: heading.label, title: heading.title, lines: [] };
        return;
      }
      if (current) current.lines.push(line); else intro.push(line);
    });
    if (current) dayBlocks.push(current);

    const title = constraints.destination ? `A ${constraints.duration || "tailored"} trip to ${constraints.destination}` : "Your tailored trip";
    const subtitle = constraints.travel_style ? `${constraints.travel_style} · Built around your preferences` : "A practical draft built from the research available right now.";

    let html = `<article class="itinerary">
      <header class="itinerary-head">
        <div><h3 class="itinerary-title">${escapeHtml(title)}</h3><p class="itinerary-subtitle">${escapeHtml(subtitle)}</p></div>
        <span class="itinerary-stamp">TRIPMATE / DRAFT</span>
      </header>`;

    if (constraints.origin || constraints.budget || constraints.duration) {
      html += `<div class="plan-summary">
        ${constraints.origin ? `<div class="summary-card"><span>Starting from</span><strong>${escapeHtml(constraints.origin)}</strong></div>` : ""}
        ${constraints.duration ? `<div class="summary-card"><span>Time away</span><strong>${escapeHtml(constraints.duration)}</strong></div>` : ""}
        ${constraints.budget ? `<div class="summary-card"><span>Budget</span><strong>${escapeHtml(constraints.budget)}</strong></div>` : ""}
        ${constraints.travel_style ? `<div class="summary-card"><span>Style</span><strong>${escapeHtml(constraints.travel_style)}</strong></div>` : ""}
      </div>`;
    }

    if (intro.length) {
      const introText = intro.slice(0, 5).join(" ");
      html += `<div class="plan-section-title">Trip at a glance</div><p class="plan-paragraph">${inlineFormat(introText)}</p>`;
    }

    if (dayBlocks.length) {
      dayBlocks.forEach((day) => {
        const bullets = [];
        const paragraphs = [];
        day.lines.forEach(line => {
          if (/^[-*•]\s+/.test(line)) bullets.push(line.replace(/^[-*•]\s+/, ""));
          else if (/^\d+[.)]\s+/.test(line)) bullets.push(line.replace(/^\d+[.)]\s+/, ""));
          else paragraphs.push(line.replace(/^#{1,6}\s*/, ""));
        });
        html += `<section class="day-block"><div class="day-label">${escapeHtml(day.label.toUpperCase())}</div><div class="day-content"><h4>${escapeHtml(day.title || "Plan")}</h4>`;
        paragraphs.slice(0, 3).forEach(p => { html += `<p>${inlineFormat(p)}</p>`; });
        if (bullets.length) html += `<ul>${bullets.slice(0, 8).map(b => `<li>${inlineFormat(b)}</li>`).join("")}</ul>`;
        html += `</div></section>`;
      });
    } else {
      const chunks = lastOutput.split(/\n\s*\n/).map(x => x.trim()).filter(Boolean);
      chunks.slice(0, 10).forEach(chunk => {
        const lines = chunk.split(/\r?\n/).filter(Boolean);
        const heading = lines[0].replace(/^#{1,6}\s*/, "");
        const body = lines.slice(1).join(" ") || heading;
        html += `<div class="plan-section-title">${escapeHtml(heading.slice(0, 80))}</div><p class="plan-paragraph">${inlineFormat(body)}</p>`;
      });
    }

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

  function renderResearch(data) {
    const items = [
      ["Flight evidence", data.flight_results],
      ["Accommodation research", data.hotel_results],
      ["Weather", data.weather_results],
      ["Research synthesis", data.research_summary],
    ].filter(([, value]) => value && String(value).trim());

    if (!items.length) { researchPanel.classList.add("hidden"); return; }
    $("agentCount").textContent = `${(data.selected_agents || []).length || items.length} research streams`;
    $("researchGrid").innerHTML = items.map(([title, value]) => `<article class="research-card"><div class="research-title">${escapeHtml(title)}</div><p>${escapeHtml(String(value).slice(0, 1300))}</p></article>`).join("");
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
      itineraryBox.textContent = error.message;
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
