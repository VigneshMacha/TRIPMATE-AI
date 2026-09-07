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
  const threadBadge = $("activeThreadBadge");
  const itineraryBox = $("itineraryContainer");
  const hitlPanel = $("hitlPanel");
  const feedbackInput = $("reviewFeedback");
  const approveBtn = $("approveBtn");
  const rejectBtn = $("rejectBtn");
  const outputBadge = $("outputStateBadge");
  const graphStatusChip = $("graphStatusChip");
  const agentList = $("agentList");
  const planMeta = $("planMeta");

  const stepIds = [
    "step-guardrails",
    "step-supervisor",
    "step-specialists",
    "step-synthesis",
    "step-hitl",
  ];

  function showToast(message) {
    const toast = $("toast");
    toast.textContent = message;
    toast.classList.add("visible");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("visible"), 2600);
  }

  function setBadge(element, text, state = "idle") {
    element.textContent = text;
    element.className = `status-badge status-${state}`;
  }

  function setStep(stepId, state) {
    const el = $(stepId);
    if (!el) return;
    el.classList.remove("step-active", "step-completed");
    if (state === "active") el.classList.add("step-active");
    if (state === "done") el.classList.add("step-completed");
  }

  function resetSteps() {
    stepIds.forEach((id) => setStep(id, "idle"));
  }

  function renderAgents(agents = []) {
    agentList.innerHTML = "";
    if (!agents.length) {
      agentList.innerHTML = '<span class="muted">No specialist routing available</span>';
      return;
    }
    agents.forEach((agent) => {
      const chip = document.createElement("span");
      chip.className = "agent-chip";
      chip.textContent = agent.replace("_agent", "");
      agentList.appendChild(chip);
    });
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

  function showOutput(text, error = false) {
    lastOutput = String(text || "");
    itineraryBox.classList.remove("empty-state", "error-output");
    itineraryBox.textContent = lastOutput || "No itinerary was returned.";
    if (error) itineraryBox.classList.add("error-output");
  }

  function setBusy(busy, label = "Create travel plan") {
    submitBtn.disabled = busy;
    submitSpinner.classList.toggle("hidden", !busy);
    submitLabel.textContent = busy ? label : "Create travel plan";
  }

  async function readJson(response) {
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.success === false) {
      throw new Error(data.error || `Request failed with HTTP ${response.status}`);
    }
    return data;
  }

  function updateCharCount() {
    charCount.textContent = `${promptInput.value.length} / 8000`;
  }

  promptInput.addEventListener("input", updateCharCount);

  document.querySelectorAll("[data-prompt]").forEach((button) => {
    button.addEventListener("click", () => {
      promptInput.value = button.dataset.prompt || "";
      updateCharCount();
      promptInput.focus();
    });
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = promptInput.value.trim();
    if (!message) {
      showToast("Please enter a travel brief.");
      promptInput.focus();
      return;
    }

    const token = ++runToken;
    hitlPanel.classList.add("hidden");
    feedbackInput.value = "";
    resetSteps();
    setBusy(true, "Planning trip…");
    setBadge(graphStatusChip, "Running", "running");
    setBadge(outputBadge, "Processing", "running");
    setStep("step-guardrails", "active");
    showOutput("TripMate is validating your request and coordinating the selected research agents…");

    try {
      const response = await fetch("/api/travel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, thread_id: activeThreadId }),
      });
      const data = await readJson(response);
      if (token !== runToken) return;

      activeThreadId = data.thread_id || activeThreadId;
      threadBadge.textContent = activeThreadId ? activeThreadId.slice(0, 10) : "Idle";
      renderAgents(data.selected_agents || []);
      renderMeta(data.trip_constraints || {});
      showOutput(data.answer || data.itinerary || "No itinerary was returned.");

      setStep("step-guardrails", "done");
      setStep("step-supervisor", "done");
      setStep("step-specialists", "done");
      setStep("step-synthesis", "done");

      if (data.requires_approval) {
        setStep("step-hitl", "active");
        setBadge(graphStatusChip, "Checkpoint", "review");
        setBadge(outputBadge, "Awaiting review", "review");
        hitlPanel.classList.remove("hidden");
        feedbackInput.focus();
      } else {
        setStep("step-hitl", "done");
        setBadge(graphStatusChip, "Completed", "success");
        setBadge(outputBadge, "Final", "success");
      }
    } catch (error) {
      if (token !== runToken) return;
      showOutput(error.message, true);
      setBadge(graphStatusChip, "Failed", "error");
      setBadge(outputBadge, "Execution failed", "error");
      showToast(error.message);
    } finally {
      if (token === runToken) setBusy(false);
    }
  });

  async function submitDecision(approved) {
    if (!activeThreadId) {
      showToast("There is no active trip to review.");
      return;
    }

    const feedback = feedbackInput.value.trim();
    if (!approved && !feedback) {
      showToast("Add revision notes before requesting changes.");
      feedbackInput.focus();
      return;
    }

    const token = ++runToken;
    approveBtn.disabled = true;
    rejectBtn.disabled = true;
    setBadge(graphStatusChip, approved ? "Finalizing" : "Revising", "running");
    setBadge(outputBadge, approved ? "Finalizing" : "Regenerating", "running");
    hitlPanel.classList.add("hidden");

    try {
      const response = await fetch("/api/travel/approve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ thread_id: activeThreadId, approved, feedback }),
      });
      const data = await readJson(response);
      if (token !== runToken) return;

      showOutput(data.answer || data.itinerary || "No response was returned.");
      renderAgents(data.selected_agents || []);
      renderMeta(data.trip_constraints || {});

      if (data.requires_approval) {
        setStep("step-hitl", "active");
        setBadge(graphStatusChip, "Checkpoint", "review");
        setBadge(outputBadge, "Awaiting review", "review");
        hitlPanel.classList.remove("hidden");
        feedbackInput.value = "";
        feedbackInput.focus();
        showToast("The itinerary was regenerated. Review the new draft.");
      } else {
        setStep("step-hitl", "done");
        setBadge(graphStatusChip, "Completed", "success");
        setBadge(outputBadge, "Final", "success");
        feedbackInput.value = "";
        showToast(approved ? "Trip plan finalized." : "Revision completed.");
      }
    } catch (error) {
      if (token !== runToken) return;
      setBadge(graphStatusChip, "Failed", "error");
      setBadge(outputBadge, "Review failed", "error");
      hitlPanel.classList.remove("hidden");
      showToast(error.message);
    } finally {
      if (token === runToken) {
        approveBtn.disabled = false;
        rejectBtn.disabled = false;
      }
    }
  }

  approveBtn.addEventListener("click", () => submitDecision(true));
  rejectBtn.addEventListener("click", () => submitDecision(false));

  $("copyOutputBtn").addEventListener("click", async () => {
    if (!lastOutput.trim()) {
      showToast("There is no itinerary to copy yet.");
      return;
    }
    try {
      await navigator.clipboard.writeText(lastOutput);
      showToast("Itinerary copied to clipboard.");
    } catch {
      showToast("Clipboard access is unavailable in this browser.");
    }
  });

  $("resetThreadBtn").addEventListener("click", () => {
    runToken++;
    activeThreadId = null;
    lastOutput = "";
    promptInput.value = "";
    feedbackInput.value = "";
    threadBadge.textContent = "Idle";
    agentList.innerHTML = '<span class="muted">No run yet</span>';
    planMeta.classList.add("hidden");
    hitlPanel.classList.add("hidden");
    resetSteps();
    setBadge(graphStatusChip, "Idle", "idle");
    setBadge(outputBadge, "Waiting", "idle");
    itineraryBox.className = "output-content empty-state";
    itineraryBox.innerHTML = '<div class="empty-icon">✦</div><h3>Your itinerary will appear here</h3><p>Submit a travel brief to start the multi-agent planning workflow.</p>';
    updateCharCount();
    showToast("Ready for a new trip.");
  });

  updateCharCount();
});
