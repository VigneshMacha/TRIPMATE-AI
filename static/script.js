document.addEventListener("DOMContentLoaded", () => {
  let activeThreadId = null;

  const form = document.getElementById("travelForm");
  const promptInput = document.getElementById("userPrompt");
  const submitBtn = document.getElementById("submitBtn");
  const submitSpinner = document.getElementById("submitSpinner");
  const threadBadge = document.getElementById("activeThreadBadge");
  const itineraryBox = document.getElementById("itineraryContainer");
  const hitlPanel = document.getElementById("hitlPanel");
  const feedbackInput = document.getElementById("reviewFeedback");
  const approveBtn = document.getElementById("approveBtn");
  const rejectBtn = document.getElementById("rejectBtn");
  const outputBadge = document.getElementById("outputStateBadge");
  const graphStatusChip = document.getElementById("graphStatusChip");

  function setStep(stepId, state) {
    const el = document.getElementById(stepId);
    if (!el) return;
    el.classList.remove("step-active", "step-completed");
    if (state === "active") el.classList.add("step-active");
    if (state === "done") el.classList.add("step-completed");
  }

  function resetSteps() {
    ["step-guardrails", "step-supervisor", "step-mcp", "step-hitl"].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.classList.remove("step-active", "step-completed");
    });
  }

  // Handle plan dispatch
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const message = promptInput.value.trim();
    if (!message) return;

    submitBtn.disabled = true;
    submitSpinner.classList.remove("hidden");
    hitlPanel.classList.add("hidden");
    resetSteps();

    setStep("step-guardrails", "active");
    graphStatusChip.innerText = "RUNNING";
    outputBadge.innerText = "Processing...";

    try {
      // Simulate trace transitions
      setTimeout(() => {
        setStep("step-guardrails", "done");
        setStep("step-supervisor", "active");
      }, 400);

      setTimeout(() => {
        setStep("step-supervisor", "done");
        setStep("step-mcp", "active");
      }, 800);

      const response = await fetch("/api/travel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, thread_id: activeThreadId })
      });

      const data = await response.json();

      if (data.thread_id) {
        activeThreadId = data.thread_id;
        threadBadge.innerText = activeThreadId.slice(0, 8);
      }

      setStep("step-mcp", "done");
      setStep("step-hitl", "active");

      itineraryBox.textContent = data.response || JSON.stringify(data, null, 2);
      outputBadge.innerText = "Awaiting Approval";
      outputBadge.className = "text-[10px] font-mono px-2 py-0.5 rounded bg-amber-100 text-amber-800";
      graphStatusChip.innerText = "CHECKPOINT";

      // Show HITL panel for supervisor review
      hitlPanel.classList.remove("hidden");
    } catch (err) {
      itineraryBox.textContent = `Error executing graph: ${err.message}`;
      outputBadge.innerText = "Execution Failed";
      outputBadge.className = "text-[10px] font-mono px-2 py-0.5 rounded bg-rose-100 text-rose-800";
    } finally {
      submitBtn.disabled = false;
      submitSpinner.classList.add("hidden");
    }
  });

  // Handle HITL Decision (Approve or Reject/Revise)
  async function submitDecision(approved) {
    if (!activeThreadId) return;

    const feedback = feedbackInput.value.trim();
    outputBadge.innerText = approved ? "Finalizing..." : "Requesting Revision...";
    hitlPanel.classList.add("hidden");

    try {
      const res = await fetch("/api/travel/approve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          thread_id: activeThreadId,
          approved,
          feedback
        })
      });

      const result = await res.json();
      itineraryBox.textContent = result.response || JSON.stringify(result, null, 2);

      if (approved) {
        setStep("step-hitl", "done");
        outputBadge.innerText = "Approved";
        outputBadge.className = "text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-100 text-emerald-800";
        graphStatusChip.innerText = "COMPLETED";
      } else {
        outputBadge.innerText = "Revision In Progress";
        outputBadge.className = "text-[10px] font-mono px-2 py-0.5 rounded bg-blue-100 text-blue-800";
        graphStatusChip.innerText = "RE-ROUTED";
      }
    } catch (err) {
      itineraryBox.textContent = `HITL processing error: ${err.message}`;
    }
  }

  approveBtn.addEventListener("click", () => submitDecision(true));
  rejectBtn.addEventListener("click", () => submitDecision(false));

  document.getElementById("resetThreadBtn").addEventListener("click", () => {
    activeThreadId = null;
    threadBadge.innerText = "idle";
    promptInput.value = "";
    feedbackInput.value = "";
    itineraryBox.textContent = "Plan output will render here once the multi-agent graph completes orchestration...";
    hitlPanel.classList.add("hidden");
    outputBadge.innerText = "No active run";
    outputBadge.className = "text-[10px] font-mono px-2 py-0.5 rounded bg-slate-100 text-slate-500";
    graphStatusChip.innerText = "IDLE";
    resetSteps();
  });
});