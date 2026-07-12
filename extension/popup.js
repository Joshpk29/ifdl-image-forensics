const apiUrlInput = document.getElementById("apiUrl");
const enabledInput = document.getElementById("enabled");
const status = document.getElementById("status");

chrome.storage.sync.get(["apiUrl", "enabled"], (res) => {
  apiUrlInput.value = res.apiUrl || "http://localhost:8000/analyze";
  enabledInput.checked = res.enabled !== false;
});

document.getElementById("save").addEventListener("click", () => {
  chrome.storage.sync.set(
    { apiUrl: apiUrlInput.value.trim(), enabled: enabledInput.checked },
    () => {
      status.textContent = "Saved.";
      setTimeout(() => (status.textContent = ""), 1500);
    }
  );
});

// "Why flagged" — SIDA's written explanation for the most recent flagged
// image, handed off from the content script via chrome.storage.local
// (see storeLastFlagged in content.js). Nothing to show if SIDA isn't
// running or nothing's been flagged yet, so the section stays hidden.
chrome.storage.local.get(["lastFlagged"], (res) => {
  const flagged = res.lastFlagged;
  if (!flagged || !flagged.explanation) return;

  const pct = Math.round((flagged.confidence || 0) * 100);
  const classLabel = flagged.explanationClass ? ` · ${flagged.explanationClass.replace("_", " ")}` : "";
  document.getElementById("lastFlaggedMeta").textContent =
    `${pct}% confidence${classLabel} · ${timeAgo(flagged.timestamp)}`;
  document.getElementById("lastFlaggedExplanation").textContent = flagged.explanation;
  document.getElementById("lastFlagged").classList.remove("hidden");
});

function timeAgo(timestamp) {
  if (!timestamp) return "";
  const seconds = Math.floor((Date.now() - timestamp) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}
