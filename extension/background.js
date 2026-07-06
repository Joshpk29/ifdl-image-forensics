// background.js — service worker
// Handles talking to the REST API so content scripts stay lightweight.

const DEFAULT_API_URL = "http://localhost:8000/analyze";

chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.sync.get(["apiUrl"], (res) => {
    if (!res.apiUrl) {
      chrome.storage.sync.set({ apiUrl: DEFAULT_API_URL });
    }
  });
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "ANALYZE_IMAGE") {
    analyzeImage(message.imageUrl)
      .then((result) => sendResponse({ ok: true, result }))
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true; // keep the message channel open for async response
  }
});

async function analyzeImage(imageUrl) {
  const { apiUrl } = await chrome.storage.sync.get(["apiUrl"]);
  const endpoint = apiUrl || DEFAULT_API_URL;

  const res = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_url: imageUrl }),
  });

  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }

  return res.json(); // { manipulated, confidence, localization_map_png_base64, models }
}
