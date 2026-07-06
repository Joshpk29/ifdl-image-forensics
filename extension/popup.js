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
