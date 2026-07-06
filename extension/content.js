// content.js — runs on the feed page
// Finds feed images, requests analysis from the background worker, and
// overlays a score badge + click-to-reveal localization heatmap on each one.
//
// Overlays are kept OUT of the page's own DOM tree (appended to
// document.body, position: fixed, tracked to the image's live bounding
// rect) rather than wrapping the <img> itself. Feed sites like Instagram/
// Twitter are React-driven and aggressively reconcile the DOM — wrapping
// their <img> elements risks breaking the site's own rendering. Tracking
// position externally is more code but much safer.

const processed = new WeakSet();
// entry: { img, container, badge, overlayImg, result, mapVisible }
const entries = [];

function scanForImages() {
  const images = document.querySelectorAll("img");
  images.forEach((img) => {
    if (processed.has(img)) return;
    if (img.naturalWidth < 150 || img.naturalHeight < 150) return; // skip icons/avatars
    processed.add(img);
    queueAnalysis(img);
  });
}

function queueAnalysis(img) {
  const src = img.currentSrc || img.src;
  if (!src) return;

  chrome.runtime.sendMessage(
    { type: "ANALYZE_IMAGE", imageUrl: src },
    (response) => {
      if (!response || !response.ok) return;
      applyOverlay(img, response.result);
    }
  );
}

function applyOverlay(img, result) {
  if (!result) return;
  if (!img.isConnected) return; // image was removed before the analysis came back

  const container = document.createElement("div");
  container.className = "ifdl-container";

  const badge = document.createElement("button");
  badge.type = "button";
  badge.className = "ifdl-badge";
  badge.classList.add(result.manipulated ? "ifdl-badge-flagged" : "ifdl-badge-clear");
  const pct = Math.round((result.confidence || 0) * 100);
  badge.textContent = result.manipulated ? `⚠ ${pct}%` : `${pct}%`;
  badge.title = result.manipulated
    ? `Likely manipulated (${pct}% confidence). Click to see where.`
    : `Low manipulation confidence (${pct}%). Click to see the map anyway.`;

  let overlayImg = null;
  if (result.localization_map_png_base64) {
    overlayImg = document.createElement("img");
    overlayImg.className = "ifdl-heatmap";
    overlayImg.src = `data:image/png;base64,${result.localization_map_png_base64}`;
    overlayImg.alt = "";
  }

  badge.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!overlayImg) return;
    entry.mapVisible = !entry.mapVisible;
    overlayImg.classList.toggle("ifdl-heatmap-visible", entry.mapVisible);
  });

  container.appendChild(badge);
  if (overlayImg) container.appendChild(overlayImg);
  document.body.appendChild(container);

  const entry = { img, container, badge, overlayImg, result, mapVisible: false };
  entries.push(entry);
  positionEntry(entry);
}

function positionEntry(entry) {
  if (!entry.img.isConnected) {
    entry.container.remove();
    return false;
  }
  const rect = entry.img.getBoundingClientRect();
  // Image scrolled/resized out of any reasonable viewport area — hide rather
  // than leave a stale badge floating over unrelated content.
  if (rect.width === 0 || rect.height === 0) {
    entry.container.style.display = "none";
    return true;
  }
  entry.container.style.display = "";
  entry.container.style.top = `${rect.top}px`;
  entry.container.style.left = `${rect.left}px`;
  entry.container.style.width = `${rect.width}px`;
  entry.container.style.height = `${rect.height}px`;
  return true;
}

function positionAllEntries() {
  for (let i = entries.length - 1; i >= 0; i--) {
    const stillConnected = positionEntry(entries[i]);
    if (!stillConnected) entries.splice(i, 1);
  }
}

// Reposition on scroll (capture:true catches scroll inside nested feed
// containers, not just window-level scroll) and on resize.
window.addEventListener("scroll", positionAllEntries, { passive: true, capture: true });
window.addEventListener("resize", positionAllEntries, { passive: true });
// Cheap fallback for layout shifts that don't fire scroll/resize (e.g. new
// posts inserted above the current one, shifting everything down).
setInterval(positionAllEntries, 500);

// Observe DOM changes for infinite-scroll feeds
const observer = new MutationObserver(() => {
  scanForImages();
  positionAllEntries();
});
observer.observe(document.body, { childList: true, subtree: true });

scanForImages();
