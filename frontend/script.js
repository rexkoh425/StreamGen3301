const DEFAULT_BACKEND = "http://localhost:8000";
const STORAGE_KEY = "pi-camera-backend-url";

const backendInput = document.querySelector("#backend-url");
const connectBtn = document.querySelector("#connect");
const startBtn = document.querySelector("#start");
const stopBtn = document.querySelector("#stop");
const refreshStatusBtn = document.querySelector("#refresh-status");
const statusEl = document.querySelector("#status");
const streamImg = document.querySelector("#stream");
const toggleStreamBtn = document.querySelector("#toggle-stream");
const captureFrameBtn = document.querySelector("#capture-frame");
const framePreview = document.querySelector("#frame-preview");

let isStreamVisible = true;
let statusTimer = null;

function loadStoredBackend() {
  const stored = window.localStorage.getItem(STORAGE_KEY);
  backendInput.value = stored || DEFAULT_BACKEND;
}

function saveBackend(url) {
  window.localStorage.setItem(STORAGE_KEY, url);
}

function backendUrl() {
  const value = backendInput.value.trim();
  return value.length ? value : DEFAULT_BACKEND;
}

function applyStreamSource() {
  const url = backendUrl();
  streamImg.src = `${url.replace(/\/$/, "")}/stream.mjpg?_=${Date.now()}`;
}

function updateStatusText(text, backendHint = "") {
  const valueSpan = document.createElement("span");
  valueSpan.className = "status-value";
  valueSpan.textContent = text;

  const small = document.createElement("small");
  small.className = "backend-hint";
  small.textContent = backendHint;

  statusEl.textContent = "Status: ";
  statusEl.appendChild(valueSpan);
  if (backendHint) {
    statusEl.appendChild(document.createTextNode(" "));
    statusEl.appendChild(small);
  }
}

async function fetchJSON(path, options = {}) {
  const url = `${backendUrl().replace(/\/$/, "")}${path}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with ${response.status}`);
  }
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  return response.text();
}

async function refreshStatus() {
  try {
    const data = await fetchJSON("/camera/status");
    const statusText = data.running ? "running" : "stopped";
    const backendHint = data.backend ? `backend: ${data.backend}` : "";
    updateStatusText(statusText, backendHint);
  } catch (error) {
    updateStatusText("offline");
    console.error("Unable to fetch status:", error);
  }
}

async function startCamera() {
  try {
    await fetchJSON("/camera/start", { method: "POST" });
    await refreshStatus();
    applyStreamSource();
  } catch (error) {
    console.error("Failed to start camera:", error);
    updateStatusText("error starting camera");
  }
}

async function stopCamera() {
  try {
    await fetchJSON("/camera/stop", { method: "POST" });
    await refreshStatus();
  } catch (error) {
    console.error("Failed to stop camera:", error);
    updateStatusText("error stopping camera");
  }
}

function toggleStream() {
  isStreamVisible = !isStreamVisible;
  const container = document.querySelector(".stream-container");
  if (isStreamVisible) {
    container.classList.remove("hidden");
    toggleStreamBtn.textContent = "Hide Stream";
    applyStreamSource();
  } else {
    container.classList.add("hidden");
    toggleStreamBtn.textContent = "Show Stream";
  }
}

function captureFrame() {
  const url = `${backendUrl().replace(/\/$/, "")}/frame?_=${Date.now()}`;
  framePreview.src = url;
}

function connectBackend() {
  const url = backendUrl();
  saveBackend(url);
  applyStreamSource();
  refreshStatus();
  updateStatusText("connecting...");
}

function setupEventHandlers() {
  connectBtn.addEventListener("click", connectBackend);
  startBtn.addEventListener("click", startCamera);
  stopBtn.addEventListener("click", stopCamera);
  refreshStatusBtn.addEventListener("click", refreshStatus);
  toggleStreamBtn.addEventListener("click", toggleStream);
  captureFrameBtn.addEventListener("click", captureFrame);
}

function autoRefreshStatus() {
  if (statusTimer) {
    clearInterval(statusTimer);
  }
  statusTimer = setInterval(refreshStatus, 5000);
}

function init() {
  loadStoredBackend();
  setupEventHandlers();
  applyStreamSource();
  refreshStatus();
  autoRefreshStatus();
}

document.addEventListener("DOMContentLoaded", init);
