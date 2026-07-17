// frontend/script.js
const API_BASE = "http://127.0.0.1:5000";

let currentAuthMode = "login";
let currentUserName = "";

// ======= UI helpers =======
function showScreen(name) {
  document.querySelectorAll(".screen").forEach(s => s.classList.remove("active"));
  const el = document.getElementById(`screen-${name}`);
  if (el) el.classList.add("active");
}

function setText(selector, text) {
  const el = document.getElementById(selector);
  if (el) el.textContent = text;
}

function showElement(id) {
  const el = document.getElementById(id);
  if (el) el.classList.remove("hidden");
}
function hideElement(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add("hidden");
}

// ======= Auth + UI boot =======
function switchAuthMode(mode) {
  currentAuthMode = mode;
  document.getElementById("tab-login").classList.toggle("active", mode === "login");
  document.getElementById("tab-register").classList.toggle("active", mode === "register");
  document.getElementById("name-field").classList.toggle("hidden", mode === "login");
  document.getElementById("auth-btn").textContent = mode === "login" ? "Login" : "Register";
}

function handleAuthSubmit(e) {
  e.preventDefault();
  const name = (currentAuthMode === "register") ? (document.getElementById("name").value.trim() || "Patient") : "Patient";
  const email = document.getElementById("email").value.trim();
  const pass = document.getElementById("password").value.trim();
  if (!email || !pass) return alert("Enter email & password");

  currentUserName = name;
  setText("welcome-user", `Welcome, ${currentUserName}`);
  setText("status-summary", "Your MG status is being monitored. Use the cards to run tests.");
  const avatar = document.getElementById("profile-avatar");
  if (avatar) avatar.textContent = currentUserName.charAt(0).toUpperCase();
  showScreen("home");
}

// ======= API helper =======
async function callApi(endpoint, { method = "POST", jsonBody = null, formData = null } = {}) {
  const url = `${API_BASE}${endpoint}`;
  let opts = { method, headers: {} };

  if (formData) {
    opts.body = formData; // DO NOT set Content-Type — browser will set multipart boundary
  } else if (jsonBody) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(jsonBody);
  }

  const res = await fetch(url, opts);
  let dataText = await res.text();
  try {
    const data = JSON.parse(dataText || "{}");
    if (!res.ok) throw new Error(data.message || `Server ${res.status}`);
    return data;
  } catch (err) {
    // If JSON parse fails, keep the raw text
    if (!res.ok) throw new Error(`Server error ${res.status}: ${dataText}`);
    try { return JSON.parse(dataText); } catch { return { raw: dataText }; }
  }
}

// ======= FATIGUE flow (unchanged) =======
function onFatigueClick() { showElement("fatigue-modal"); }
function closeFatigueModal() { hideElement("fatigue-modal"); setText("fatigue-result", ""); }

async function submitFatigueForm() {
  const sleep = parseFloat(document.getElementById("fatigue-sleep").value);
  const stress = parseInt(document.getElementById("fatigue-stress").value);
  const muscle = parseFloat(document.getElementById("fatigue-muscle").value);
  const out = document.getElementById("fatigue-result");
  if (isNaN(sleep) || isNaN(stress) || isNaN(muscle)) return out.textContent = "Please fill all fields.";

  out.textContent = "Running AI fatigue test…";
  try {
    const data = await callApi("/api/fatigue", { jsonBody: { sleep_hours: sleep, stress_level: stress, muscle_strength: muscle } });
    out.textContent = data.summary || `Status: ${data.status}`;
    setText("status-summary", `Latest: ${data.status} | Sleep:${data.sleep}h Stress:${data.stress}`);
  } catch (err) {
    console.error(err); out.textContent = "Error: " + err.message;
  }
}

// ======= VOICE flow (browser recording + send) =======
function onVoiceClick() { showElement("voice-modal"); setText("voice-status", "Preparing microphone..."); startVoiceRecordingFlow(); }
function closeVoiceModal() { hideElement("voice-modal"); setText("voice-status", ""); }

// Record audio for `durationMs` ms and return a Blob
function recordAudioBrowser(durationMs = 5000) {
  return new Promise(async (resolve, reject) => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return reject(new Error("Browser does not support getUserMedia()"));
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const options = { mimeType: "audio/webm" }; // widely supported; backend can accept webm/ogg/wav
      const mediaRecorder = new MediaRecorder(stream, options);

      let chunks = [];
      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size) chunks.push(e.data);
      };

      mediaRecorder.onstop = () => {
        stream.getTracks().forEach(t => t.stop());
        const blob = new Blob(chunks, { type: chunks[0]?.type || "audio/webm" });
        resolve(blob);
      };

      mediaRecorder.onerror = (e) => {
        stream.getTracks().forEach(t => t.stop());
        reject(e.error || new Error("Recording error"));
      };

      mediaRecorder.start();
      setTimeout(() => {
        try { mediaRecorder.stop(); } catch {}
      }, durationMs);
    } catch (err) {
      reject(err);
    }
  });
}

async function startVoiceRecordingFlow() {
  const statusEl = document.getElementById("voice-status");
  try {
    setText("voice-status", "Recording (5s)… please speak clearly.");
    const audioBlob = await recordAudioBrowser(5000); // record 5s
    setText("voice-status", "Uploading audio to server…");

    const fd = new FormData();
    // name the file field 'audio' — backend expects request.files['audio']
    const filename = `voice_${Date.now()}.webm`;
    fd.append("audio", audioBlob, filename);

    // POST to backend endpoint that accepts multipart form data
    const data = await callApi("/api/voice", { formData: fd, method: "POST" });
    setText("voice-status", data.summary || `Voice status: ${data.status}`);
    // update central UI
    setText("status-summary", `Last voice: ${data.status}`);
  } catch (err) {
    console.error(err);
    if (err.name === "NotAllowedError" || /Permission/i.test(err.message)) {
      setText("voice-status", "Microphone permission denied. Allow mic in browser and reload the page.");
    } else {
      setText("voice-status", "Error: " + (err.message || err));
    }
  }
}

// ======= PTOSIS flow (webcam preview + send single frame) =======
let _ptosisStream = null;
function onPtosisClick() {
  showElement("ptosis-modal");
  setText("ptosis-status", "Requesting camera permission…");
  startPtosisCamera();
}
function closePtosisModal() {
  hideElement("ptosis-modal");
  stopPtosisCamera();
  setText("ptosis-status", "");
}

async function startPtosisCamera() {
  const video = document.getElementById("camera");
  try {
    _ptosisStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
    video.srcObject = _ptosisStream;
    video.play();
    setText("ptosis-status", "Camera active — capturing frame in 3s. Please look straight.");
    // wait a bit then capture
    setTimeout(captureAndSendFrame, 3000);
  } catch (err) {
    console.error(err);
    if (err.name === "NotAllowedError") {
      setText("ptosis-status", "Camera permission denied. Allow camera in browser settings.");
    } else {
      setText("ptosis-status", "Camera error: " + err.message);
    }
  }
}

function stopPtosisCamera() {
  if (_ptosisStream) {
    _ptosisStream.getTracks().forEach(t => t.stop());
    _ptosisStream = null;
  }
  const video = document.getElementById("camera");
  if (video) { video.pause(); video.srcObject = null; }
}

async function captureAndSendFrame() {
  const video = document.getElementById("camera");
  const statusEl = document.getElementById("ptosis-status");
  if (!video || !video.srcObject) {
    setText("ptosis-status", "No camera stream available.");
    return;
  }

  // draw to canvas
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);

  // convert to blob and send
  canvas.toBlob(async (blob) => {
    if (!blob) return setText("ptosis-status", "Failed to capture frame.");
    setText("ptosis-status", "Sending frame for analysis…");

    const fd = new FormData();
    fd.append("frame", blob, `frame_${Date.now()}.jpg`);

    try {
      const data = await callApi("/api/ptosis-image", { formData: fd, method: "POST" });
      setText("ptosis-status", data.summary || `Eye: ${data.status} | EAR: ${data.avg_ear}`);
      setText("status-summary", `Last eye: ${data.status}`);
    } catch (err) {
      console.error(err);
      setText("ptosis-status", "Error: " + err.message);
    } finally {
      // stop camera after capture
      stopPtosisCamera();
    }
  }, "image/jpeg", 0.9);
}

// ======= Health log / other actions =======
function onHealthLogClick() {
  window.open("../data/central_log.html", "_blank");
}
function onChatClick() {
  alert("Chatbot (demo): will be available soon. Backend/chat code can be connected to this button.");
}
function onEmergencyClick() {
  alert("If breathing/swallowing problems, go to nearest emergency immediately.");
}

// start
showScreen("welcome");

// expose some functions for inline buttons (if needed)
window.switchAuthMode = switchAuthMode;
window.handleAuthSubmit = handleAuthSubmit;
window.onFatigueClick = onFatigueClick;
window.closeFatigueModal = closeFatigueModal;
window.submitFatigueForm = submitFatigueForm;
window.onVoiceClick = onVoiceClick;
window.closeVoiceModal = closeVoiceModal;
window.onPtosisClick = onPtosisClick;
window.closePtosisModal = closePtosisModal;
window.onHealthLogClick = onHealthLogClick;
window.onChatClick = onChatClick;
window.onEmergencyClick = onEmergencyClick;
