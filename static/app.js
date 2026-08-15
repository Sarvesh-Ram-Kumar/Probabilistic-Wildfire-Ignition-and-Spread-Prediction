const canvas = document.getElementById("mainCanvas");
const ctx = canvas.getContext("2d");
const modeIndicator = document.getElementById("modeIndicator");

const windDial = document.getElementById("windDial");
const windDialCtx = windDial.getContext("2d");
const windAngleInput = document.getElementById("windAngle");
const windStrengthInput = document.getElementById("windStrength");
const windAngleLabel = document.getElementById("windAngleLabel");
const windStrengthLabel = document.getElementById("windStrengthLabel");

const regionRadiusInput = document.getElementById("regionRadius");
const regionRadiusLabel = document.getElementById("regionRadiusLabel");
const regionDensityInput = document.getElementById("regionDensity");
const regionHumanInput = document.getElementById("regionHuman");

const DENSITY_COLORS = { low: "#4caf50", medium: "#d9a441", high: "#d94141" };

let state = {
  regions: [],
  wall: null,
  wind: { angle: 0, strength: 0.5 },
  env: { temperature: "medium", rainfall: "medium", lightning: "no" }
};

let mode = "place"; // 'place' | 'wall1' | 'wall2'
let wallFirstPoint = null;

let simTimeline = null;   // array of {second, events}
let simSummary = null;    // per-region final outcome
let burningByTime = null; // t -> Set of ids burning at that time (cumulative)
let currentT = 0;
let playing = false;
let playTimer = null;

// ---------- Env / region default controls ----------

document.getElementById("envTemp").addEventListener("change", updateEnv);
document.getElementById("envRain").addEventListener("change", updateEnv);
document.getElementById("envLightning").addEventListener("change", updateEnv);

async function updateEnv() {
  state.env.temperature = document.getElementById("envTemp").value;
  state.env.rainfall = document.getElementById("envRain").value;
  state.env.lightning = document.getElementById("envLightning").value;
  await fetch("/api/env", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state.env)
  });
}

regionRadiusInput.addEventListener("input", () => {
  regionRadiusLabel.textContent = regionRadiusInput.value;
});

document.getElementById("drawWallBtn").addEventListener("click", () => {
  wallFirstPoint = null;
  setMode("wall1");
});

document.getElementById("clearWallBtn").addEventListener("click", async () => {
  await fetch("/api/wall", { method: "DELETE" });
  state.wall = null;
  draw();
});

document.getElementById("clearAllBtn").addEventListener("click", async () => {
  await fetch("/api/clear", { method: "POST" });
  state.regions = [];
  state.wall = null;
  resetSimUI();
  draw();
});

document.getElementById("simulateBtn").addEventListener("click", runSimulation);

function setMode(m) {
  mode = m;
  const labels = {
    place: "Mode: place region",
    wall1: "Mode: click first wall point",
    wall2: "Mode: click second wall point"
  };
  modeIndicator.textContent = labels[m] || "";
}
setMode("place");

// ---------- Canvas click ----------

canvas.addEventListener("click", async (e) => {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top) * scaleY;

  if (mode === "place") {
    const res = await fetch("/api/region", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        x, y,
        radius: parseFloat(regionRadiusInput.value),
        density: regionDensityInput.value,
        human_proximity: regionHumanInput.value
      })
    });
    const region = await res.json();
    state.regions.push(region);
    renderRegionList();
    draw();

  } else if (mode === "wall1") {
    wallFirstPoint = { x, y };
    setMode("wall2");

  } else if (mode === "wall2") {
    const wall = { x1: wallFirstPoint.x, y1: wallFirstPoint.y, x2: x, y2: y };
    const res = await fetch("/api/wall", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(wall)
    });
    state.wall = await res.json();
    setMode("place");
    draw();
  }
});

// ---------- Region list panel ----------

function renderRegionList() {
  const container = document.getElementById("regionList");
  if (state.regions.length === 0) {
    container.innerHTML = `<p class="hint">No regions yet — click the canvas.</p>`;
    return;
  }
  container.innerHTML = "";
  state.regions.forEach(r => {
    const row = document.createElement("div");
    row.className = "region-row";
    row.innerHTML = `
      <div class="row-top">
        <span>Region #${r.id} (r=${Math.round(r.radius)})</span>
        <button class="remove-btn" data-id="${r.id}">✕ remove</button>
      </div>
      <label>Density
        <select data-field="density" data-id="${r.id}">
          <option value="low" ${r.density === "low" ? "selected" : ""}>Low</option>
          <option value="medium" ${r.density === "medium" ? "selected" : ""}>Medium</option>
          <option value="high" ${r.density === "high" ? "selected" : ""}>High</option>
        </select>
      </label>
      <label>Human proximity
        <select data-field="human_proximity" data-id="${r.id}">
          <option value="no" ${r.human_proximity === "no" ? "selected" : ""}>Far</option>
          <option value="yes" ${r.human_proximity === "yes" ? "selected" : ""}>Near</option>
        </select>
      </label>
      <label>Radius
        <input type="range" min="15" max="70" value="${r.radius}" data-field="radius" data-id="${r.id}">
      </label>
    `;
    container.appendChild(row);
  });

  container.querySelectorAll("select, input[type=range]").forEach(el => {
    el.addEventListener("change", onRegionFieldChange);
  });
  container.querySelectorAll(".remove-btn").forEach(btn => {
    btn.addEventListener("click", onRegionRemove);
  });
}

async function onRegionFieldChange(e) {
  const id = parseInt(e.target.dataset.id);
  const field = e.target.dataset.field;
  const value = field === "radius" ? parseFloat(e.target.value) : e.target.value;
  const res = await fetch(`/api/region/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ [field]: value })
  });
  const updated = await res.json();
  const idx = state.regions.findIndex(r => r.id === id);
  if (idx >= 0) state.regions[idx] = updated;
  draw();
}

async function onRegionRemove(e) {
  const id = parseInt(e.target.dataset.id);
  await fetch(`/api/region/${id}`, { method: "DELETE" });
  state.regions = state.regions.filter(r => r.id !== id);
  renderRegionList();
  draw();
}

// ---------- Wind dial ----------

function drawWindDial() {
  const cx = 55, cy = 55, r = 42;
  windDialCtx.clearRect(0, 0, 110, 110);
  windDialCtx.strokeStyle = "#2a2f3a";
  windDialCtx.beginPath();
  windDialCtx.arc(cx, cy, r, 0, Math.PI * 2);
  windDialCtx.stroke();

  const rad = (state.wind.angle * Math.PI) / 180;
  const len = r * (0.4 + 0.6 * state.wind.strength);
  const ex = cx + len * Math.cos(rad);
  const ey = cy + len * Math.sin(rad);

  windDialCtx.strokeStyle = "#ff6b3d";
  windDialCtx.lineWidth = 3;
  windDialCtx.beginPath();
  windDialCtx.moveTo(cx, cy);
  windDialCtx.lineTo(ex, ey);
  windDialCtx.stroke();

  const headLen = 8;
  windDialCtx.beginPath();
  windDialCtx.moveTo(ex, ey);
  windDialCtx.lineTo(ex - headLen * Math.cos(rad - Math.PI / 6), ey - headLen * Math.sin(rad - Math.PI / 6));
  windDialCtx.lineTo(ex - headLen * Math.cos(rad + Math.PI / 6), ey - headLen * Math.sin(rad + Math.PI / 6));
  windDialCtx.closePath();
  windDialCtx.fillStyle = "#ff6b3d";
  windDialCtx.fill();
  windDialCtx.lineWidth = 1;
}

async function updateWind() {
  state.wind.angle = parseFloat(windAngleInput.value);
  state.wind.strength = parseFloat(windStrengthInput.value);
  windAngleLabel.textContent = state.wind.angle;
  windStrengthLabel.textContent = state.wind.strength.toFixed(2);
  drawWindDial();
  await fetch("/api/wind", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state.wind)
  });
}
windAngleInput.addEventListener("input", updateWind);
windStrengthInput.addEventListener("input", updateWind);

let dialDragging = false;
windDial.addEventListener("mousedown", () => dialDragging = true);
window.addEventListener("mouseup", () => dialDragging = false);
windDial.addEventListener("mousemove", (e) => {
  if (!dialDragging) return;
  const rect = windDial.getBoundingClientRect();
  const x = e.clientX - rect.left - 55;
  const y = e.clientY - rect.top - 55;
  let angle = Math.atan2(y, x) * 180 / Math.PI;
  if (angle < 0) angle += 360;
  windAngleInput.value = Math.round(angle);
  updateWind();
});

// ---------- Draw ----------

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (state.wall) {
    ctx.strokeStyle = "#6fd3ff";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(state.wall.x1, state.wall.y1);
    ctx.lineTo(state.wall.x2, state.wall.y2);
    ctx.stroke();
    ctx.lineWidth = 1;
  }

  const burningNow = burningByTime ? burningByTime[currentT] : null;

  for (const r of state.regions) {
    const isBurning = burningNow && burningNow.has(r.id);
    ctx.beginPath();
    ctx.arc(r.x, r.y, r.radius, 0, Math.PI * 2);
    ctx.fillStyle = isBurning ? "rgba(255,61,61,0.35)" : hexToRgba(DENSITY_COLORS[r.density], 0.25);
    ctx.fill();
    ctx.strokeStyle = isBurning ? "#ff3d3d" : DENSITY_COLORS[r.density];
    ctx.lineWidth = isBurning ? 3 : 2;
    ctx.stroke();
    ctx.lineWidth = 1;

    ctx.fillStyle = "#e8eaed";
    ctx.font = "11px sans-serif";
    ctx.fillText(`#${r.id}`, r.x - 8, r.y + 4);
  }
}

function hexToRgba(hex, alpha) {
  const bigint = parseInt(hex.slice(1), 16);
  const rr = (bigint >> 16) & 255, gg = (bigint >> 8) & 255, bb = bigint & 255;
  return `rgba(${rr},${gg},${bb},${alpha})`;
}

// ---------- Simulation ----------

async function runSimulation() {
  if (state.regions.length === 0) {
    alert("Place at least one region first.");
    return;
  }
  const seconds = parseInt(document.getElementById("simSeconds").value) || 50;
  const res = await fetch("/api/simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ seconds })
  });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }

  simTimeline = data.timeline;
  simSummary = data.summary;

  // build cumulative burning sets per time step
  burningByTime = {};
  let burning = new Set();
  burningByTime[0] = new Set();
  for (const step of simTimeline) {
    for (const ev of step.events) burning.add(ev.id);
    burningByTime[step.second] = new Set(burning);
  }

  currentT = 0;
  document.getElementById("timelineControls").style.display = "flex";
  const slider = document.getElementById("timeSlider");
  slider.min = 0;
  slider.max = simTimeline.length;
  slider.value = 0;
  document.getElementById("timeLabel").textContent = "t = 0s";

  renderEventLogUpTo(0);
  draw();
}

document.getElementById("timeSlider").addEventListener("input", (e) => {
  currentT = parseInt(e.target.value);
  document.getElementById("timeLabel").textContent = `t = ${currentT}s`;
  renderEventLogUpTo(currentT);
  draw();
});

document.getElementById("playBtn").addEventListener("click", () => {
  if (!simTimeline) return;
  playing = !playing;
  document.getElementById("playBtn").textContent = playing ? "⏸ Pause" : "▶ Play";
  if (playing) {
    playTimer = setInterval(() => {
      if (currentT >= simTimeline.length) {
        playing = false;
        document.getElementById("playBtn").textContent = "▶ Play";
        clearInterval(playTimer);
        return;
      }
      currentT++;
      document.getElementById("timeSlider").value = currentT;
      document.getElementById("timeLabel").textContent = `t = ${currentT}s`;
      renderEventLogUpTo(currentT);
      draw();
    }, 300);
  } else {
    clearInterval(playTimer);
  }
});

function renderEventLogUpTo(t) {
  const log = document.getElementById("eventLog");
  let html = "";
  for (const step of simTimeline) {
    if (step.second > t) break;
    for (const ev of step.events) {
      const tagClass = ev.cause_type === "spread" ? "spread" : (ev.cause_type === "both" ? "both" : "cause");
      html += `<div class="event-line"><span class="tag ${tagClass}">${ev.cause_type}</span>` +
              `t=${step.second}s — Region #${ev.id} ignited (${ev.cause_detail})</div>`;
    }
  }
  if (!html) html = `<p class="hint">No ignition events yet.</p>`;
  log.innerHTML = html;
  log.scrollTop = log.scrollHeight;
}

function resetSimUI() {
  simTimeline = null;
  simSummary = null;
  burningByTime = null;
  currentT = 0;
  playing = false;
  clearInterval(playTimer);
  document.getElementById("timelineControls").style.display = "none";
  document.getElementById("eventLog").innerHTML = "";
  renderRegionList();
}

// ---------- Init ----------

async function loadState() {
  const res = await fetch("/api/state");
  const data = await res.json();
  state.regions = data.regions;
  state.wall = data.wall;
  state.wind = data.wind;
  state.env = data.env;

  windAngleInput.value = state.wind.angle;
  windStrengthInput.value = state.wind.strength;
  windAngleLabel.textContent = state.wind.angle;
  windStrengthLabel.textContent = state.wind.strength.toFixed(2);
  drawWindDial();

  document.getElementById("envTemp").value = state.env.temperature;
  document.getElementById("envRain").value = state.env.rainfall;
  document.getElementById("envLightning").value = state.env.lightning;

  renderRegionList();
  draw();
}

loadState();
