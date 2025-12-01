import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// Config (dynamic EAR calibration + movement)
const CONFIG = {
  requiredBlinks: 2,              // blinks needed
  earThreshold: null,             // will be calibrated; fallback used if calibration fails
  fallbackEAR: 0.22,              // default threshold if calibration not done
  calibrationFrames: 40,          // frames to compute average open-eye EAR
  earFramesClosed: 3,             // consecutive closed frames to count blink
  minOpenFramesAfterBlink: 2,     // ensure eye reopened before next blink
  moveThresholdRatio: 0.08,       // fraction of face width for movement (8% is more achievable)
  moveMinFrames: 5,               // require movement persistence (reduced from 8)
  movementCalibrationFrames: 15,  // frames to establish stable center position (reduced from 20)
  timeLimitSec: 45,               // verification timeout
  smoothAlpha: 0.3,               // smoothing factor for EAR
  debug: true                     // set true to log details
};

// Elements
const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const challengeListEl = document.getElementById("challengeList");
const challengeStatusEl = document.getElementById("challengeStatus");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");

// State
let stream = null;
let vision = null;
let landmarker = null;
let running = false;
let lastVideoTime = -1;
let rafId = 0;
let spoofFlagged = false;
let challengeSequence = [];
let currentChallengeIndex = 0;
let challengeCompleted = false;
let sizeHistory = [];
let motionEnergyHistory = [];
let lastFrameImageData = null;
let mouthOpenFrames = 0;
let forwardMoveFrames = 0;

// Blink detection state
let earClosedFrames = 0;
let blinkCount = 0;
let blinkLatched = false;
let openFramesAfterBlink = 0;
let calibratedFrames = 0;
let earAverageOpen = 0;
let smoothEAR = null;

// Movement detection state
let faceCenter0 = null; // initial center (calibrated average)
let faceCenterHistory = []; // history for calibration
let movementCalibrated = false;
let movedLeft = false;
let movedRight = false;
let moveLeftFrames = 0;
let moveRightFrames = 0;
let returnedToCenter = true; // must be at center before movement counts

// Head pose estimation state
let baseYaw = null; // calibrated yaw (left/right rotation)
let yawHistory = [];

// Eye landmark indices (MediaPipe FaceMesh style)
const LEFT_EYE = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE = [362, 385, 387, 263, 373, 380];

// Utility
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

// Extract yaw (left/right rotation) from transformation matrix
function getYawFromMatrix(matrix) {
  // Matrix is a 4x4 transformation matrix in column-major order
  // Extract rotation components
  const m00 = matrix.data[0];
  const m02 = matrix.data[2];
  
  // Calculate yaw angle in radians, then convert to degrees
  const yaw = Math.atan2(m02, m00) * (180 / Math.PI);
  return yaw;
}
function computeEAR(lms, ids) {
  const p1 = lms[ids[0]], p2 = lms[ids[1]], p3 = lms[ids[2]], p4 = lms[ids[3]], p5 = lms[ids[4]], p6 = lms[ids[5]];
  const vertical = dist(p2, p6) + dist(p3, p5);
  const horizontal = dist(p1, p4) * 2.0; // standard formula denominator (2*horizontal)
  if (horizontal === 0) return 0;
  return vertical / horizontal;
}

function faceBounds(landmarks) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of landmarks) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  return { minX, minY, maxX, maxY, cx: (minX + maxX) / 2, cy: (minY + maxY) / 2, w: (maxX - minX), h: (maxY - minY) };
}

function setStatus(text, cls = "") {
  statusEl.textContent = text;
  statusEl.className = `status ${cls}`.trim();
}

function setResult(text, cls = "") {
  resultEl.textContent = text;
  resultEl.className = `result ${cls}`.trim();
}

function setChallengeStatus(text, cls = "") {
  challengeStatusEl.textContent = text;
  challengeStatusEl.className = `result ${cls}`.trim();
}

async function ensureModels() {
  if (landmarker) return;
  setStatus("Loading models…");
  vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  landmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true, // Enable blend shapes for mouth/smile detection
    outputFacialTransformationMatrixes: true, // Enable head pose estimation
  });
}

async function startCamera() {
  if (running) return;
  
  try {
    // Check if mediaDevices is supported
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus("Camera not supported on this device", "err");
      setResult("Your browser doesn't support camera access", "err");
      return;
    }

    await ensureModels();
    setStatus("Requesting camera…");
    
    // Request camera with mobile-friendly constraints
    const constraints = {
      video: {
        facingMode: "user",
        width: { ideal: 640, max: 1280 },
        height: { ideal: 480, max: 720 }
      },
      audio: false
    };
    
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    
    // Wait for video metadata to load
    await new Promise((resolve, reject) => {
      video.onloadedmetadata = () => {
        video.play()
          .then(resolve)
          .catch(reject);
      };
      video.onerror = reject;
      // Timeout after 10 seconds
      setTimeout(() => reject(new Error("Video load timeout")), 10000);
    });
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    resetSession();
    running = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    setStatus("Look at the camera. Blink twice.", "warn");
    setResult("Verification in progress…", "warn");
    lastVideoTime = -1;
    rafId = requestAnimationFrame(loop);
  } catch (err) {
    console.error("Camera error:", err);
    let errorMessage = "Could not access camera";
    
    if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
      errorMessage = "Camera permission denied. Please allow camera access.";
    } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
      errorMessage = "No camera found on this device.";
    } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
      errorMessage = "Camera is already in use by another app.";
    } else if (err.name === "OverconstrainedError") {
      errorMessage = "Camera doesn't meet requirements.";
    } else if (err.name === "SecurityError") {
      errorMessage = "Camera access requires HTTPS on mobile devices.";
    } else if (err.message) {
      errorMessage = err.message;
    }
    
    setStatus(errorMessage, "err");
    setResult("Camera initialization failed", "err");
    
    // Clean up if stream was partially created
    if (stream) {
      for (const track of stream.getTracks()) {
        track.stop();
      }
      stream = null;
    }
  }
}

function stopCamera() {
  running = false;
  cancelAnimationFrame(rafId);
  if (stream) {
    for (const t of stream.getTracks()) t.stop();
    stream = null;
  }
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus("Stopped.");
  setResult("Awaiting start…");
}

function resetSession() {
  earClosedFrames = 0;
  blinkCount = 0;
  blinkLatched = false;
  openFramesAfterBlink = 0;
  calibratedFrames = 0;
  earAverageOpen = 0;
  smoothEAR = null;
  faceCenter0 = null;
  faceCenterHistory = [];
  movementCalibrated = false;
  movedLeft = movedRight = false;
  moveLeftFrames = 0;
  moveRightFrames = 0;
  returnedToCenter = true;
  baseYaw = null;
  yawHistory = [];
  startDeadline = CONFIG.timeLimitSec > 0 ? performance.now() + CONFIG.timeLimitSec * 1000 : 0;
  CONFIG.earThreshold = null; // reset calibration
  spoofFlagged = false;
  challengeSequence = buildChallengeSequence();
  currentChallengeIndex = 0;
  challengeCompleted = false;
  sizeHistory = [];
  motionEnergyHistory = [];
  lastFrameImageData = null;
  mouthOpenFrames = 0;
  forwardMoveFrames = 0;
  renderChallengeList();
  setChallengeStatus("Calibrating…", "warn");
}
// Challenge system ------------------------------------------------------
function buildChallengeSequence() {
  const actions = [
    { key: 'blink', label: `Blink ${CONFIG.requiredBlinks}×`, done: false },
    { key: 'turnLeft', label: 'Turn Head Left', done: false },
    { key: 'turnRight', label: 'Turn Head Right', done: false },
    { key: 'mouth', label: 'Open Mouth', done: false },
    { key: 'forward', label: 'Move Forward (closer)', done: false }
  ];
  // Shuffle
  for (let i = actions.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [actions[i], actions[j]] = [actions[j], actions[i]];
  }
  return actions;
}

function renderChallengeList(progressPercent = null) {
  if (!challengeListEl) return;
  challengeListEl.innerHTML = '';
  challengeSequence.forEach((c, idx) => {
    const li = document.createElement('li');
    let text = c.label;
    
    if (c.done) {
      text += ' ✓';
      li.style.color = '#17c964';
    } else if (idx === currentChallengeIndex) {
      // Show percentage for active challenge
      if (progressPercent !== null && progressPercent > 0) {
        text += ` (${Math.round(progressPercent)}%)`;
      }
      li.style.fontWeight = '600';
      li.style.color = '#00d0ff';
      
      // Add progress bar
      if (progressPercent !== null) {
        const progressBar = document.createElement('div');
        progressBar.style.cssText = 'height: 4px; background: rgba(0,208,255,0.2); border-radius: 2px; margin-top: 4px; overflow: hidden;';
        const progressFill = document.createElement('div');
        progressFill.style.cssText = `height: 100%; background: #00d0ff; width: ${progressPercent}%; transition: width 0.2s ease;`;
        progressBar.appendChild(progressFill);
        li.appendChild(document.createElement('br'));
        li.appendChild(progressBar);
      }
    } else {
      li.style.color = '#a5adba';
    }
    
    li.textContent = text;
    if (idx === currentChallengeIndex && progressPercent !== null && progressPercent > 0) {
      // Re-add progress bar after textContent (which clears innerHTML)
      const progressBar = document.createElement('div');
      progressBar.style.cssText = 'height: 4px; background: rgba(0,208,255,0.2); border-radius: 2px; margin-top: 4px; overflow: hidden;';
      const progressFill = document.createElement('div');
      progressFill.style.cssText = `height: 100%; background: #00d0ff; width: ${progressPercent}%; transition: width 0.2s ease;`;
      progressBar.appendChild(progressFill);
      li.appendChild(progressBar);
    }
    
    challengeListEl.appendChild(li);
  });
}

function getChallengeProgress() {
  const currentKey = challengeSequence[currentChallengeIndex]?.key;
  if (!currentKey) return 0;
  
  let current = 0;
  let required = 0;
  
  switch (currentKey) {
    case 'blink':
      current = blinkCount;
      required = CONFIG.requiredBlinks;
      break;
    case 'turnLeft':
      if (!movementCalibrated) return 0; // Still calibrating
      current = moveLeftFrames;
      required = CONFIG.moveMinFrames;
      break;
    case 'turnRight':
      if (!movementCalibrated) return 0; // Still calibrating
      current = moveRightFrames;
      required = CONFIG.moveMinFrames;
      break;
    case 'mouth':
      current = mouthOpenFrames;
      required = 4;
      break;
    case 'forward':
      current = forwardMoveFrames;
      required = 2;
      break;
  }
  
  return Math.min((current / required) * 100, 100);
}

function advanceChallengeIf(conditionMet) {
  if (challengeCompleted || spoofFlagged) return;
  if (!conditionMet) return;
  const current = challengeSequence[currentChallengeIndex];
  if (!current) return;
  current.done = true;
  currentChallengeIndex++;
  if (currentChallengeIndex >= challengeSequence.length) {
    challengeCompleted = true;
    setChallengeStatus('All challenges done.', 'ok');
  }
  renderChallengeList(0); // Reset progress for next challenge
}

let startDeadline = 0;

function drawOverlay(landmarks, bounds) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!landmarks) return;

  ctx.save();
  ctx.strokeStyle = "rgba(0,208,255,0.9)";
  ctx.lineWidth = 2;

  const scaleX = canvas.width;
  const scaleY = canvas.height;

  // Draw face bounds
  ctx.strokeRect(bounds.minX * scaleX, bounds.minY * scaleY, bounds.w * scaleX, bounds.h * scaleY);
  
  // Draw movement indicators
  const currentKey = challengeSequence[currentChallengeIndex]?.key;
  if (currentKey === 'turnLeft') {
    ctx.fillStyle = "rgba(0,208,255,0.8)";
    ctx.font = "bold 32px Arial";
    ctx.fillText("←", 30, canvas.height / 2);
  } else if (currentKey === 'turnRight') {
    ctx.fillStyle = "rgba(0,208,255,0.8)";
    ctx.font = "bold 32px Arial";
    ctx.fillText("→", canvas.width - 50, canvas.height / 2);
  } else if (currentKey === 'forward') {
    ctx.fillStyle = "rgba(0,208,255,0.8)";
    ctx.font = "bold 24px Arial";
    ctx.fillText("Move Closer", canvas.width / 2 - 60, 40);
  } else if (currentKey === 'mouth') {
    ctx.fillStyle = "rgba(0,208,255,0.8)";
    ctx.font = "bold 24px Arial";
    ctx.fillText("Open Mouth", canvas.width / 2 - 60, 40);
  }

  // Draw eyes
  const drawEye = (ids) => {
    ctx.beginPath();
    const first = landmarks[ids[0]];
    ctx.moveTo(first.x * scaleX, first.y * scaleY);
    for (let i = 1; i < ids.length; i++) {
      const p = landmarks[ids[i]];
      ctx.lineTo(p.x * scaleX, p.y * scaleY);
    }
    ctx.closePath();
    ctx.stroke();
  };
  drawEye(LEFT_EYE);
  drawEye(RIGHT_EYE);

  ctx.restore();
}

function updateBlink(earL, earR) {
  let earRaw = (earL + earR) / 2;
  // Smooth EAR
  if (smoothEAR === null) smoothEAR = earRaw;
  smoothEAR = smoothEAR + CONFIG.smoothAlpha * (earRaw - smoothEAR);

  // Calibration phase: gather open-eye EAR
  if (CONFIG.earThreshold === null) {
    earAverageOpen += smoothEAR;
    calibratedFrames++;
    if (calibratedFrames >= CONFIG.calibrationFrames) {
      const avg = earAverageOpen / calibratedFrames;
      CONFIG.earThreshold = avg * 0.75; // threshold relative to average
      if (CONFIG.earThreshold > 0.28) CONFIG.earThreshold = 0.28; // clamp upper bound
      if (CONFIG.earThreshold < 0.16) CONFIG.earThreshold = 0.16; // clamp lower bound
      setStatus(`Calibration done. Blink ${CONFIG.requiredBlinks} times.`, "warn");
      if (CONFIG.debug) console.log("Calibrated EAR threshold:", CONFIG.earThreshold.toFixed(3));
    } else {
      const calProgress = Math.round((calibratedFrames / CONFIG.calibrationFrames) * 100);
      setStatus(`Calibrating… ${calProgress}%`, "warn");
      return; // do not process blink yet
    }
  }

  const threshold = CONFIG.earThreshold ?? CONFIG.fallbackEAR;
  if (smoothEAR < threshold) {
    earClosedFrames++;
    openFramesAfterBlink = 0;
  } else {
    if (earClosedFrames >= CONFIG.earFramesClosed && !blinkLatched) {
      blinkCount++;
      blinkLatched = true;
      if (CONFIG.debug) console.log("Blink detected", blinkCount);
    }
    earClosedFrames = 0;
    openFramesAfterBlink++;
    if (blinkLatched && openFramesAfterBlink >= CONFIG.minOpenFramesAfterBlink) {
      blinkLatched = false;
      openFramesAfterBlink = 0;
    }
  }
  // If blink challenge is active
  const blinkActionActive = challengeSequence[currentChallengeIndex]?.key === 'blink';
  if (blinkActionActive && blinkCount >= CONFIG.requiredBlinks) {
    advanceChallengeIf(true);
  }
}

function updateMovement(transformationMatrix) {
  const currentKey = challengeSequence[currentChallengeIndex]?.key;
  
  // Only process if a turn challenge is active
  if (currentKey !== 'turnLeft' && currentKey !== 'turnRight') {
    // Reset movement state when not in a turn challenge
    yawHistory = [];
    baseYaw = null;
    movementCalibrated = false;
    moveLeftFrames = 0;
    moveRightFrames = 0;
    returnedToCenter = true;
    return;
  }
  
  if (!transformationMatrix) return;
  
  const currentYaw = getYawFromMatrix(transformationMatrix);
  
  // Calibrate base yaw over several frames
  if (!movementCalibrated) {
    yawHistory.push(currentYaw);
    if (yawHistory.length >= CONFIG.movementCalibrationFrames) {
      // Calculate stable base yaw as average
      baseYaw = yawHistory.reduce((a, b) => a + b, 0) / yawHistory.length;
      movementCalibrated = true;
      if (CONFIG.debug) console.log("Head pose calibrated, base yaw:", baseYaw.toFixed(1), "degrees");
    }
    return;
  }
  
  const yawDiff = currentYaw - baseYaw;
  const turnThreshold = 15; // degrees (was 8% of face width, now absolute angle)
  const centerThreshold = 8; // degrees for "at center"
  
  // Check if user's head is at center position
  const isAtCenter = Math.abs(yawDiff) < centerThreshold;
  
  if (isAtCenter) {
    returnedToCenter = true;
    // Decay movement counters when at center
    moveLeftFrames = Math.max(0, moveLeftFrames - 1);
    moveRightFrames = Math.max(0, moveRightFrames - 1);
  }
  
  // Only count movement if user started from center
  if (!returnedToCenter) {
    return;
  }
  
  // Count frames when turned left or right
  // Negative yaw = turned left (face moving left in camera view)
  // Positive yaw = turned right (face moving right in camera view)
  if (yawDiff < -turnThreshold) {
    moveLeftFrames++;
    moveRightFrames = 0;
    if (CONFIG.debug && moveLeftFrames % 5 === 0) console.log("Turning left:", yawDiff.toFixed(1), "deg, frames:", moveLeftFrames);
  } else if (yawDiff > turnThreshold) {
    moveRightFrames++;
    moveLeftFrames = 0;
    if (CONFIG.debug && moveRightFrames % 5 === 0) console.log("Turning right:", yawDiff.toFixed(1), "deg, frames:", moveRightFrames);
  }
  
  // Check if turn left challenge is completed
  if (currentKey === 'turnLeft' && moveLeftFrames >= CONFIG.moveMinFrames) {
    if (CONFIG.debug) console.log("Turn LEFT completed! Final angle:", yawDiff.toFixed(1), "degrees");
    advanceChallengeIf(true);
    // Reset for next movement challenge
    yawHistory = [];
    baseYaw = null;
    movementCalibrated = false;
    moveLeftFrames = 0;
    moveRightFrames = 0;
    returnedToCenter = true;
  }
  
  // Check if turn right challenge is completed
  if (currentKey === 'turnRight' && moveRightFrames >= CONFIG.moveMinFrames) {
    if (CONFIG.debug) console.log("Turn RIGHT completed! Final angle:", yawDiff.toFixed(1), "degrees");
    advanceChallengeIf(true);
    // Reset for next movement challenge
    yawHistory = [];
    baseYaw = null;
    movementCalibrated = false;
    moveLeftFrames = 0;
    moveRightFrames = 0;
    returnedToCenter = true;
  }
}
// Mouth open detection using MediaPipe blend shapes
function updateMouthOpen(blendshapes) {
  const currentKey = challengeSequence[currentChallengeIndex]?.key;
  if (currentKey !== 'mouth' || !blendshapes) return;
  
  // Find jawOpen blend shape (indicates mouth opening)
  const jawOpen = blendshapes.find(b => b.categoryName === 'jawOpen');
  
  if (jawOpen && jawOpen.score > 0.3) { // Threshold: 30% jaw open
    mouthOpenFrames++;
    if (CONFIG.debug && mouthOpenFrames % 5 === 0) {
      console.log('Mouth open:', jawOpen.score.toFixed(2));
    }
  } else {
    mouthOpenFrames = Math.max(0, mouthOpenFrames - 1);
  }
  
  if (mouthOpenFrames >= 4) {
    if (CONFIG.debug) console.log('Mouth challenge completed!');
    advanceChallengeIf(true);
  }
}

function updateForwardMovement(bounds) {
  const currentKey = challengeSequence[currentChallengeIndex]?.key;
  if (currentKey !== 'forward') return;
  sizeHistory.push(bounds.w);
  if (sizeHistory.length > 30) sizeHistory.shift();
  if (sizeHistory.length >= 10) {
    const first = sizeHistory[0];
    const maxVal = Math.max(...sizeHistory);
    // Require at least 8% increase in face width (reduced from 12%)
    if (maxVal > first * 1.08) {
      forwardMoveFrames++;
    }
  }
  if (forwardMoveFrames >= 2) advanceChallengeIf(true); // Reduced from 3 to 2 frames
}

// Spoof heuristics ------------------------------------------------------
function analyzeSpoof(frame, bounds) {
  if (spoofFlagged || !bounds) return;
  // Compute motion energy inside face bounding box
  try {
    const w = canvas.width, h = canvas.height;
    const sx = Math.floor(bounds.minX * w);
    const sy = Math.floor(bounds.minY * h);
    const sw = Math.floor(bounds.w * w);
    const sh = Math.floor(bounds.h * h);
    ctx.drawImage(video, 0, 0, w, h);
    const frameData = ctx.getImageData(sx, sy, sw, sh);
    let diffSum = 0;
    if (lastFrameImageData && lastFrameImageData.data.length === frameData.data.length) {
      const a = frameData.data;
      const b = lastFrameImageData.data;
      for (let i = 0; i < a.length; i += 4) {
        // grayscale diff
        const da = (a[i] + a[i+1] + a[i+2]) / 3;
        const db = (b[i] + b[i+1] + b[i+2]) / 3;
        diffSum += Math.abs(da - db);
      }
    }
    lastFrameImageData = frameData;
    motionEnergyHistory.push(diffSum / (sw * sh));
    if (motionEnergyHistory.length > 25) motionEnergyHistory.shift();
    // Evaluate low variance scenario after calibration done
    if (CONFIG.earThreshold !== null && motionEnergyHistory.length >= 20) {
      const avgMotion = motionEnergyHistory.reduce((a,c)=>a+c,0)/motionEnergyHistory.length;
      if (avgMotion < 0.8 && blinkCount === 0 && calibratedFrames > CONFIG.calibrationFrames + 20) {
        // Very low motion, no blink: possible static image
        spoofFlag('Static image detected');
      }
    }
    // EAR stability spoof check (printed eye holes / video loop)
    if (CONFIG.earThreshold !== null && blinkCount === 0 && calibratedFrames > CONFIG.calibrationFrames + 40) {
      // If smoothEAR fluctuates extremely little
      const delta = Math.abs(smoothEAR - CONFIG.earThreshold);
      if (delta < 0.005) spoofFlag('Eye pattern static');
    }
  } catch (e) { /* ignore */ }
}

function spoofFlag(reason) {
  if (spoofFlagged) return;
  spoofFlagged = true;
  setStatus(`Fake detected: ${reason}`, 'err');
  setResult('FAKE DETECTED ✗', 'err');
  stopCamera();
}

function loop() {
  if (!running) return;
  const now = performance.now();
  if (startDeadline && now > startDeadline) {
    setStatus("Time limit reached", "err");
    setResult("Verification failed: timeout.", "err");
    stopCamera();
    return;
  }

  const ts = video.currentTime * 1000;
  if (ts === lastVideoTime) {
    rafId = requestAnimationFrame(loop);
    return;
  }
  lastVideoTime = ts;

  const out = landmarker.detectForVideo(video, ts);
  const faces = out.faceLandmarks;

  if (faces && faces.length > 0) {
    const lms = faces[0];
    const bounds = faceBounds(lms);
    drawOverlay(lms, bounds);

    const earL = computeEAR(lms, LEFT_EYE);
    const earR = computeEAR(lms, RIGHT_EYE);
    const blendshapes = out.faceBlendshapes && out.faceBlendshapes.length > 0 ? out.faceBlendshapes[0].categories : null;
    const transformMatrix = out.facialTransformationMatrixes && out.facialTransformationMatrixes.length > 0 ? out.facialTransformationMatrixes[0] : null;
    
    updateBlink(earL, earR);
    updateMovement(transformMatrix);
    updateMouthOpen(blendshapes);
    updateForwardMovement(bounds);
    analyzeSpoof(video, bounds);

    const currentKey = challengeSequence[currentChallengeIndex]?.key;
    if (spoofFlagged) {
      // Already flagged
    } else if (CONFIG.earThreshold === null) {
      // Calibration messaging handled earlier
    } else if (!challengeCompleted) {
      const progress = getChallengeProgress();
      setChallengeStatus(`Do: ${challengeSequence[currentChallengeIndex]?.label || 'Completing…'}`, 'warn');
      // Update progress every few frames to avoid excessive re-renders
      if (Math.floor(performance.now() / 100) % 2 === 0) {
        renderChallengeList(progress);
      }
    } else {
      setChallengeStatus('Challenges complete.', 'ok');
      setStatus('Evaluating result…', 'ok');
    }

    // Draw debug text
    if (CONFIG.debug && CONFIG.earThreshold !== null) {
      ctx.save();
      ctx.fillStyle = "rgba(0,0,0,0.5)";
      ctx.fillRect(8, 8, 320, 145);
      ctx.fillStyle = "#00d0ff";
      ctx.font = "12px monospace";
      ctx.fillText(`EAR raw: ${((earL+earR)/2).toFixed(3)}`, 16, 24);
      ctx.fillText(`EAR smooth: ${smoothEAR.toFixed(3)}`, 16, 38);
      ctx.fillText(`Threshold: ${CONFIG.earThreshold.toFixed(3)}`, 16, 52);
      ctx.fillText(`Blinks: ${blinkCount}`, 16, 66);
      const currentAction = challengeSequence[currentChallengeIndex]?.key;
      let progressFrames = 0;
      if (currentAction === 'turnLeft') progressFrames = moveLeftFrames;
      if (currentAction === 'turnRight') progressFrames = moveRightFrames;
      if (currentAction === 'mouth') progressFrames = mouthOpenFrames;
      if (currentAction === 'forward') progressFrames = forwardMoveFrames;
      
      // Show blend shape info
      const blendshapes = out.faceBlendshapes && out.faceBlendshapes.length > 0 ? out.faceBlendshapes[0].categories : null;
      const jawOpen = blendshapes ? blendshapes.find(b => b.categoryName === 'jawOpen') : null;
      const transformMatrix = out.facialTransformationMatrixes && out.facialTransformationMatrixes.length > 0 ? out.facialTransformationMatrixes[0] : null;
      const currentYaw = transformMatrix ? getYawFromMatrix(transformMatrix) : null;
      const yawDiff = baseYaw !== null && currentYaw !== null ? currentYaw - baseYaw : null;
      
      ctx.fillText(`Move L: ${moveLeftFrames} R: ${moveRightFrames} (need ${CONFIG.moveMinFrames})`, 16, 80);
      ctx.fillText(`Mouth: ${mouthOpenFrames}/4 | JawOpen: ${jawOpen ? jawOpen.score.toFixed(2) : 'N/A'}`, 16, 94);
      ctx.fillText(`Head Yaw: ${yawDiff !== null ? yawDiff.toFixed(1) : 'calibrating'}° (base: ${baseYaw ? baseYaw.toFixed(1) : 'N/A'}°)`, 16, 108);
      ctx.fillText(`MoveCal: ${movementCalibrated} | AtCenter: ${returnedToCenter}`, 16, 122);
      
      // Progress bar for current challenge
      if (progressFrames > 0 && currentAction) {
        const maxFrames = currentAction === 'turnLeft' || currentAction === 'turnRight' ? CONFIG.moveMinFrames : 
                         currentAction === 'mouth' ? 4 : 2;
        const progress = Math.min(progressFrames / maxFrames, 1);
        ctx.fillStyle = "rgba(23,201,100,0.7)";
        ctx.fillRect(16, 132, 200 * progress, 6);
        ctx.strokeStyle = "rgba(255,255,255,0.3)";
        ctx.strokeRect(16, 132, 200, 6);
      }
      ctx.restore();
    }

    if (!spoofFlagged && challengeCompleted) {
      setResult('VERIFIED USER ✓', 'ok');
      stopCamera();
      return;
    }
  } else {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    setStatus("Show your face to the camera", "warn");
  }

  rafId = requestAnimationFrame(loop);
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);

// Check for HTTPS on page load (required for mobile camera access)
window.addEventListener("DOMContentLoaded", () => {
  if (location.protocol !== "https:" && location.hostname !== "localhost" && location.hostname !== "127.0.0.1") {
    setStatus("⚠️ HTTPS required for mobile camera access", "warn");
    setResult("Please use HTTPS or localhost", "warn");
  }
});

// Privacy note: we never capture frames or send them anywhere.
// Everything runs locally in the browser and data lives only in-memory.
