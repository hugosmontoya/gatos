import {
  HandLandmarker,
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ---- Mapeo de gestos e imágenes -----------------------------------------
const GESTURE_MEMES = {
  paz: ["imagenestarea2/paz.png"],
  ILY: ["imagenestarea2/ILY.png"],
  pulgarArriba: ["imagenestarea2/pulgararriba.png"],
  saludo: ["imagenestarea2/saludo.png"],
  doblePuño: ["imagenestarea2/doblepuño.png"],
  dobleIndice: ["imagenestarea2/dobleindice.png"],
  bocaAbierta: ["imagenestarea2/bocabierta.png"],
  manosArriba: ["imagenestarea2/manosarriba.png"],
  default: ["memes/pokercat.jpg"],
};

const STABLE_FRAMES_REQUIRED = 4;
const DEFAULT_FALLBACK_MS = 600;
const FACE_STALE_MS = 1200;
const MOUTH_OPEN_THRESHOLD = 0.15;

const video = document.getElementById("video");
const memeImg = document.getElementById("memeImg");
const debugHud = document.getElementById("debugHud");

let handLandmarker, faceLandmarker;
let lastVideoTime = -1;
let currentGesture = "default";
let candidateGesture = "default";
let candidateStreak = 0;
let lastNonDefaultAt = performance.now();
let lastFace = null; // { mouthCenter, faceWidth, mouthOpen, yawDeg, t }
let lastFaceSeenThisFrame = false;

async function init() {
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });

  faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFacialTransformationMatrixes: true,
  });

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();

  requestAnimationFrame(loop);
}

// ---- Geometría 3D ------------------------------------------------------
function vec(a, b) {
  return { x: b.x - a.x, y: b.y - a.y, z: (b.z || 0) - (a.z || 0) };
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}

function angleDeg(v1, v2) {
  const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  const m1 = Math.hypot(v1.x, v1.y, v1.z);
  const m2 = Math.hypot(v2.x, v2.y, v2.z);
  if (m1 < 1e-9 || m2 < 1e-9) return 180;
  return (Math.acos(Math.min(1, Math.max(-1, dot / (m1 * m2)))) * 180) / Math.PI;
}

function fingerExtended(lm, mcp, pip, tip) {
  const angle = angleDeg(vec(lm[mcp], lm[pip]), vec(lm[pip], lm[tip]));
  return angle < 45;
}

function classifyHand(lm) {
  const handScale = dist(lm[0], lm[9]) || 1e-6;

  const indexUp = fingerExtended(lm, 5, 6, 8);
  const middleUp = fingerExtended(lm, 9, 10, 12);
  const ringUp = fingerExtended(lm, 13, 14, 16);
  const pinkyUp = fingerExtended(lm, 17, 18, 20);

  const thumbPinkySpread = dist(lm[4], lm[17]) / handScale;
  const thumbOut = thumbPinkySpread > 1.05;

  const curledCount = [indexUp, middleUp, ringUp, pinkyUp].filter((v) => !v).length;

  return {
    indexUp,
    middleUp,
    ringUp,
    pinkyUp,
    thumbOut,
    curledCount,
    handScale,
    indexTip: lm[8],
    wrist: lm[0],
    palmCenter: lm[9],
    pts: lm,
  };
}

function updateFace(faceResult) {
  const now = performance.now();
  const sawFace = !!(faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0);

  if (sawFace) {
    const f = faceResult.faceLandmarks[0];
    const upperLip = f[13];
    const lowerLip = f[14];
    const rightCheek = f[234];
    const leftCheek = f[454];
    const mouthCenter = {
      x: (upperLip.x + lowerLip.x) / 2,
      y: (upperLip.y + lowerLip.y) / 2,
      z: ((upperLip.z || 0) + (lowerLip.z || 0)) / 2,
    };
    const faceWidth = dist(rightCheek, leftCheek);
    const mouthOpen = dist(upperLip, lowerLip) / (faceWidth || 1e-6);

    lastFace = { mouthCenter, faceWidth, mouthOpen, t: now };
  }
  lastFaceSeenThisFrame = sawFace;
}

function isPointing(h) {
  return h.indexUp && !h.middleUp && !h.ringUp && !h.pinkyUp;
}

function isPeaceSign(h) {
  // Símbolo de la paz: dos dedos extendidos (índice y medio), anular y meñique doblados
  return h.indexUp && h.middleUp && !h.ringUp && !h.pinkyUp;
}

function isIlySign(h) {
  // ILY: pulgar, índice y meñique extendidos; medio y anular doblados
  return h.thumbOut && h.indexUp && h.pinkyUp && !h.middleUp && !h.ringUp;
}

function isThumbsUp(h) {
  // Pulgar arriba: 4 dedos doblados y la punta del pulgar hacia arriba
  if (h.curledCount !== 4) return false;
  const pts = h.pts;
  const thumbTipY = pts[4].y;
  return thumbTipY < pts[3].y && thumbTipY < pts[2].y && thumbTipY < pts[5].y;
}

function isSaludo(h) {
  // Saludo: mano abierta con dedos extendidos
  return h.curledCount === 0 || (h.indexUp && h.middleUp && h.ringUp && h.pinkyUp);
}

function decideGesture(handResult) {
  const now = performance.now();
  const faceIsFresh = !!lastFace && now - lastFace.t < FACE_STALE_MS;
  const mouthOpen = faceIsFresh ? lastFace.mouthOpen : 0;

  if (!handResult.landmarks || handResult.landmarks.length === 0) {
    if (faceIsFresh && mouthOpen > MOUTH_OPEN_THRESHOLD) {
      return "bocaAbierta";
    }
    return "default";
  }

  const hands = handResult.landmarks.map(classifyHand);

  // 1. Gestos de 2 manos
  if (hands.length >= 2) {
    const [h0, h1] = hands;

    // Manos abiertas arriba
    const headTopY = faceIsFresh
      ? lastFace.mouthCenter.y - lastFace.faceWidth * 0.8
      : 0.45;
    const handsUp =
      (h0.palmCenter.y < headTopY || h0.palmCenter.y < 0.45) &&
      (h1.palmCenter.y < headTopY || h1.palmCenter.y < 0.45);
    if (handsUp && h0.curledCount <= 2 && h1.curledCount <= 2) {
      return "manosArriba";
    }

    // Doble puño
    if (
      h0.curledCount === 4 &&
      h1.curledCount === 4 &&
      !isThumbsUp(h0) &&
      !isThumbsUp(h1)
    ) {
      return "doblePuño";
    }

    // Doble índice
    if (isPointing(h0) && isPointing(h1)) {
      return "dobleIndice";
    }
  }

  // 2. Gesto facial: Boca abierta
  if (faceIsFresh && mouthOpen > MOUTH_OPEN_THRESHOLD) {
    return "bocaAbierta";
  }

  // 3. Gestos de 1 mano
  for (const h of hands) {
    if (isPeaceSign(h)) return "paz";
    if (isIlySign(h)) return "ILY";
    if (isThumbsUp(h)) return "pulgarArriba";
    if (isSaludo(h)) return "saludo";
  }

  return "default";
}

function pickImage(gesture) {
  const images = GESTURE_MEMES[gesture] || GESTURE_MEMES["default"];
  return images[Math.floor(Math.random() * images.length)];
}

function applyGesture(gesture) {
  if (gesture === currentGesture) return;
  currentGesture = gesture;
  memeImg.src = pickImage(gesture);
}

function loop() {
  const now = performance.now();
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const ts = performance.now();

    const handResult = handLandmarker.detectForVideo(video, ts);
    const faceResult = faceLandmarker.detectForVideo(video, ts);
    updateFace(faceResult);

    const gesture = decideGesture(handResult);

    if (gesture === candidateGesture) {
      candidateStreak++;
    } else {
      candidateGesture = gesture;
      candidateStreak = 1;
    }

    if (candidateStreak >= STABLE_FRAMES_REQUIRED) {
      applyGesture(gesture);
    }

    if (gesture !== "default") lastNonDefaultAt = now;
    if (
      now - lastNonDefaultAt > DEFAULT_FALLBACK_MS &&
      currentGesture !== "default"
    ) {
      applyGesture("default");
    }

    updateDebugHud();
  }
  requestAnimationFrame(loop);
}

function updateDebugHud() {
  if (!debugHud) return;
  const mouthVal =
    lastFace && performance.now() - lastFace.t < FACE_STALE_MS
      ? lastFace.mouthOpen
      : 0;
  debugHud.textContent =
    `Gesto actual: ${currentGesture}\n` +
    `Boca abierta: ${mouthVal.toFixed(2)} (umbral > ${MOUTH_OPEN_THRESHOLD.toFixed(2)})`;
}

init().catch((err) => console.error(err));
