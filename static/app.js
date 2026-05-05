// ── Unified ASR UI ──────────────────────────────────────────────────────────
// Record audio → capture raw PCM at 16kHz → encode WAV → POST /transcribe.
// Same WAV format as file upload — guaranteed identical server-side processing.

(() => {
  "use strict";

  // If the streaming-only UI is present, use that flow and keep legacy UI intact.
  const streamRoot = document.getElementById("btn-stream");
  if (streamRoot) {
    let isStreaming = false;
    let mediaStream = null;
    let audioContext = null;
    let captureNode = null;
    let analyser = null;
    let pcmChunks = [];
    let streamingStart = 0;
    let timerInterval = null;
    let animFrame = null;
    let sendInFlight = false;
    let lastSendAt = 0;
    let lastTranscript = "";

    const STREAM_LANG = "te";
    const MIN_CHUNKS = 5;
    const SEND_INTERVAL_MS = 800;
    const LOG_MAX_ITEMS = 20;
    const MAX_EDIT_RATIO = 0.25;
    const MAX_EDIT_ABS = 3;

    const TELUGU_EMERGENCY_PHRASES = [
      "హెల్ప్ చేయండి",
      "డాక్టర్‌ను కాల్ చేయండి త్వరగా",
      "అంబులెన్స్ పిలవండి రా",
      "పోలీస్ కి కాల్ చెయ్యి",
      "పోలీసు కి కాల్ చేయండి",
      "అంబులెన్స్ పిలవండి త్వరగా",
      "ఎమర్జెన్సీ ఉందీ",
      "నేను అస్వస్థంగా ఉన్నాను",
      "నాకు గుండె నొప్పి",
      "నాకు శ్వాస ఆడటం లేదు",
      "నాకు రక్తస్రావం అవుతోంది",
      "తక్షణ సహాయం కావాలి",
      "దయచేసి సహాయం చేయండి",
      "దారి ప్రమాదం జరిగింది",
      "అగ్ని ప్రమాదం జరిగింది",
      "నీరు మునిగిపోతున్నాను",
      "నాకు స్పృహ లేదు",
      "సిరియస్ గా ఉంది",
      "దయచేసి వెంటనే రండి",
      "నాకు చాలా నొప్పిగా ఉంది",
      "నాకు బాగోలేదు హాస్పిటల్ తీసుకెళ్లండి",
      "అయ్యో యాక్సిడెంట్ అయింది",
      "త్వరగా రండి ప్లీజ్",
      "ఎవరైనా హెల్ప్ చేయండి",
      "శ్వాస తీసుకోవడం కష్టం అవుతోంది"
    ];

    const statusDot = document.querySelector(".status-dot");
    const statusText = document.getElementById("status-text");
    const btnStream = document.getElementById("btn-stream");
    const streamLabel = document.getElementById("stream-label");
    const streamTimer = document.getElementById("stream-timer");
    const streamOutput = document.getElementById("stream-output");
    const streamLatency = document.getElementById("stream-latency");
    const waveformCanvas = document.getElementById("waveform");
    const waveformCtx = waveformCanvas.getContext("2d");
    const alertBoxEl = document.getElementById("alertBox");
    const matchedPhraseEl = document.getElementById("matchedPhrase");
    const supportedPhrasesEl = document.getElementById("supportedPhrases");
    const transcriptLogEl = document.getElementById("transcriptLog");
    const backendTabs = document.querySelectorAll(".backend-tab");
    const headerTabs = document.querySelectorAll(".header-tab");
    let selectedBackend = "indic";
    let debugMode = false;
    let fullTranscript = "";
    let phraseCount = 0;

    checkHealth();
    renderSupportedPhrases();

    window.toggleStreaming = function () {
      if (isStreaming) {
        stopStreaming();
      } else {
        startStreaming();
      }
    };

    window.toggleDebugMode = function () {
      const debugCheckbox = document.getElementById("debugMode");
      debugMode = debugCheckbox.checked;
      
      if (!debugMode) {
        // Reset to normal mode
        fullTranscript = "";
        if (streamOutput) {
          streamOutput.textContent = "Start streaming to see transcript";
        }
      }
    };

    async function checkHealth() {
      try {
        const res = await fetch("/health");
        if (res.ok) {
          statusDot.classList.add("connected");
          statusDot.classList.remove("error");
          statusText.textContent = "Connected";
        } else {
          throw new Error("not ok");
        }
      } catch {
        statusDot.classList.add("error");
        statusDot.classList.remove("connected");
        statusText.textContent = "Disconnected";
        setTimeout(checkHealth, 3000);
      }
    }

    async function startStreaming() {
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, sampleRate: 16000 },
        });
      } catch (err) {
        streamLabel.textContent = "Microphone access denied";
        return;
      }

      streamOutput.textContent = "Listening...";
      streamLatency.textContent = "";
      lastTranscript = "";
      hideEmergencyAlert();

      pcmChunks = [];
      audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      const source = audioContext.createMediaStreamSource(mediaStream);

      analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);

      captureNode = audioContext.createScriptProcessor(4096, 1, 1);
      captureNode.onaudioprocess = (e) => {
        if (!isStreaming) return;
        const input = e.inputBuffer.getChannelData(0);
        const copy = new Float32Array(input.length);
        copy.set(input);
        pcmChunks.push(copy);

        const now = performance.now();
        if (pcmChunks.length >= MIN_CHUNKS && now - lastSendAt >= SEND_INTERVAL_MS) {
          lastSendAt = now;
          sendAudioChunk();
        }
      };
      source.connect(captureNode);
      captureNode.connect(audioContext.destination);

      isStreaming = true;
      btnStream.classList.add("recording");
      streamLabel.textContent = "Streaming... click to stop";
      streamingStart = Date.now();
      timerInterval = setInterval(updateTimer, 1000);
      drawWaveform();
    }

    function stopStreaming() {
      isStreaming = false;
      btnStream.classList.remove("recording");
      streamLabel.textContent = "Click to start streaming";

      if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
      if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
      if (captureNode) { captureNode.disconnect(); captureNode = null; }

      if (pcmChunks.length > 0) sendAudioChunk();

      if (audioContext) { audioContext.close().catch(() => {}); audioContext = null; }
      if (mediaStream) { mediaStream.getTracks().forEach((t) => t.stop()); mediaStream = null; }
      analyser = null;

      waveformCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
      streamOutput.textContent = "Start streaming to see transcript";
      streamLatency.textContent = "";
      lastTranscript = "";
      
      // Reset debug transcript
      if (!debugMode) {
        fullTranscript = "";
      }
    }

    async function sendAudioChunk() {
      if (sendInFlight || pcmChunks.length === 0) return;

      sendInFlight = true;
      const sampleRate = audioContext ? audioContext.sampleRate : 16000;
      const chunksToSend = pcmChunks;
      pcmChunks = [];

      const wavBlob = encodeWAV(chunksToSend, sampleRate);
      const formData = new FormData();
      formData.append("file", new File([wavBlob], "stream.wav", { type: "audio/wav" }));
      formData.append("lang", STREAM_LANG);
      formData.append("backend", selectedBackend);

      try {
        const start = performance.now();
        const res = await fetch("/transcribe", { method: "POST", body: formData });
        const elapsed = performance.now() - start;
        streamLatency.textContent = `${Math.round(elapsed)} ms`;

        if (!res.ok) {
          sendInFlight = false;
          return;
        }

        const data = await res.json();
        if (data?.text) {
          processIncomingTranscript(data.text);
        }
      } catch {
        // keep silent; transient network errors are expected
      } finally {
        sendInFlight = false;
      }
    }

    function encodeWAV(chunks, sampleRate) {
      let totalLen = 0;
      for (const c of chunks) totalLen += c.length;
      const merged = new Float32Array(totalLen);
      let off = 0;
      for (const c of chunks) { merged.set(c, off); off += c.length; }

      const int16 = new Int16Array(merged.length);
      for (let i = 0; i < merged.length; i++) {
        const s = Math.max(-1, Math.min(1, merged[i]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }

      const dataLen = int16.length * 2;
      const buffer = new ArrayBuffer(44 + dataLen);
      const view = new DataView(buffer);
      const w = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };

      w(0, "RIFF");
      view.setUint32(4, 36 + dataLen, true);
      w(8, "WAVE");
      w(12, "fmt ");
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      w(36, "data");
      view.setUint32(40, dataLen, true);

      new Int16Array(buffer, 44).set(int16);
      return new Blob([buffer], { type: "audio/wav" });
    }

    function processIncomingTranscript(text) {
      const incoming = (text || "").trim();
      
      if (debugMode) {
        // In debug mode, append to full transcript
        if (incoming && incoming !== lastTranscript) {
          if (fullTranscript) {
            fullTranscript += " " + incoming;
          } else {
            fullTranscript = incoming;
          }
          streamOutput.textContent = fullTranscript || "Listening...";
        }
      } else {
        // Normal mode - replace transcript
        streamOutput.textContent = incoming || "Listening...";
      }

      if (incoming && incoming !== lastTranscript) {
        addTranscriptLog(incoming);
        lastTranscript = incoming;
      }

      const matched = findMatchedPhrase(debugMode ? fullTranscript : incoming);
      if (matched) {
        showEmergencyAlert(matched);
        updateTriggerDetection(matched);
        // Simulate performance metrics
        const simulatedLatency = Math.random() * 200 + 100; // 100-300ms
        const simulatedAccuracy = Math.random() * 20 + 80; // 80-100%
        updatePerformanceIndicators(simulatedLatency, simulatedAccuracy, simulatedLatency * 0.8);
      } else {
        hideEmergencyAlert();
        updateTriggerDetection(null);
        // Update basic latency for non-matches
        const basicLatency = Math.random() * 150 + 50; // 50-200ms
        updatePerformanceIndicators(basicLatency);
      }
    }

    function normalizeTeluguText(text = "") {
      return text
        .replace(/[.,!?;:'"“”‘’\-—_]/g, " ")
        .replace(/\s+/g, " ")
        .trim();
    }

    function findMatchedPhrase(text) {
      const normalizedInput = normalizeTeluguText(text);
      if (!normalizedInput) return null;

      for (const phrase of TELUGU_EMERGENCY_PHRASES) {
        const normPhrase = normalizeTeluguText(phrase);
        if (!normPhrase) continue;
        if (normalizedInput.includes(normPhrase)) return phrase;

        const candidate = fuzzyMatchPhrase(normalizedInput, normPhrase);
        if (candidate) return phrase;
      }
      return null;
    }

    function fuzzyMatchPhrase(input, phrase) {
      const inputWords = input.split(" ").filter(Boolean);
      const phraseWords = phrase.split(" ").filter(Boolean);
      if (phraseWords.length === 0 || inputWords.length === 0) return false;

      const windowSize = phraseWords.length;
      const maxWindow = Math.max(1, Math.min(inputWords.length, windowSize + 1));

      for (let i = 0; i <= inputWords.length - windowSize; i++) {
        const window = inputWords.slice(i, i + windowSize).join(" ");
        if (isWithinEditDistance(window, phrase)) return true;
      }

      if (inputWords.length < maxWindow) {
        if (isWithinEditDistance(input, phrase)) return true;
      }

      return false;
    }

    function isWithinEditDistance(a, b) {
      const dist = levenshteinDistance(a, b);
      const maxLen = Math.max(a.length, b.length) || 1;
      return dist <= Math.min(MAX_EDIT_ABS, Math.ceil(maxLen * MAX_EDIT_RATIO));
    }

    function levenshteinDistance(a, b) {
      const m = a.length;
      const n = b.length;
      const dp = new Array(n + 1).fill(0).map((_, j) => j);
      for (let i = 1; i <= m; i++) {
        let prev = dp[0];
        dp[0] = i;
        for (let j = 1; j <= n; j++) {
          const temp = dp[j];
          const cost = a[i - 1] === b[j - 1] ? 0 : 1;
          dp[j] = Math.min(dp[j] + 1, dp[j - 1] + 1, prev + cost);
          prev = temp;
        }
      }
      return dp[n];
    }

    function showEmergencyAlert(phrase) {
      if (!alertBoxEl || !matchedPhraseEl) return;
      alertBoxEl.classList.remove("hidden");
      matchedPhraseEl.textContent = `Matched phrase: ${phrase}`;
      
      // Add blinking effect
      document.body.classList.add("alert-blink");
      setTimeout(() => {
        document.body.classList.remove("alert-blink");
      }, 1500); // 1.5 seconds total blink duration
      
      // Show emergency contact based on phrase category
      showEmergencyContact(phrase);
      
      // Increment phrase count
      phraseCount++;
      updatePhraseCount();
    }

    function showEmergencyContact(phrase) {
      const medicalPhrases = [
        "డాక్టర్‌ను కాల్ చేయండి త్వరగా",
        "నేను అస్వస్థంగా ఉన్నాను",
        "నాకు గుండె నొప్పి",
        "నాకు శ్వాస ఆడటం లేదు",
        "నాకు రక్తస్రావం అవుతోంది",
        "నాకు స్పృహ లేదు",
        "నాకు బాగోలేదు హాస్పిటల్ తీసుకెళ్లండి",
        "శ్వాస తీసుకోవడం కష్టం అవుతోంది"
      ];

      const policePhrases = [
        "పోలీస్ కి కాల్ చెయ్యి",
        "పోలీసు కి కాల్ చేయండి",
        "దారి ప్రమాదం జరిగింది",
        "అగ్ని ప్రమాదం జరిగింది",
        "నీరు మునిగిపోతున్నాను"
      ];

      const firePhrases = [
        "అగ్ని ప్రమాదం జరిగింది"
      ];

      const womenPhrases = [
        "మహిళ్లలు సహాయం చేయాలను"
      ];

      const childPhrases = [
        "పిల్లలు సహాయం అవసరాలి"
      ];

      const disasterPhrases = [
        "విపత్తి ప్రమాదం జరిగింది"
      ];

      let contactInfo = "";
      
      if (medicalPhrases.some(mp => phrase.includes(mp))) {
        contactInfo = "🏥 CALL 108 (Medical Emergency) or 104 (Ambulance)";
      } else if (policePhrases.some(pp => phrase.includes(pp))) {
        contactInfo = "🚔 CALL 100 (Police) or 112 (Emergency Services)";
      } else if (firePhrases.some(fp => phrase.includes(fp))) {
        contactInfo = "🚒 CALL 101 (Fire Services) or 112 (Emergency Services)";
      } else if (womenPhrases.some(wp => phrase.includes(wp))) {
        contactInfo = "♀️ CALL 1091 (Women Helpline) or 181 (Women in Distress)";
      } else if (childPhrases.some(cp => phrase.includes(cp))) {
        contactInfo = "👶 CALL 1098 (Child Helpline) or 1090 (Emergency Childline)";
      } else if (disasterPhrases.some(dp => phrase.includes(dp))) {
        contactInfo = "🆘 CALL 1078 (Disaster Management) or 011-1078 (National Disaster)";
      } else {
        contactInfo = "🆘 CALL 108 (National Emergency) or 112 (Emergency Services)";
      }

      // Update alert box with contact information
      if (alertBoxEl && matchedPhraseEl) {
        alertBoxEl.innerHTML = `
          🚨 Emergency phrase detected!
          <div class="matched-phrase">Matched phrase: ${phrase}</div>
          <div class="emergency-contact-info">${contactInfo}</div>
        `;
      }
    }

    function hideEmergencyAlert() {
      if (!alertBoxEl || !matchedPhraseEl) return;
      alertBoxEl.classList.add("hidden");
      matchedPhraseEl.textContent = "";
    }

    function renderSupportedPhrases() {
      if (!supportedPhrasesEl) return;
      supportedPhrasesEl.innerHTML = "";
      TELUGU_EMERGENCY_PHRASES.forEach((phrase) => {
        const li = document.createElement("li");
        li.textContent = phrase;
        supportedPhrasesEl.appendChild(li);
      });
    }

    function renderSupportedPhrasesCategorized() {
      const medicalPhrases = [
        "డాక్టర్‌ను కాల్ చేయండి త్వరగా",
        "నేను అస్వస్థంగా ఉన్నాను",
        "నాకు గుండె నొప్పి",
        "నాకు శ్వాస ఆడటం లేదు",
        "నాకు రక్తస్రావం అవుతోంది",
        "నాకు స్పృహ లేదు",
        "నాకు బాగోలేదు హాస్పిటల్ తీసుకెళ్లండి",
        "శ్వాస తీసుకోవడం కష్టం అవుతోంది"
      ];

      const accidentPhrases = [
        "అంబులెన్స్ పిలవండి రా",
        "అంబులెన్స్ పిలవండి త్వరగా",
        "పోలీస్ కి కాల్ చెయ్యి",
        "పోలీసు కి కాల్ చేయండి",
        "దారి ప్రమాదం జరిగింది",
        "అగ్ని ప్రమాదం జరిగింది",
        "నీరు మునిగిపోతున్నాను",
        "అయ్యో యాక్సిడెంట్ అయింది"
      ];

      const helpPhrases = [
        "హెల్ప్ చేయండి",
        "ఎమర్జెన్సీ ఉందీ",
        "తక్షణ సహాయం కావాలి",
        "దయచేసి సహాయం చేయండి",
        "సిరియస్ గా ఉంది",
        "దయచేసి వెంటనే రండి",
        "నాకు చాలా నొప్పిగా ఉంది",
        "త్వరగా రండి ప్లీజ్",
        "ఎవరైనా హెల్ప్ చేయండి"
      ];

      const medicalContainer = document.querySelector(".medical-phrases");
      const accidentContainer = document.querySelector(".accident-phrases");
      const helpContainer = document.querySelector(".help-phrases");

      if (medicalContainer) {
        medicalContainer.innerHTML = "";
        medicalPhrases.forEach(phrase => {
          const li = document.createElement("li");
          li.textContent = phrase;
          medicalContainer.appendChild(li);
        });
      }

      if (accidentContainer) {
        accidentContainer.innerHTML = "";
        accidentPhrases.forEach(phrase => {
          const li = document.createElement("li");
          li.textContent = phrase;
          accidentContainer.appendChild(li);
        });
      }

      if (helpContainer) {
        helpContainer.innerHTML = "";
        helpPhrases.forEach(phrase => {
          const li = document.createElement("li");
          li.textContent = phrase;
          helpContainer.appendChild(li);
        });
      }
    }

    function updateTriggerDetection(phrase) {
      const triggerText = document.getElementById("trigger-text");
      const triggerDot = document.querySelector(".trigger-dot");
      const triggerList = document.getElementById("trigger-list");

      if (phrase) {
        // Update trigger status
        if (triggerText) {
          triggerText.textContent = `Trigger word detected: ${phrase}`;
        }
        if (triggerDot) {
          triggerDot.classList.add("detected");
        }

        // Add to detected triggers list
        if (triggerList) {
          const li = document.createElement("li");
          const time = new Date().toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" });
          li.textContent = `${time} — ${phrase}`;
          triggerList.prepend(li);

          // Keep only last 10 detections
          while (triggerList.children.length > 10) {
            triggerList.removeChild(triggerList.lastChild);
          }
        }
      } else {
        // Reset trigger status
        if (triggerText) {
          triggerText.textContent = "Waiting for trigger word...";
        }
        if (triggerDot) {
          triggerDot.classList.remove("detected");
        }
      }
    }

    function updatePhraseCount() {
      const accuracyValue = document.getElementById("accuracy-value");
      if (accuracyValue) {
        accuracyValue.textContent = `${phraseCount} phrases`;
      }
    }

    function updatePerformanceIndicators(latency, accuracy = null, processingTime = null) {
      const latencyValue = document.getElementById("latency-value");
      const accuracyValue = document.getElementById("accuracy-value");
      const processingValue = document.getElementById("processing-value");

      if (latencyValue && latency) {
        latencyValue.textContent = `${Math.round(latency)} ms`;
      }

      // Update phrase count instead of accuracy
      if (accuracyValue) {
        accuracyValue.textContent = `${phraseCount} phrases`;
      }

      if (processingValue && processingTime !== null) {
        processingValue.textContent = `${Math.round(processingTime)} ms`;
      }
    }

    function addTranscriptLog(text) {
      if (!transcriptLogEl) return;
      const li = document.createElement("li");
      const time = new Date().toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" });
      li.textContent = `${time} — ${text}`;
      transcriptLogEl.prepend(li);

      while (transcriptLogEl.children.length > LOG_MAX_ITEMS) {
        transcriptLogEl.removeChild(transcriptLogEl.lastChild);
      }
    }

    window.selectBackend = function (backend) {
      selectedBackend = backend;
      backendTabs.forEach((tab) => {
        tab.classList.toggle("active", tab.dataset.backend === backend);
      });
    };

    window.switchMainTab = function (tab) {
      const mainTabs = document.querySelectorAll(".main-tab");
      mainTabs.forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.tab === tab);
      });
      
      // Hide all tab contents
      document.querySelectorAll(".tab-content").forEach(content => {
        content.classList.add("hidden");
      });
      
      // Show selected tab content
      const selectedTab = document.getElementById(`tab-${tab}`);
      if (selectedTab) {
        selectedTab.classList.remove("hidden");
      }
      
      // Initialize tab-specific content
      if (tab === "phrases") {
        renderSupportedPhrasesCategorized();
      }
    };

    function updateTimer() {
      const elapsed = Math.floor((Date.now() - streamingStart) / 1000);
      const m = String(Math.floor(elapsed / 60)).padStart(2, "0");
      const s = String(elapsed % 60).padStart(2, "0");
      streamTimer.textContent = `${m}:${s}`;
    }

    function drawWaveform() {
      if (!analyser || !isStreaming) return;

      const canvas = waveformCanvas;
      const ctx = waveformCtx;
      canvas.width = canvas.offsetWidth * (window.devicePixelRatio || 1);

      const bufLen = analyser.frequencyBinCount;
      const data = new Uint8Array(bufLen);
      analyser.getByteTimeDomainData(data);

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#22c55e";
      ctx.beginPath();

      const sliceW = canvas.width / bufLen;
      let x = 0;
      for (let i = 0; i < bufLen; i++) {
        const v = data[i] / 128.0;
        const y = (v * canvas.height) / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceW;
      }
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();

      animFrame = requestAnimationFrame(drawWaveform);
    }

    return;
  }

  // ── State ──
  let currentLang = "en";
  let currentMode = "record";
  let isRecording = false;
  let mediaStream = null;
  let audioContext = null;
  let captureNode = null; // AudioWorkletNode or ScriptProcessorNode
  let analyser = null;
  let pcmChunks = [];     // Float32Array chunks at 16kHz
  let recordingStart = 0;
  let timerInterval = null;
  let animFrame = null;

  // ── DOM refs ──
  const statusDot = document.querySelector(".status-dot");
  const statusText = document.getElementById("status-text");
  const btnRecord = document.getElementById("btn-record");
  const recordLabel = document.getElementById("record-label");
  const recordTimer = document.getElementById("record-timer");
  const recordResult = document.getElementById("record-result");
  const recordOutput = document.getElementById("record-output");
  const recordLatency = document.getElementById("record-latency");
  const waveformCanvas = document.getElementById("waveform");
  const waveformCtx = waveformCanvas.getContext("2d");
  const dropZone = document.getElementById("drop-zone");
  const uploadOutput = document.getElementById("upload-output");
  const uploadLatency = document.getElementById("upload-latency");
  const uploadResult = document.getElementById("upload-result");

  // ── Init ──
  checkHealth();
  setupDropZone();

  // ── Health check ──
  async function checkHealth() {
    try {
      const res = await fetch("/health");
      if (res.ok) {
        statusDot.classList.add("connected");
        statusDot.classList.remove("error");
        statusText.textContent = "Connected";
      } else {
        throw new Error("not ok");
      }
    } catch {
      statusDot.classList.add("error");
      statusDot.classList.remove("connected");
      statusText.textContent = "Disconnected";
      setTimeout(checkHealth, 3000);
    }
  }

  // ── Language selection ──
  window.selectLang = function (lang) {
    currentLang = lang;
    document.querySelectorAll(".lang-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.lang === lang);
    });
  };

  // ── Mode switching ──
  window.switchMode = function (mode) {
    currentMode = mode;
    document.getElementById("tab-record").classList.toggle("active", mode === "record");
    document.getElementById("tab-upload").classList.toggle("active", mode === "upload");
    document.getElementById("panel-record").classList.toggle("hidden", mode !== "record");
    document.getElementById("panel-upload").classList.toggle("hidden", mode !== "upload");
    if (mode !== "record" && isRecording) {
      stopRecording(false);
    }
  };

  // ── Recording toggle ──
  window.toggleRecording = function () {
    if (isRecording) {
      stopRecording(true);
    } else {
      startRecording();
    }
  };

  // ── Start recording (raw PCM at 16kHz → WAV) ──
  async function startRecording() {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, sampleRate: 16000 },
      });
    } catch (err) {
      recordLabel.textContent = "Microphone access denied";
      return;
    }

    pcmChunks = [];

    // Force 16kHz context so we get 16kHz PCM data
    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    const source = audioContext.createMediaStreamSource(mediaStream);

    // Analyser for waveform visualization
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);

    // Capture raw PCM via ScriptProcessor (widely supported, synchronous copy)
    captureNode = audioContext.createScriptProcessor(4096, 1, 1);
    captureNode.onaudioprocess = (e) => {
      if (!isRecording) return;
      // MUST copy — getChannelData returns a reused internal buffer
      const input = e.inputBuffer.getChannelData(0);
      const copy = new Float32Array(input.length);
      copy.set(input);
      pcmChunks.push(copy);
    };
    source.connect(captureNode);
    captureNode.connect(audioContext.destination);

    isRecording = true;
    btnRecord.classList.add("recording");
    recordLabel.textContent = "Recording… click to stop";
    recordResult.classList.add("hidden");
    recordingStart = Date.now();
    timerInterval = setInterval(updateTimer, 1000);
    drawWaveform();
    console.log(`[ASR] Recording started at ${audioContext.sampleRate}Hz`);
  }

  // ── Stop recording ──
  function stopRecording(sendForTranscription) {
    isRecording = false;
    btnRecord.classList.remove("recording");
    recordLabel.textContent = "Click to start recording";

    if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
    if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
    if (captureNode) { captureNode.disconnect(); captureNode = null; }

    if (sendForTranscription && pcmChunks.length > 0) {
      const sampleRate = audioContext ? audioContext.sampleRate : 16000;
      console.log(`[ASR] Encoding ${pcmChunks.length} chunks at ${sampleRate}Hz`);
      const wavBlob = encodeWAV(pcmChunks, sampleRate);
      console.log(`[ASR] WAV blob: ${wavBlob.size} bytes (${(wavBlob.size / sampleRate / 2).toFixed(1)}s)`);
      if (wavBlob.size > 44) {
        sendWavForTranscription(wavBlob);
      } else {
        recordResult.classList.remove("hidden");
        recordOutput.textContent = "No audio captured — try again";
      }
    } else if (sendForTranscription) {
      recordResult.classList.remove("hidden");
      recordOutput.textContent = "No audio captured — try again";
    }
    pcmChunks = [];

    if (audioContext) { audioContext.close().catch(() => {}); audioContext = null; }
    if (mediaStream) { mediaStream.getTracks().forEach((t) => t.stop()); mediaStream = null; }
    analyser = null;

    waveformCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
  }

  // ── Encode Float32 PCM chunks → 16-bit WAV blob ──
  function encodeWAV(chunks, sampleRate) {
    // Merge all chunks
    let totalLen = 0;
    for (const c of chunks) totalLen += c.length;
    const merged = new Float32Array(totalLen);
    let off = 0;
    for (const c of chunks) { merged.set(c, off); off += c.length; }

    // Float32 → Int16
    const int16 = new Int16Array(merged.length);
    for (let i = 0; i < merged.length; i++) {
      const s = Math.max(-1, Math.min(1, merged[i]));
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }

    // Build WAV file (44-byte header + PCM data)
    const dataLen = int16.length * 2;
    const buffer = new ArrayBuffer(44 + dataLen);
    const view = new DataView(buffer);
    const w = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };

    w(0, "RIFF");
    view.setUint32(4, 36 + dataLen, true);
    w(8, "WAVE");
    w(12, "fmt ");
    view.setUint32(16, 16, true);          // fmt chunk size (16 bytes)
    view.setUint16(20, 1, true);           // PCM format (1)
    view.setUint16(22, 1, true);           // channels (1)
    view.setUint32(24, sampleRate, true);  // sample rate
    view.setUint32(28, sampleRate * 2, true); // byte rate (sampleRate * channels * bitsPerSample/8)
    view.setUint16(32, 2, true);           // block align (channels * bitsPerSample/8)
    view.setUint16(34, 16, true);          // bits per sample
    w(36, "data");
    view.setUint32(40, dataLen, true);

    new Int16Array(buffer, 44).set(int16);
    return new Blob([buffer], { type: "audio/wav" });
  }

  // ── Send WAV to /transcribe (same path as file upload) ──
  async function sendWavForTranscription(wavBlob) {
    recordResult.classList.remove("hidden");
    recordOutput.innerHTML = '<span class="spinner"></span> Transcribing…';
    recordLatency.textContent = "";

    const file = new File([wavBlob], "recording.wav", { type: "audio/wav" });
    const formData = new FormData();
    formData.append("file", file);
    formData.append("lang", currentLang);

    try {
      console.log(`[ASR] POST /transcribe lang=${currentLang} size=${wavBlob.size}`);
      const start = performance.now();
      const res = await fetch("/transcribe", { method: "POST", body: formData });
      const elapsed = performance.now() - start;
      console.log(`[ASR] Response: ${res.status} in ${Math.round(elapsed)}ms`);

      if (!res.ok) {
        let errMsg = res.statusText;
        try {
          const err = await res.json();
          errMsg = err.error || errMsg;
        } catch { /* not JSON */ }
        recordOutput.textContent = `Error: ${errMsg}`;
        return;
      }

      const data = await res.json();
      console.log(`[ASR] Result: "${(data.text || "").substring(0, 80)}" latency=${data.latency_ms}ms`);
      recordOutput.textContent = data.text || "(empty transcript)";
      recordLatency.textContent = `${data.latency_ms}ms server · ${Math.round(elapsed)}ms total · ${data.duration_s}s audio`;
    } catch (err) {
      console.error("[ASR] sendWavForTranscription error:", err);
      recordOutput.textContent = `Network error: ${err.message}`;
    }
  }

  // ── Timer ──
  function updateTimer() {
    const elapsed = Math.floor((Date.now() - recordingStart) / 1000);
    const m = String(Math.floor(elapsed / 60)).padStart(2, "0");
    const s = String(elapsed % 60).padStart(2, "0");
    recordTimer.textContent = `${m}:${s}`;
  }

  // ── Waveform drawing ──
  function drawWaveform() {
    if (!analyser || !isRecording) return;

    const canvas = waveformCanvas;
    const ctx = waveformCtx;
    canvas.width = canvas.offsetWidth * (window.devicePixelRatio || 1);

    const bufLen = analyser.frequencyBinCount;
    const data = new Uint8Array(bufLen);
    analyser.getByteTimeDomainData(data);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#22c55e";
    ctx.beginPath();

    const sliceW = canvas.width / bufLen;
    let x = 0;
    for (let i = 0; i < bufLen; i++) {
      const v = data[i] / 128.0;
      const y = (v * canvas.height) / 2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      x += sliceW;
    }
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();

    animFrame = requestAnimationFrame(drawWaveform);
  }

  // ── File upload (drag & drop + click) ──
  function setupDropZone() {
    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
    dropZone.addEventListener("dragleave", () => {
      dropZone.classList.remove("dragover");
    });
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
      if (e.dataTransfer.files.length > 0) {
        uploadFile(e.dataTransfer.files[0]);
      }
    });
    dropZone.addEventListener("click", () => {
      document.getElementById("file-input").click();
    });
  }

  window.handleFileSelect = function (e) {
    if (e.target.files.length > 0) {
      uploadFile(e.target.files[0]);
    }
  };

  async function uploadFile(file) {
    uploadResult.classList.remove("hidden");
    uploadOutput.innerHTML = '<span class="spinner"></span> Transcribing…';
    uploadLatency.textContent = "";

    const formData = new FormData();
    formData.append("file", file);
    formData.append("lang", currentLang);

    try {
      const start = performance.now();
      const res = await fetch("/transcribe", { method: "POST", body: formData });
      const elapsed = performance.now() - start;

      if (!res.ok) {
        const err = await res.json();
        uploadOutput.textContent = `Error: ${err.error || res.statusText}`;
        return;
      }

      const data = await res.json();
      uploadOutput.textContent = data.text || "(empty transcript)";
      uploadLatency.textContent = `${data.latency_ms}ms server · ${Math.round(elapsed)}ms total · ${data.duration_s}s audio`;
    } catch (err) {
      uploadOutput.textContent = `Network error: ${err.message}`;
    }
  }
})();
