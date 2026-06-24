const MUSIC_STATE_KEY = "three_body_global_music_state";
const MUSIC_PROGRESS_KEY = "three_body_global_music_progress";

function readMusicState() {
  try {
    return JSON.parse(localStorage.getItem(MUSIC_STATE_KEY) || "{}");
  } catch {
    return {};
  }
}

function writeMusicState(state) {
  localStorage.setItem(MUSIC_STATE_KEY, JSON.stringify(state));
}

function createMusicToggle() {
  const button = document.createElement("button");
  button.id = "global-music-toggle";
  button.className = "music-toggle";
  button.type = "button";
  button.setAttribute("aria-label", "music on");
  button.setAttribute("title", "music");
  button.innerHTML = `
    <svg class="music-off" viewBox="0 0 24 24" aria-hidden="true">
      <path fill="currentColor" d="M9 18V5l10-2v13h-2V5.45l-6 1.2V18a3 3 0 1 1-2 0Z"/>
      <path fill="currentColor" d="M4.22 4.22 19.78 19.78l-1.41 1.41L2.81 5.64l1.41-1.42Z"/>
    </svg>
    <svg class="music-on" viewBox="0 0 24 24" aria-hidden="true">
      <path fill="currentColor" d="M9 18V5l10-2v13h-2V5.45l-6 1.2V18a3 3 0 1 1-2 0Z"/>
    </svg>
  `;
  document.body.appendChild(button);
  return button;
}

function createAudio() {
  const audio = document.createElement("audio");
  audio.id = "global-music";
  audio.src = "/assets/global-music";
  audio.loop = true;
  audio.preload = "auto";
  audio.volume = 0.42;
  document.body.appendChild(audio);
  return audio;
}

function updateButton(button, isOn) {
  button.classList.toggle("is-on", isOn);
  button.setAttribute("aria-label", isOn ? "music off" : "music on");
}

function persistProgress(audio) {
  if (Number.isFinite(audio.currentTime)) {
    localStorage.setItem(MUSIC_PROGRESS_KEY, String(audio.currentTime));
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const button = createMusicToggle();
  const audio = createAudio();
  const state = readMusicState();
  const savedProgress = Number(localStorage.getItem(MUSIC_PROGRESS_KEY) || "0");

  if (Number.isFinite(savedProgress) && savedProgress > 0) {
    audio.addEventListener(
      "loadedmetadata",
      () => {
        if (audio.duration && savedProgress < audio.duration) {
          audio.currentTime = savedProgress;
        }
      },
      { once: true },
    );
  }

  updateButton(button, Boolean(state.enabled));

  function bindDeferredResume() {
    const resume = () => {
      playMusic({ keepEnabledOnFail: true });
      window.removeEventListener("pointerdown", resume);
      window.removeEventListener("keydown", resume);
    };
    window.addEventListener("pointerdown", resume);
    window.addEventListener("keydown", resume);
  }

  async function playMusic(options = {}) {
    try {
      await audio.play();
      writeMusicState({ enabled: true });
      updateButton(button, true);
    } catch {
      if (options.keepEnabledOnFail) {
        writeMusicState({ enabled: true });
        updateButton(button, true);
        bindDeferredResume();
        return;
      }
      writeMusicState({ enabled: false });
      updateButton(button, false);
    }
  }

  function pauseMusic() {
    persistProgress(audio);
    audio.pause();
    writeMusicState({ enabled: false });
    updateButton(button, false);
  }

  button.addEventListener("click", () => {
    if (audio.paused) {
      playMusic();
    } else {
      pauseMusic();
    }
  });

  audio.addEventListener("timeupdate", () => persistProgress(audio));
  window.addEventListener("pagehide", () => persistProgress(audio));

  if (state.enabled) {
    playMusic({ keepEnabledOnFail: true });
  }
});
