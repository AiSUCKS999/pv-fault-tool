// Native File and video metadata only: never read or upload the video bytes.
const send = (type, rest = {}) => parent.postMessage({isStreamlitMessage: true, type, ...rest}, "*");
send("streamlit:componentReady", {apiVersion: 1});
send("streamlit:setFrameHeight", {height: 85});
const picker = document.getElementById("videoFile");
const status = document.getElementById("status");
let sequence = 0;
picker.addEventListener("change", () => {
  const file = picker.files[0];
  if (!file) return;
  const selection = ++sequence;
  const data = {selection_id: crypto.randomUUID(), name: file.name, mime: file.type,
    size_bytes: file.size, last_modified_ms: file.lastModified};
  const video = document.createElement("video");
  video.preload = "metadata";
  const url = URL.createObjectURL(file);
  let finished = false;
  const finish = error => {
    if (finished) return;
    finished = true;
    clearTimeout(timer);
    if (!error) {
      data.duration_seconds = Number.isFinite(video.duration) ? video.duration : null;
      data.width = video.videoWidth;
      data.height = video.videoHeight;
    } else data.metadata_error = error;
    URL.revokeObjectURL(url);
    video.removeAttribute("src");
    video.load();
    if (selection !== sequence) return;
    status.textContent = `${file.name} · ${file.size.toLocaleString()} bytes · metadata only`;
    send("streamlit:setComponentValue", {value: data});
  };
  const timer = setTimeout(() => finish("Video duration/dimensions were unavailable within 10 seconds"), 10000);
  video.onloadedmetadata = () => finish("");
  video.onerror = () => finish("Browser could not read this video format; filename and size are retained");
  video.src = url;
});
