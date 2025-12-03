import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from collections import deque
import time
import base64
import io
import soundfile as sf


# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="AI Posture Calibration", page_icon="🐢")


# -----------------------------
# TTS (음성 안내) 생성 함수
# -----------------------------
def generate_tts_audio(text):
    """텍스트를 WAV 바이트로 생성해 streamlit에서 재생 가능한 형식으로 변환"""
    try:
        import openai  # Streamlit Cloud에서도 지원됨
        client = openai.OpenAI()

        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text
        )

        audio_bytes = speech.read()
        return audio_bytes
    except Exception:
        return None


def audio_html(audio_bytes):
    """WAV → base64 → HTML audio 태그로 변환"""
    b64 = base64.b64encode(audio_bytes).decode()
    return f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
    """


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .good-text { color: #2ecc71; font-weight: bold; font-size: 22px;}
    .mild-text { color: #f1c40f; font-weight: bold; font-size: 22px;}
    .severe-text { color: #e74c3c; font-weight: bold; font-size: 22px;}
    .warning-box { background-color: #fadbd8; border: 2px solid #e74c3c;
                   padding: 15px; border-radius: 10px; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("🐢 AI Real-time Turtle Neck Calibration System")
st.write("Step 1. 카메라를 보고 **가장 바른 자세** 유지 → Step 2. Calibration 버튼 클릭 → Step 3. 실시간 분석 시작")


mp_pose = mp.solutions.pose


# -----------------------------
# Feature extraction
# -----------------------------
def extract_features_from_landmarks(landmarks, img_shape):
    l_sh = landmarks[11]; r_sh = landmarks[12]
    center_x = (l_sh.x + r_sh.x) / 2
    center_y = (l_sh.y + r_sh.y) / 2
    width = np.linalg.norm(np.array([l_sh.x, l_sh.y]) - np.array([r_sh.x, r_sh.y]))
    if width == 0: width = 1.0

    indices = [0, 2, 5, 7, 8, 11, 12]
    features = []

    h, w, _ = img_shape
    draw_points = []

    for idx in indices:
        lm = landmarks[idx]
        nx = (lm.x - center_x) / width
        ny = (lm.y - center_y) / width
        features.extend([nx, ny])
        draw_points.append((int(lm.x * w), int(lm.y * h)))

    return features, draw_points


# -----------------------------
# Fuzzy probability from distance
# -----------------------------
def distance_to_probs(dist, t_good=0.12, t_mild=0.28):
    d = float(dist)

    good_score = max(0, 1 - d / max(t_good, 1e-6))

    if d <= t_good:
        mild_score = d / max(t_good, 1e-6)
    elif d <= t_mild:
        mild_score = 1 - (d - t_good) / max(t_mild - t_good, 1e-6)
    else:
        mild_score = 0

    if d <= t_mild:
        severe_score = 0
    else:
        severe_score = min(1, (d - t_mild) / max(t_mild, 1e-6))

    scores = {"good": good_score, "mild": mild_score, "severe": severe_score}
    s = sum(scores.values())

    if s == 0:
        return {"good": 1/3, "mild": 1/3, "severe": 1/3}

    for k in scores:
        scores[k] /= s

    return scores


# -----------------------------
# Video Processor Class
# -----------------------------
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False, min_detection_confidence=0.5, model_complexity=1
        )

        self.baseline = None
        self.collecting = False
        self.baseline_buffer = []
        self.distance_hist = deque(maxlen=10)

        self.latest_pred = None
        self.latest_probs = {"good": 0, "mild": 0, "severe": 0}
        self.latest_dist = 0

    def start_calibration(self):
        self.collecting = True
        self.baseline_buffer = []
        self.baseline = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if results.pose_landmarks:
            feats, pts = extract_features_from_landmarks(
                results.pose_landmarks.landmark, img.shape
            )

            # Calibration mode
            if self.collecting:
                self.baseline_buffer.append(feats)
                if len(self.baseline_buffer) >= 20:
                    self.baseline = np.mean(self.baseline_buffer, axis=0)
                    self.collecting = False
                for x, y in pts:
                    cv2.circle(img, (x, y), 6, (0, 255, 0), -1)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            # After calibration
            if self.baseline is not None:
                diff = np.array(feats) - np.array(self.baseline)
                dist = np.linalg.norm(diff)
                self.distance_hist.append(dist)
                self.latest_dist = np.mean(self.distance_hist)

                probs = distance_to_probs(self.latest_dist)
                self.latest_probs = probs
                self.latest_pred = max(probs, key=probs.get)

            # Draw points
            for x, y in pts:
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# -----------------------------
# Layout Tabs
# -----------------------------
tab1, tab2 = st.tabs(["📷 Real-time Calibration", "🖼 Upload (Disabled)"])


# -----------------------------
# TAB 1
# -----------------------------
with tab1:
    st.header("Real-time Webcam (Calibrated)")

    col1, col2 = st.columns([2, 1])

    # -------------------------
    # LEFT — Webcam Stream
    # -------------------------
    with col1:
        ctx = webrtc_streamer(
            key="posture-calib",
            video_processor_factory=VideoProcessor,
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        st.markdown("### Step 1. Hold your best posture")
        if ctx and ctx.video_processor:
            if st.button("📌 Start Calibration"):
                ctx.video_processor.start_calibration()

                audio = generate_tts_audio("Starting calibration. Please hold your best posture.")
                if audio:
                    st.markdown(audio_html(audio), unsafe_allow_html=True)

    # -------------------------
    # RIGHT — Status Panel
    # -------------------------
    with col2:
        st.subheader("Posture Status")

        status_ph = st.empty()

        # Moved bars to the TOP area
        st.markdown("#### Good / Mild / Severe (real-time)")
        bar_good = st.empty()
        bar_mild = st.empty()
        bar_severe = st.empty()

        warning_ph = st.empty()


# -----------------------------
# Real-time Update Loop
# -----------------------------
if ctx and ctx.state.playing:
    while True:
        vp = ctx.video_processor
        if vp is None:
            time.sleep(0.1)
            continue

        pred = vp.latest_pred
        probs = vp.latest_probs

        if vp.collecting:
            status_ph.info("🧭 Calibrating... Please hold your best posture.")
        elif vp.baseline is None:
            status_ph.warning("Waiting for calibration...")
        else:
            # --------------------
            # Status Text
            # --------------------
            if pred == "good":
                status_ph.markdown("<p class='good-text'>GOOD 😊</p>", unsafe_allow_html=True)
            elif pred == "mild":
                status_ph.markdown("<p class='mild-text'>MILD 😐</p>", unsafe_allow_html=True)
            else:
                status_ph.markdown("<p class='severe-text'>SEVERE 🐢</p>", unsafe_allow_html=True)

                audio = generate_tts_audio("Warning. Severe forward head posture detected.")
                if audio:
                    st.markdown(audio_html(audio), unsafe_allow_html=True)

            # --------------------
            # Bars (no % numbers)
            # --------------------
            bar_good.progress(int(probs["good"] * 100))
            bar_mild.progress(int(probs["mild"] * 100))
            bar_severe.progress(int(probs["severe"] * 100))

            # Warning box for severe
            if pred == "severe":
                warning_ph.markdown(
                    """
                    <div class='warning-box'>
                        🚨 <b>Severe Forward Head Posture</b><br>
                        Please straighten your neck.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                warning_ph.empty()

        time.sleep(0.1)


# -----------------------------
# TAB 2 Disabled
# -----------------------------
with tab2:
    st.info("This demo focuses on real-time calibrated posture detection.")
