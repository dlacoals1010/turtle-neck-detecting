import time
from collections import deque

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import WebRtcMode, VideoTransformerBase, webrtc_streamer
import streamlit.components.v1 as components


# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="AI Posture Calibration", page_icon="🐢")

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
st.write(
    "Step 1. 카메라를 보고 **가장 바른 자세** 유지\n"
    "Step 2. Calibration 버튼 클릭\n"
    "Step 3. 그 자세를 기준으로 Good / Mild / Severe를 실시간 분석합니다."
)

mp_pose = mp.solutions.pose


# -----------------------------
# 브라우저 음성 안내 (speechSynthesis)
# -----------------------------
def speak(text: str):
    """
    브라우저의 speechSynthesis API를 사용해 텍스트를 읽어줌.
    (추가 라이브러리 / API 키 필요 없음)
    """
    safe = text.replace('"', '\\"')
    components.html(
        f"""
        <script>
        var utter = new SpeechSynthesisUtterance("{safe}");
        speechSynthesis.speak(utter);
        </script>
        """,
        height=0,
        width=0,
    )


# -----------------------------
# Feature extraction
# -----------------------------
def extract_features_from_landmarks(landmarks, img_shape):
    l_sh = landmarks[11]
    r_sh = landmarks[12]
    center_x = (l_sh.x + r_sh.x) / 2
    center_y = (l_sh.y + r_sh.y) / 2
    width = np.linalg.norm(
        np.array([l_sh.x, l_sh.y]) - np.array([r_sh.x, r_sh.y])
    )
    if width == 0:
        width = 1.0

    indices = [0, 2, 5, 7, 8, 11, 12]  # 코, 눈, 귀, 어깨
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
# Distance -> fuzzy probs
# -----------------------------
def distance_to_probs(dist, t_good=0.12, t_mild=0.28):
    d = float(dist)

    # good: 0에서 t_good까지 선형으로 줄어듦
    good_score = max(0.0, 1.0 - d / max(t_good, 1e-6))

    # mild: t_good 근처에서 높고 양쪽에서 0
    if d <= t_good:
        mild_score = d / max(t_good, 1e-6)
    elif d <= t_mild:
        mild_score = 1.0 - (d - t_good) / max(t_mild - t_good, 1e-6)
    else:
        mild_score = 0.0

    # severe: t_mild 이후부터 커짐
    if d <= t_mild:
        severe_score = 0.0
    else:
        severe_score = min(1.0, (d - t_mild) / max(t_mild, 1e-6))

    scores = {"good": good_score, "mild": mild_score, "severe": severe_score}
    s = sum(scores.values())
    if s == 0:
        return {"good": 1 / 3, "mild": 1 / 3, "severe": 1 / 3}
    for k in scores:
        scores[k] /= s
    return scores


# -----------------------------
# Video Processor Class
# -----------------------------
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            model_complexity=1,
        )

        # calibration 관련
        self.baseline = None
        self.collecting = False
        self.baseline_buffer = []

        self.distance_hist = deque(maxlen=10)

        self.latest_pred = None
        self.latest_probs = {"good": 0.0, "mild": 0.0, "severe": 0.0}
        self.latest_dist = 0.0

    def start_calibration(self):
        self.collecting = True
        self.baseline_buffer = []
        self.baseline = None
        self.distance_hist.clear()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if results.pose_landmarks:
            feats, pts = extract_features_from_landmarks(
                results.pose_landmarks.landmark, img.shape
            )

            # 1) calibration 단계
            if self.collecting:
                self.baseline_buffer.append(feats)
                if len(self.baseline_buffer) >= 20:
                    self.baseline = np.mean(self.baseline_buffer, axis=0)
                    self.collecting = False

                # 점 찍기
                for x, y in pts:
                    cv2.circle(img, (x, y), 6, (0, 255, 0), -1)

                return av.VideoFrame.from_ndarray(img, format="bgr24")

            # 2) calibration 완료 후
            if self.baseline is not None:
                diff = np.array(feats) - np.array(self.baseline)
                dist = float(np.linalg.norm(diff))
                self.distance_hist.append(dist)
                self.latest_dist = float(np.mean(self.distance_hist))

                probs = distance_to_probs(self.latest_dist)
                self.latest_probs = probs
                self.latest_pred = max(probs, key=probs.get)

            # 점 찍기
            for x, y in pts:
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["📷 Real-time Calibration", "🖼 Upload (Disabled)"])


# -----------------------------
# TAB 1: Real-time Calibration
# -----------------------------
with tab1:
    st.header("Real-time Webcam (Personal Calibration)")

    col1, col2 = st.columns([2, 1])

    # LEFT — webcam
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
                # calibration 안내 음성
                speak("Starting calibration. Please hold your best posture.")

    # RIGHT — status panel
    with col2:
        st.subheader("Posture Status")

        status_ph = st.empty()

        # bars 상단으로 배치
        st.markdown("#### Good / Mild / Severe (real-time)")
        bar_good = st.empty()
        bar_mild = st.empty()
        bar_severe = st.empty()

        warning_ph = st.empty()


# -----------------------------
# Real-time update loop
# -----------------------------
# 음성 중복 재생 방지용 state
if "last_voice_state" not in st.session_state:
    st.session_state["last_voice_state"] = None

if ctx and ctx.state.playing:
    while True:
        vp = ctx.video_processor
        if vp is None:
            time.sleep(0.1)
            continue

        pred = vp.latest_pred
        probs = vp.latest_probs

        # calibration 상태 안내
        if vp.collecting:
            status_ph.info("🧭 Calibrating... Please hold your best posture.")
        elif vp.baseline is None:
            status_ph.warning("Waiting for calibration...")
        else:
            # 상태 텍스트
            if pred == "good":
                status_ph.markdown(
                    "<p class='good-text'>GOOD 😊</p>", unsafe_allow_html=True
                )
            elif pred == "mild":
                status_ph.markdown(
                    "<p class='mild-text'>MILD 😐</p>", unsafe_allow_html=True
                )
            elif pred == "severe":
                status_ph.markdown(
                    "<p class='severe-text'>SEVERE 🐢</p>", unsafe_allow_html=True
                )

            # bars (숫자 없이)
            bar_good.progress(int(probs["good"] * 100))
            bar_mild.progress(int(probs["mild"] * 100))
            bar_severe.progress(int(probs["severe"] * 100))

            # severe 경고 박스 + 음성 (상태가 바뀔 때만 재생)
            if pred == "severe":
                warning_ph.markdown(
                    """
                    <div class='warning-box'>
                        🚨 <b>Severe Forward Head Posture</b><br>
                        Please straighten your neck.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                warning_ph.empty()

            # 음성 안내 (상태가 바뀔 때만)
            if pred != st.session_state["last_voice_state"]:
                if pred == "good":
                    speak("Good posture.")
                elif pred == "mild":
                    speak("Mild forward head posture.")
                elif pred == "severe":
                    speak("Warning. Severe forward head posture detected.")
                st.session_state["last_voice_state"] = pred

        time.sleep(0.1)


# -----------------------------
# TAB 2: Disabled
# -----------------------------
with tab2:
    st.info("This demo focuses on real-time calibrated posture detection.")
