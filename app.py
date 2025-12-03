import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import av
import time
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration

# --- Page Configuration ---
st.set_page_config(page_title="AI Posture Correction Pro", page_icon="🐢", layout="wide")

# --- JS & CSS (audio + TTS + 스타일) ---
def get_audio_html():
    js_code = """
        <script>
        // Simple beep alert (for SEVERE)
        function playAlert() {
            var audio = new Audio('https://actions.google.com/sounds/v1/alarms/beep_short.ogg');
            audio.volume = 0.5;
            audio.play();
        }

        // ---- Voice guidance using Web Speech API ----
        window.lastPostureStatus = null;

        function speakPostureStatus(status) {
            if (!('speechSynthesis' in window)) return;

            var text = '';
            if (status === 'GOOD') {
                text = 'Posture is go
