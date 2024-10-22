# app.py
import streamlit as st
import torch
from transformers import AutoProcessor, AutoModelForTextToSpeech
import soundfile as sf
from github import Github
import os
import datetime
import numpy as np
from scipy.io.wavfile import write
import tempfile

# 페이지 설정
st.set_page_config(
    page_title="TTS 더빙 앱",
    layout="wide"
)

# 깃허브 토큰 설정 (환경 변수에서 가져오기)
@st.cache_resource
def init_github():
    token = os.getenv('GITHUB_TOKEN')
    if token:
        return Github(token)
    return None

# TTS 모델 초기화
@st.cache_resource
def load_model(model_name):
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForTextToSpeech.from_pretrained(model_name)
    return processor, model

def save_to_github(audio_data, filename, g):
    if g is None:
        st.error("GitHub 토큰이 설정되지 않았습니다.")
        return None
    
    try:
        repo = g.get_user().get_repo("tts-recordings")
        date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"recordings/{date_str}_{filename}.wav"
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            write(temp_file.name, 22050, audio_data)
            with open(temp_file.name, 'rb') as f:
                content = f.read()
        
        # GitHub에 업로드
        repo.create_file(
            file_path,
            f"Upload TTS recording {filename}",
            content
        )
        return file_path
    except Exception as e:
        st.error(f"GitHub 저장 실패: {str(e)}")
        return None

def main():
    st.title("🎙️ TTS 더빙 앱")
    
    # 사이드바 설정
    st.sidebar.title("설정")
    
    # 모델 선택
    model_options = {
        "한국어 (여성)": "facebook/mms-tts-kor",
        "한국어 (남성)": "snakers4/silero-models",
        "영어": "facebook/mms-tts-eng",
        "일본어": "facebook/mms-tts-jpn"
    }
    
    selected_model = st.sidebar.selectbox(
        "음성 모델 선택",
        list(model_options.keys())
    )
    
    # 음성 설정
    speed = st.sidebar.slider("음성 속도", 0.5, 2.0, 1.0, 0.1)
    pitch = st.sidebar.slider("음성 높낮이", 0.5, 2.0, 1.0, 0.1)
    
    # 메인 영역
    text_input = st.text_area(
        "텍스트 입력",
        height=150,
        placeholder="여기에 변환하고 싶은 텍스트를 입력하세요..."
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔊 음성 생성", type="primary"):
            if text_input:
                with st.spinner("음성을 생성하는 중..."):
                    try:
                        # 모델 로드
                        processor, model = load_model(model_options[selected_model])
                        
                        # 텍스트를 음성으로 변환
                        inputs = processor(text=text_input, return_tensors="pt")
                        speech = model.generate_speech(
                            inputs["input_ids"], 
                            processor.tokenizer
                        )
                        
                        # 속도와 피치 조절
                        speech = torch.nn.functional.interpolate(
                            speech.unsqueeze(0),
                            scale_factor=1/speed,
                            mode='linear',
                            align_corners=False
                        ).squeeze(0)
                        
                        # 오디오 재생
                        st.audio(speech.numpy(), sample_rate=22050)
                        
                        # 세션 스테이트에 저장
                        st.session_state.last_audio = speech.numpy()
                    except Exception as e:
                        st.error(f"음성 생성 실패: {str(e)}")
            else:
                st.warning("텍스트를 입력해주세요.")
    
    with col2:
        if st.button("💾 GitHub에 저장"):
            if 'last_audio' in st.session_state:
                with st.spinner("GitHub에 저장하는 중..."):
                    g = init_github()
                    if g:
                        file_path = save_to_github(
                            st.session_state.last_audio,
                            "tts_recording",
                            g
                        )
                        if file_path:
                            st.success(f"저장 완료! 경로: {file_path}")
                    else:
                        st.error("GitHub 연동이 필요합니다.")
            else:
                st.warning("먼저 음성을 생성해주세요.")
    
    # 사용 설명
    with st.expander("📖 사용 방법"):
        st.markdown("""
        1. 사이드바에서 원하는 음성 모델을 선택합니다.
        2. 속도와 음높이를 조절합니다.
        3. 텍스트 입력창에 변환하고 싶은 텍스트를 입력합니다.
        4. '음성 생성' 버튼을 눌러 TTS를 실행합니다.
        5. 생성된 음성을 듣고 마음에 들면 'GitHub에 저장' 버튼을 눌러 저장합니다.
        
        주의: GitHub 저장 기능을 사용하려면 환경 변수에 GITHUB_TOKEN을 설정해야 합니다.
        """)

if __name__ == "__main__":
    main()
