import cv2
import streamlit as st
import numpy as np
from PIL import Image

# 이미지 파일 경로 리스트
image_paths = ["Back.png", "DSUB.png", "Front.png", "RJ45.png"]

# 사이드바를 통해 이미지 선택
selected_image = st.sidebar.selectbox("Select an image", image_paths)

# Canny Edge Detection 파라미터
st.sidebar.subheader("Canny Edge Detection Parameters")
low_threshold = st.sidebar.slider("Low Threshold", 0, 500, 100, key='low_threshold')
high_threshold = st.sidebar.slider("High Threshold", 0, 500, 200, key='high_threshold')

# Adaptive Threshold 파라미터
st.sidebar.subheader("Adaptive Threshold Parameters")
block_size = st.sidebar.slider("Block Size", 1, 10, 5, key='block_size')
constant = st.sidebar.slider("Constant", 0, 20, 5, key='constant')

# 이미지 읽기 및 처리
image = cv2.imread(selected_image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Canny edge detection
edges = cv2.Canny(gray_blur, low_threshold, high_threshold)

# Adaptive Threshold
block_size = block_size * 2 + 3  # 블록 크기는 홀수여야 하므로 변환
adaptive_thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant - 10)

# 결과 출력
col1, col2, col3 = st.columns(3)

with col1:
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Original Image', use_column_width=True)
with col2:
    st.image(edges, caption='Canny Edge Detection', use_column_width=True, channels='GRAY')
with col3:
    st.image(adaptive_thresh, caption='Adaptive Threshold', use_column_width=True, channels='GRAY')
