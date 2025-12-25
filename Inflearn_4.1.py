"""
Streamlit 설치와 user message 작성
1. Page Config 설정
2. 제목 설정
3. 채팅 입력창의 캡션 설정
4. 입력 채팅 저장을 위한 Session State 활용
"""

import streamlit as st

st.set_page_config(page_title="소득세 챗봇", page_icon="🤖")

# Streamlit에서 Title을 "h1"으로 자동 설정
st.title("🤖 소득세 챗봇")
st.caption("소득세와 관련된 모든 것을 답해드립니다!")

# message_list: 입력된 채팅 내용 저장
if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_question := st.chat_input(placeholder="소득세와 관련된 궁금한 내용들을 말씀해주세요."):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.chat_message("ai"):
        st.write("AI 메세지")
    st.session_state.message_list.append({"role": "ai", "content": "AI 메세지"})