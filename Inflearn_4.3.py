"""
Chat History추가와 streaming 구현
1. LLM관련 코드는 다른 파일에 저장
"""

import streamlit as st

from dotenv import load_dotenv
from llm import get_ai_response

load_dotenv()

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

    # 로딩 표시 생성
    with st.spinner("답변을 생성하는 중입니다."):
        ai_response = get_ai_response(user_input=user_question)

        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
        st.session_state.message_list.append({"role": "ai", "content": ai_message})