"""
LangChain으로 작성한 코드를 활용한 LLM 답변
1. RAG가 적용된 기존 LLM 코드 불러오기
2. get_ai_message 함수 정의를 통해 AI 메세지 생성
"""

import streamlit as st

from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import ChatUpstage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client

load_dotenv()

st.set_page_config(page_title="소득세 챗봇", page_icon="🤖")

# Streamlit에서 Title을 "h1"으로 자동 설정
st.title("🤖 소득세 챗봇")
st.caption("소득세와 관련된 모든 것을 답해드립니다!")

def get_ai_message(user_input):
    # 임베딩 모델 저장
    embedding = UpstageEmbeddings(
        api_key="up_tknqyPYEHnMeHX0wnaofSJhwhYWRf",
        model='embedding-query'
    )

    database = PineconeVectorStore.from_existing_index(
        index_name='tax-markdown-index',
        embedding=embedding,
    )

    llm = ChatUpstage(
        api_key="up_tknqyPYEHnMeHX0wnaofSJhwhYWRf",
        model='solar-mini'
    )

    dictionary = ["사람을 나타내는 표현 -> 거주자"]

    # 모델이 문서의 정보를 이해할 수 있도록 질문 변경 Prompt 생성
    prompt_dict = ChatPromptTemplate.from_template(f"""
    사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
    만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
    그런 경우에는 질문만 리턴해주세요.
    사전: {dictionary}

    질문: {{question}}
    """)

    dictionary_chain = prompt_dict | llm | StrOutputParser()

    client = Client()
    prompt = client.pull_prompt("rlm/rag-prompt")

    chain = (
            {
                "context": lambda x: database.similarity_search(
                    dictionary_chain.invoke({"question": x["question"]}),
                    k=3),
                "question": dictionary_chain,
            }
            | prompt
            | llm
    )

    ai_message = chain.invoke({"question": user_input}).content

    return ai_message


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
        ai_message = get_ai_message(user_input=user_question)

        with st.chat_message("ai"):
            st.write(ai_message)
        st.session_state.message_list.append({"role": "ai", "content": ai_message})