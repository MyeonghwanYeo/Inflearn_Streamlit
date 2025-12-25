from langchain_upstage import UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import ChatUpstage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langsmith import Client

from config import answer_examples

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    return store[session_id]


def get_database():
    # 임베딩 모델 저장
    embedding = UpstageEmbeddings(
        model='embedding-query'
    )

    database = PineconeVectorStore.from_existing_index(
        index_name='tax-markdown-index',
        embedding=embedding,
    )

    return database


def get_llm(model='solar-mini'):
    llm = ChatUpstage(
        model=model,
    )

    return llm


def get_history_retriever():
    llm = get_llm()
    database = get_database()

    # Retriever 생성
    retriever = database.as_retriever(search_kwargs={'k': 3})

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    return history_aware_retriever


def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 -> 거주자"]

    # 모델이 문서의 정보를 이해할 수 있도록 질문 변경 Prompt 생성
    prompt_dict = ChatPromptTemplate.from_template(f"""
    사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
    만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
    그런 경우에는 질문만 리턴해주세요.
    사전: {dictionary}

    질문: {{question}}
    """)

    llm = get_llm()

    dictionary_chain = prompt_dict | llm | StrOutputParser()

    return dictionary_chain


def get_prompt():
    client = Client()
    prompt = client.pull_prompt("rlm/rag-prompt")

    return prompt


def get_rag_chain():
    llm = get_llm()

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )

    system_prompt = (
        # "You are an assistant for question-answering tasks. "
        # "Use the following pieces of retrieved context to answer "
        # "the question. If you don't know the answer, say that you "
        # "don't know. Use three sentences maximum and keep the "
        # "answer concise."
        # "\n\n"
        # "{context}"
        "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요."
        "아래에 제공된 문서를 활용해서 답변해주시고,"
        "답변을 알 수 없다면 모른다고 답변해주세요."
        "답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해주시고,"
        "2-3 문장정도의 짧은 내용의 답변을 원합니다."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = get_history_retriever()

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')

    return conversational_rag_chain


def get_ai_response(user_input):
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()

    # 최종 체인 생성
    tax_chain = {"input": dictionary_chain} | rag_chain

    # 스트리밍 형식으로 답변 생성
    ai_response = tax_chain.stream(
        {
            "question": user_input
        },

        config=
        {
            "configurable": {"session_id": "abc123"}
        },
    )

    return ai_response