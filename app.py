import os
import io
import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent
from langchain.agents.agent import AgentExecutor
# ---------------------------------------------------------
# 0. API KEY 설정 (하드코딩)
# ---------------------------------------------------------
# ⚠️ 아래 따옴표 안에 실제 발급받은 Google API Key를 넣으세요.
# 예: api_key = "AIzaSy..."
import streamlit as st
api_key = st.secrets["GEMINI_API_KEY"]

# ---------------------------------------------------------
# 1. 페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="약국 똑똑이 비서",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. UI 디자인 (CSS 주입)
# ---------------------------------------------------------
def inject_custom_css():
    st.markdown("""
    <style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    [data-testid="stSidebar"] { background-color: #1e293b; border-radius: 20px 0 0 20px; margin-right: 10px; }
    .stButton > button {
        background-color: #3b82f6; color: white; border: none; border-radius: 30px;
        padding: 10px 20px; font-weight: bold; transition: all 0.3s ease; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton > button:hover { background-color: #2563eb; transform: translateY(-2px); }
    .stFileUploader { background-color: #1e293b; border: 2px dashed #475569; border-radius: 20px; padding: 30px; text-align: center; }
    .stChatMessage { background-color: #1e293b; border-radius: 20px; padding: 15px; margin-bottom: 10px; border: 1px solid #334155; }
    th { background-color: #334155; color: #ffffff; font-weight: bold; }
    td { background-color: #1e293b; color: #e2e8f0; }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ---------------------------------------------------------
# 3. LangChain 설정 및 데이터 처리 도구
# ---------------------------------------------------------

# LLM 초기화
@st.cache_resource
def initialize_llm(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        api_key=api_key,
        temperature=0.3
    )

# 데이터 처리 도구
@tool
def analyze_financial_data(question: str):
    """엑셀 데이터를 바탕으로 질문에 답변합니다."""
    try:
        df = st.session_state['df']
        df = df.dropna(subset=['대분류', '금액'])
        df['금액'] = pd.to_numeric(df['금액'], errors='coerce').fillna(0)

        income_df = df[df['대분류'] == '수입']
        fixed_df = df[df['대분류'] == '고정비용']
        drug_df = df[df['대분류'] == '의약품_구입비']

        income_sum = income_df.groupby(['년', '월'])['금액'].sum()
        fixed_sum = fixed_df.groupby(['년', '월'])['금액'].sum()
        drug_sum = drug_df.groupby(['년', '월'])['금액'].sum()

        summary = pd.concat([income_sum, fixed_sum, drug_sum], axis=1)
        summary.columns = ['수입', '고정비용', '의약품_구입비']
        summary = summary.fillna(0)
        summary['순수익'] = summary['수입'] - (summary['고정비용'] + summary['의약품_구입비'])
        summary = summary.astype(int)

        data_context = summary.to_string()
        return f"다음은 회계 데이터 요약입니다:\n{data_context}\n\n질문: {question}"

    except Exception as e:
        return f"데이터 분석 중 오류 발생: {str(e)}"

# ---------------------------------------------------------
# 4. 메인 화면 구성
# ---------------------------------------------------------

st.title("💊 약국 회계 똑똑이 비서")
st.markdown("엑셀 파일을 올리면 월별 순수익을 자동으로 계산해 드려요!")

# 사이드바에 안내 문구만 남기기
with st.sidebar:
    st.markdown("### 👋 엄마를 위한 비서")
    st.write("파일만 올리면 알아서 해결해 드려요!")

# 파일 업로드 영역
uploaded_file = st.file_uploader(
    "📂 '약국 가계부.xlsx' 파일을 이곳에 드래그 앤 드롭 하세요",
    type=['xlsx'],
    label_visibility="collapsed"
)

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file, sheet_name="시트1")
        st.session_state['df'] = df
        st.session_state['file_name'] = uploaded_file.name
        st.success("✅ 파일이 성공적으로 로드되었습니다!")

        # --- 자동 요약 섹션 ---
        with st.container():
            st.subheader("📊 자동 생성된 요약 보고서")

            income_df = df[df['대분류'] == '수입']
            fixed_df = df[df['대분류'] == '고정비용']
            drug_df = df[df['대분류'] == '의약품_구입비']

            income_sum = income_df.groupby(['년', '월'])['금액'].sum()
            fixed_sum = fixed_df.groupby(['년', '월'])['금액'].sum()
            drug_sum = drug_df.groupby(['년', '월'])['금액'].sum()

            summary = pd.concat([income_sum, fixed_sum, drug_sum], axis=1)
            summary.columns = ['수입', '고정비용', '의약품_구입비']
            summary = summary.fillna(0)
            summary['순수익'] = summary['수입'] - (summary['고정비용'] + summary['의약품_구입비'])
            summary = summary.astype(int)

            st.dataframe(summary, use_container_width=True)

        # --- 채팅 섹션 ---
        st.divider()
        st.subheader("💬 궁금한 게 있으신가요?")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("질문을 입력하세요..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()

                try:
                    llm = initialize_llm(api_key)
                    tools = [analyze_financial_data]

                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", "당신은 친절한 약국 회계 전문가입니다. "
                                   "제공된 데이터를 바탕으로 엄마(약국 사장님)가 쉽게 이해하도록 구체적으로 설명해 주세요. "
                                   "숫자가 나오면 반드시 원화 단위(원)와 콤마(,)를 찍어서 보여주세요."),
                        ("human", "{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad"),
                    ])

                    agent = create_tool_calling_agent(llm, tools, prompt_template)
                    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

                    response = agent_executor.invoke({"input": prompt})
                    full_response = response['output']

                    message_placeholder.markdown(full_response)
                except Exception as e:
                    message_placeholder.markdown(f"죄송해요, 처리 중 문제가 발생했어요: {e}")

            st.session_state.messages.append({"role": "assistant", "content": full_response})

    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했어요: {e}\n파일 형식이 '시트1'에 맞는지 확인해주세요.")

else:
    st.info("👆 파일을 올려주시면 분석을 시작합니다!")
