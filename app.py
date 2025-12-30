import os
import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent
from langchain.agents.agent import AgentExecutor
import altair as alt  # 차트 라이브러리 추가

# ---------------------------------------------------------
# 0. API KEY 설정
# ---------------------------------------------------------
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
# 2. UI 디자인 (CSS 주입 - 가독성 극대화 버전)
# ---------------------------------------------------------
def inject_custom_css():
    st.markdown("""
    <style>
    /* 전체 폰트 크기 상향 */
    html, body, [class*="css"] {
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif;
        font-size: 18px;
    }

    .stApp { background-color: #0f172a; color: #ffffff !important; }

    /* 사이드바 */
    [data-testid="stSidebar"] { background-color: #1e293b; color: #ffffff; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #60a5fa !important; /* 사이드바 제목 하늘색 강조 */
    }

    /* 메트릭 카드 (숫자 강조 박스) 디자인 */
    div[data-testid="stMetric"] {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    div[data-testid="stMetric"] label { color: #94a3b8 !important; font-size: 1.2rem !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #3b82f6 !important; font-size: 2rem !important; font-weight: bold; }

    /* 데이터프레임 */
    .stDataFrame { background-color: #ffffff; border-radius: 10px; padding: 10px; }
    [data-testid="stTable"] { background-color: #ffffff !important; color: #000000 !important; font-size: 1.1rem; }

    /* 채팅 메시지 */
    .stChatMessage { background-color: #1e293b; border-radius: 20px; padding: 15px; margin-bottom: 10px; border: 1px solid #475569; }

    /* 버튼 */
    .stButton > button {
        background-color: #2563eb; color: white !important; border-radius: 30px;
        padding: 12px 24px; font-weight: bold; font-size: 1.2rem;
        border: 1px solid #60a5fa;
    }
    .stButton > button:hover { background-color: #1d4ed8; transform: scale(1.02); }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ---------------------------------------------------------
# 3. LangChain 및 로직
# ---------------------------------------------------------
@st.cache_resource
def initialize_llm(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        api_key=api_key,
        temperature=0.3
    )

@tool
def analyze_financial_data(question: str):
    """엑셀 데이터를 바탕으로 질문에 답변합니다."""
    try:
        df = st.session_state['df']
        # 데이터 전처리
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

        return f"데이터 요약:\n{summary.to_string()}\n\n질문: {question}"
    except Exception as e:
        return f"오류 발생: {str(e)}"

# ---------------------------------------------------------
# 4. 메인 화면 구성
# ---------------------------------------------------------

# 사이드바 설정
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3022/3022709.png", width=80) # 약국 아이콘 (외부 링크)
    st.title("💊 약국 비서")
    st.markdown("---")
    st.markdown("### ⚙️ 사용 방법")
    st.info("1. 엑셀 파일을 업로드하세요.\n2. 왼쪽에서 원하는 연도를 선택하세요.\n3. 궁금한 점은 채팅으로 물어보세요!")

    # 파일 업로더를 사이드바로 이동 (공간 확보)
    uploaded_file = st.file_uploader("📂 가계부 파일 업로드", type=['xlsx'])

# 메인 콘텐츠
st.title("💊 엄마를 위한 약국 똑똑이 비서")

if uploaded_file:
    try:
        if 'df' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
            df = pd.read_excel(uploaded_file, sheet_name="시트1")
            st.session_state['df'] = df
            st.session_state['file_name'] = uploaded_file.name
        else:
            df = st.session_state['df']

        # 데이터 전처리 (한 번만 수행)
        df['금액'] = pd.to_numeric(df['금액'], errors='coerce').fillna(0)

        # --- [기능 추가 1] 연도 선택 필터 ---
        all_years = sorted(df['년'].unique(), reverse=True)
        selected_year = st.sidebar.selectbox("📅 확인하고 싶은 연도를 선택하세요", all_years)

        # 선택된 연도 데이터만 필터링
        df_year = df[df['년'] == selected_year]

        # 요약 데이터 생성
        income_sum = df_year[df_year['대분류'] == '수입'].groupby('월')['금액'].sum()
        fixed_sum = df_year[df_year['대분류'] == '고정비용'].groupby('월')['금액'].sum()
        drug_sum = df_year[df_year['대분류'] == '의약품_구입비'].groupby('월')['금액'].sum()

        summary = pd.concat([income_sum, fixed_sum, drug_sum], axis=1)
        summary.columns = ['수입', '고정비용', '의약품_구입비']
        summary = summary.fillna(0)
        summary['순수익'] = summary['수입'] - (summary['고정비용'] + summary['의약품_구입비'])
        summary = summary.astype(int)

        # --- [기능 추가 2] 핵심 지표 카드 (KPI) ---
        st.markdown(f"### 🏆 {selected_year}년 핵심 요약")
        col1, col2, col3 = st.columns(3)
        total_profit = summary['순수익'].sum()
        avg_profit = summary['순수익'].mean()
        max_profit_month = summary['순수익'].idxmax()

        col1.metric("총 순수익", f"{total_profit:,}원")
        col2.metric("월 평균 순수익", f"{int(avg_profit):,}원")
        col3.metric("최고의 달", f"{max_profit_month}월", f"💰 {summary['순수익'].max():,}원")

        st.divider()

        # --- [기능 추가 3] 시각화 (차트) ---
        col_chart, col_table = st.columns([1.2, 1]) # 차트를 조금 더 넓게

        with col_chart:
            st.subheader("📈 월별 순수익 흐름")
            # Altair 차트 사용 (막대 그래프 + 꺾은선)
            chart_data = summary.reset_index() # '월'을 컬럼으로

            # 막대 그래프 (순수익)
            bar_chart = alt.Chart(chart_data).mark_bar(cornerRadiusTopLeft=10, cornerRadiusTopRight=10).encode(
                x=alt.X('월:O', title='월'),
                y=alt.Y('순수익:Q', title='금액 (원)'),
                color=alt.value("#3b82f6"),
                tooltip=['월', alt.Tooltip('순수익', format=',')]
            ).properties(height=400)

            # 텍스트 레이블 (금액 표시)
            text = bar_chart.mark_text(dy=-10, color='white').encode(
                text=alt.Text('순수익:Q', format=',')
            )

            st.altair_chart(bar_chart + text, use_container_width=True)

        with col_table:
            st.subheader("📋 월별 상세 표")
            st.dataframe(
                summary.style.format("{:,}"), # 천단위 콤마 자동 적용
                use_container_width=True,
                height=400
            )

        # --- 채팅 섹션 ---
        st.divider()
        st.subheader("💬 AI 비서에게 물어보세요")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("예: 8월에 지출이 왜 이렇게 많아?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                try:
                    llm = initialize_llm(api_key)
                    tools = [analyze_financial_data]

                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", "당신은 상냥한 약국 회계 전문가입니다. 데이터에 기반해 친절하게 답변해주세요. 금액은 꼭 콤마를 찍어주세요."),
                        ("human", "{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad"),
                    ])

                    agent = create_tool_calling_agent(llm, tools, prompt_template)
                    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
                    response = agent_executor.invoke({"input": prompt})
                    full_response = response['output']
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    message_placeholder.markdown(f"오류가 났어요: {e}")

            st.session_state.messages.append({"role": "assistant", "content": full_response})

    except Exception as e:
        st.error(f"파일을 읽는데 문제가 생겼어요: {e}")
else:
    # 파일 없을 때 안내 화면
    st.info("👈 왼쪽 사이드바에서 엑셀 파일을 올려주세요!")
    st.markdown("""
    ### 💡 이렇게 사용해 보세요
    1. 왼쪽 **'Browse files'** 버튼을 눌러 파일을 선택하세요.
    2. 파일이 열리면 **올해 순수익**을 바로 확인할 수 있어요.
    3. 아래 채팅창에 **"가장 돈 많이 번 달이 언제야?"** 라고 물어보세요.
    """)
