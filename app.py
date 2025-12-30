import os
import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent
from langchain.agents.agent import AgentExecutor
import altair as alt

# --- API KEY 및 페이지 설정 ---
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = "YOUR_API_KEY_HERE"

st.set_page_config(
    page_title="약국 똑똑이 비서 v2.0",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- UI 디자인 (CSS) ---
def inject_custom_css():
    st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Pretendard', sans-serif;
        font-size: 18px;
    }
    .stApp { background-color: #f8fafc; color: #1e293b !important; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
    [data-testid="stSidebar"] h1, h2, h3 { color: #2563eb !important; }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    div[data-testid="stMetric"] label { color: #64748b !important; font-size: 1.1rem !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #2563eb !important; font-size: 2.2rem !important; font-weight: 800; }
    .stChatMessage { background-color: #ffffff; border-radius: 15px; padding: 15px; margin-bottom: 10px; border: 1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    [data-testid="stChatMessageAvatarUser"] { background-color: #fbbf24; }
    [data-testid="stChatMessageAvatarAssistant"] { background-color: #3b82f6; }
    h1, h2, h3 { color: #1e293b; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- LangChain 도구 및 로직 ---
@st.cache_resource
def initialize_llm(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        api_key=api_key,
        temperature=0,
    )

@tool
def analyze_financial_data(question: str):
    """엑셀 데이터를 분석하여 질문에 답합니다."""
    try:
        df = st.session_state['df']
        selected_year = st.session_state.get('selected_year', None)
        
        df['금액'] = pd.to_numeric(df['금액'], errors='coerce').fillna(0)
        if selected_year:
            df = df[df['년'] == selected_year]

        income_grp = df[df['대분류'] == '수입'].groupby(['월'])['금액'].sum()
        expense_grp = df[df['대분류'].isin(['고정비용', '의약품_구입비'])].groupby(['월'])['금액'].sum()
        
        summary_text = "### 월별 요약 (단위: 원)\n"
        for month in sorted(income_grp.index):
            inc = income_grp.get(month, 0)
            exp = expense_grp.get(month, 0)
            profit = inc - exp
            summary_text += f"- {month}월: 수입 {inc:,.0f}, 지출 {exp:,.0f}, 순수익 {profit:,.0f}\n"

        detail_col = next((col for col in df.columns if col in ['내역', '적요', '상세', '비고']), None)
        top_expenses_text = ""
        if detail_col:
            high_cost_items = df[df['대분류'] == '고정비용'].sort_values(by='금액', ascending=False).head(10)
            top_expenses_text = "\n### 올해의 주요 고정비 지출 내역 (참고용):\n"
            for _, row in high_cost_items.iterrows():
                top_expenses_text += f"- {row['월']}월 [{row[detail_col]}]: {row['금액']:,.0f}원\n"

        return f"{summary_text}\n{top_expenses_text}\n\n사용자 질문: {question}"
    except Exception as e:
        return f"오류 발생: {str(e)}"

# --- 메인 화면 구성 ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3022/3022709.png", width=80)
    st.title("💊 약국 비서")
    st.markdown("---")
    uploaded_file = st.file_uploader("📂 엑셀 가계부 파일 업로드", type=['xlsx'])
    st.markdown("### 💡 팁")
    st.info("""
    **질문 예시:**
    - "이번 달 순수익 얼마야?"
    - "8월에 지출이 왜 이렇게 커?"
    """)

st.title("💊 엄마를 위한 약국 똑똑이 비서")

if uploaded_file:
    try:
        if 'df' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
            df = pd.read_excel(uploaded_file)
            st.session_state['df'] = df
            st.session_state['file_name'] = uploaded_file.name
        else:
            df = st.session_state['df']

        df['금액'] = pd.to_numeric(df['금액'], errors='coerce').fillna(0)
        
        all_years = sorted(df['년'].unique(), reverse=True)
        col_filter, _ = st.columns([1, 3])
        with col_filter:
            selected_year = st.selectbox("📅 연도 선택", all_years)
            st.session_state['selected_year'] = selected_year

        df_year = df[df['년'] == selected_year]
        income_sum = df_year[df_year['대분류'] == '수입'].groupby('월')['금액'].sum()
        fixed_sum = df_year[df_year['대분류'] == '고정비용'].groupby('월')['금액'].sum()
        drug_sum = df_year[df_year['대분류'] == '의약품_구입비'].groupby('월')['금액'].sum()
        
        summary = pd.concat([income_sum, fixed_sum, drug_sum], axis=1)
        summary.columns = ['수입', '고정비용', '의약품_구입비']
        summary = summary.fillna(0)
        summary['총지출'] = summary['고정비용'] + summary['의약품_구입비']
        summary['순수익'] = summary['수입'] - summary['총지출']

        st.markdown(f"### 🏆 {selected_year}년 성적표")
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("총 순수익", f"{summary['순수익'].sum():,.0f}원")
        kpi2.metric("월 평균 순수익", f"{summary['순수익'].mean():,.0f}원")
        kpi3.metric("최고의 달", f"{summary['순수익'].idxmax()}월", f"💰 +{summary['순수익'].max():,.0f}원")
        st.markdown("---")

        tab1, tab2 = st.tabs(["📊 수입 vs 지출 흐름", "🍰 고정비용 분석"])
        with tab1:
            st.subheader("들어온 돈(수입) vs 나간 돈(지출)")
            chart_data = summary.reset_index()
            bar = alt.Chart(chart_data).mark_bar(color='#a7f3d0').encode(
                x=alt.X('월:O'), y=alt.Y('수입:Q'), tooltip=['월', '수입']
            )
            line = alt.Chart(chart_data).mark_line(color='#ef4444', point=True).encode(
                x='월:O', y='총지출:Q', tooltip=['월', '총지출']
            )
            st.altair_chart((bar + line).interactive(), use_container_width=True)

        with tab2:
            st.subheader("고정비용 분석")
            cat_col = '중분류' if '중분류' in df_year.columns else ('내역' if '내역' in df_year.columns else None)
            if cat_col:
                pie_data = df_year[df_year['대분류'] == '고정비용'].groupby(cat_col)['금액'].sum().reset_index()
                pie = alt.Chart(pie_data).mark_arc(innerRadius=50).encode(
                    theta='금액', color=cat_col, tooltip=[cat_col, '금액']
                )
                st.altair_chart(pie, use_container_width=True)
            else:
                st.info("상세 내역(중분류/내역)이 없어 분석할 수 없습니다.")

        st.markdown("---")
        st.subheader("💬 엄마를 위한 AI 비서")
        
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
                msg_box = st.empty()
                msg_box.markdown("비서가 장부를 보는 중... 🧐")
                try:
                    llm = initialize_llm(api_key)
                    tools = [analyze_financial_data]
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", "당신은 친절한 약국 회계 비서입니다. 금액에 콤마를 찍어 답변하세요."),
                        ("human", "{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad"),
                    ])
                    agent = create_tool_calling_agent(llm, tools, prompt_template)
                    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
                    response = agent_executor.invoke({"input": prompt})
                    msg_box.markdown(response['output'])
                    st.session_state.messages.append({"role": "assistant", "content": response['output']})
                except Exception as e:
                    msg_box.error(f"오류: {e}")

    except Exception as e:
        st.error(f"파일 오류: {e}")

else:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3022/3022709.png", width=150)
    with col2:
        st.markdown("""
        ## 환영합니다! 👋
        어머니, 약국 운영하시느라 고생 많으셨죠?
        **👈 왼쪽에서 엑셀 파일을 선택해주세요.**
        """)
