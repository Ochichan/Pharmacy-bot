import os
import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent
from langchain.agents.agent import AgentExecutor
import altair as alt
import io

# ---------------------------------------------------------
# 0. API KEY 및 페이지 설정
# ---------------------------------------------------------
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = "YOUR_API_KEY_HERE"

st.set_page_config(
    page_title="약국 똑똑이 비서 v3.0",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 1. UI 디자인 (깔끔하고 글씨 크게)
# ---------------------------------------------------------
def inject_custom_css():
    st.markdown("""
    <style>
    /* 전체 폰트 및 배경 */
    html, body, [class*="css"] {
        font-family: 'Pretendard', 'Malgun Gothic', sans-serif;
        font-size: 18px;
    }
    .stApp { background-color: #f8fafc; color: #1e293b; }

    /* 사이드바 스타일 */
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
    [data-testid="stSidebar"] h1 { color: #2563eb; }

    /* KPI 카드 스타일 */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    div[data-testid="stMetric"] label { font-size: 1.1rem; color: #64748b; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 800; color: #2563eb; }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] { font-size: 1.0rem; }

    /* 버튼 스타일 */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #dbEafe;
        border-color: #3b82f6;
    }

    /* 채팅 메시지 */
    .stChatMessage { background-color: #ffffff; border-radius: 15px; border: 1px solid #e2e8f0; }
    [data-testid="stChatMessageAvatarUser"] { background-color: #fbbf24; }
    [data-testid="stChatMessageAvatarAssistant"] { background-color: #3b82f6; }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ---------------------------------------------------------
# 2. AI 로직 (LangChain)
# ---------------------------------------------------------
@st.cache_resource
def initialize_llm(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        api_key=api_key,
        temperature=0,
    )

@tool
def analyze_financial_data(question: str):
    """엑셀 데이터를 분석하여 질문에 답합니다. 구체적인 수치와 내역을 포함하세요."""
    try:
        df = st.session_state['df']
        selected_year = st.session_state.get('selected_year', None)
        
        # 전처리
        df['금액'] = pd.to_numeric(df['금액'], errors='coerce').fillna(0)
        
        # 해당 연도 데이터
        df_curr = df[df['년'] == selected_year]
        
        # 요약 생성
        income = df_curr[df_curr['대분류'] == '수입']['금액'].sum()
        expense = df_curr[df_curr['대분류'].isin(['고정비용', '의약품_구입비'])]['금액'].sum()
        profit = income - expense
        
        # 고액 지출 내역 (Top 5)
        detail_col = next((col for col in df.columns if col in ['내역', '적요', '상세', '비고']), None)
        top_expenses = ""
        if detail_col:
            top_items = df_curr[df_curr['대분류'] == '고정비용'].sort_values('금액', ascending=False).head(5)
            for _, row in top_items.iterrows():
                top_expenses += f"- {row['월']}월 {row[detail_col]}: {row['금액']:,.0f}원\n"
        
        context = f"""
        [분석 데이터 - {selected_year}년]
        - 총 수입: {income:,.0f}원
        - 총 지출: {expense:,.0f}원
        - 순수익: {profit:,.0f}원
        
        [주요 고정비 지출 Top 5]
        {top_expenses if top_expenses else "상세 내역 없음"}
        
        사용자 질문: {question}
        """
        return context
    except Exception as e:
        return f"데이터 분석 오류: {str(e)}"
# ---------------------------------------------------------
# 3. 메인 화면 (수정본)
# ---------------------------------------------------------

# 사이드바
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3022/3022709.png", width=80)
    st.title("💊 약국 비서")
    st.markdown("---")
    uploaded_file = st.file_uploader("📂 장부 파일(Excel) 업로드", type=['xlsx'])
    
    if uploaded_file:
        st.success("파일이 연결되었습니다!")
    else:
        st.info("왼쪽 상단의 'Browse files'를 눌러 엑셀 파일을 올려주세요.")

# 메인 타이틀
st.title("💊 엄마를 위한 약국 똑똑이 비서")

# 로직 시작
if uploaded_file:
    try:
        if 'df' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
            df = pd.read_excel(uploaded_file)
            required_cols = ['년', '월', '대분류', '금액']
            if not all(col in df.columns for col in required_cols):
                st.error(f"엑셀 파일에 다음 컬럼이 꼭 있어야 해요: {required_cols}")
                st.stop()
            st.session_state['df'] = df
            st.session_state['file_name'] = uploaded_file.name
        else:
            df = st.session_state['df']

        df['금액'] = pd.to_numeric(df['금액'], errors='coerce').fillna(0)
        
        all_years = sorted(df['년'].unique(), reverse=True)
        
        # 데이터가 없을 경우 방어 코드
        if not all_years:
            st.warning("데이터에 '년' 정보가 없어요.")
            st.stop()

        c1, c2 = st.columns([1, 4])
        with c1:
            selected_year = st.selectbox("📅 연도 선택", all_years)
            st.session_state['selected_year'] = selected_year
        
        df_curr = df[df['년'] == selected_year]
        df_prev = df[df['년'] == (selected_year - 1)]

        def create_summary(dframe):
            if dframe.empty:
                return pd.DataFrame(columns=['수입', '고정비용', '의약품_구입비', '순수익'])
            
            inc = dframe[dframe['대분류'] == '수입'].groupby('월')['금액'].sum()
            fix = dframe[dframe['대분류'] == '고정비용'].groupby('월')['금액'].sum()
            drug = dframe[dframe['대분류'] == '의약품_구입비'].groupby('월')['금액'].sum()
            
            summ = pd.concat([inc, fix, drug], axis=1).fillna(0)
            summ.columns = ['수입', '고정비용', '의약품_구입비']
            summ['순수익'] = summ['수입'] - (summ['고정비용'] + summ['의약품_구입비'])
            return summ

        summary_curr = create_summary(df_curr)
        
        # KPI 섹션
        st.markdown(f"### 🏆 {selected_year}년 운영 성적표")
        kpi1, kpi2, kpi3 = st.columns(3)

        curr_profit = summary_curr['순수익'].sum() if not summary_curr.empty else 0
        curr_avg = summary_curr['순수익'].mean() if not summary_curr.empty else 0
        curr_max_month = summary_curr['순수익'].idxmax() if not summary_curr.empty else "-"
        curr_max_val = summary_curr['순수익'].max() if not summary_curr.empty else 0

        # [수정 포인트 1] None 대신 빈 문자열 사용
        delta_profit = "" 
        if not df_prev.empty:
            summary_prev = create_summary(df_prev)
            if not summary_prev.empty:
                prev_profit = summary_prev['순수익'].sum()
                diff = curr_profit - prev_profit
                delta_profit = f"{diff:,.0f}원 (작년 대비)"

        kpi1.metric("총 순수익", f"{curr_profit:,.0f}원", delta=delta_profit or None)
        kpi2.metric("월 평균 순수익", f"{curr_avg:,.0f}원")
        kpi3.metric("최고의 달 (효자달)", f"{curr_max_month}월", f"💰 {curr_max_val:,.0f}원")

        st.markdown("---")

        # 탭 섹션
        t1, t2 = st.tabs(["📊 월별 흐름 한눈에 보기", "🍰 지출 분석"])
        
        with t1:
            if not summary_curr.empty:
                chart_data = summary_curr.reset_index()
                base = alt.Chart(chart_data).encode(x=alt.X('월:O', title='월'))
                bar = base.mark_bar(color='#a7f3d0', cornerRadius=5).encode(
                    y=alt.Y('수입:Q', title='금액'), tooltip=['월', alt.Tooltip('수입', format=',')]
                )
                line = base.mark_line(color='#ef4444', point=True).encode(
                    y=alt.Y('의약품_구입비', title='지출(약값+고정비)'),
                    tooltip=['월', alt.Tooltip('의약품_구입비', format=',')]
                )
                st.altair_chart((bar + line).interactive(), use_container_width=True)
                
                st.caption("이 표를 엑셀로 저장하고 싶으시면 아래 버튼을 누르세요.")
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    summary_curr.to_excel(writer, sheet_name='월별요약')
                
                st.download_button(
                    label="📥 월별 요약표 다운로드",
                    data=buffer,
                    file_name=f"{selected_year}_약국요약.xlsx",
                    mime="application/vnd.ms_excel"
                )
            else:
                st.info("표시할 데이터가 없습니다.")

        with t2:
            st.subheader("고정비용 상세 분석")
            cat_col = '중분류' if '중분류' in df.columns else ('내역' if '내역' in df.columns else None)
            
            # [수정 포인트 2] 데이터 존재 여부 확인 후 차트 그리기
            if cat_col and not df_curr.empty:
                pie_data = df_curr[df_curr['대분류'] == '고정비용'].groupby(cat_col)['금액'].sum().reset_index()
                if not pie_data.empty:
                    pie = alt.Chart(pie_data).mark_arc(innerRadius=60).encode(
                        theta=alt.Theta("금액", stack=True),
                        color=alt.Color(cat_col, legend=alt.Legend(title="항목")),
                        tooltip=[cat_col, alt.Tooltip('금액', format=',')],
                        order=alt.Order("금액", sort="descending")
                    )
                    st.altair_chart(pie, use_container_width=True)
                else:
                     st.info("고정비용 데이터가 없습니다.")
            else:
                st.info("상세 내역(중분류) 정보가 없거나 데이터가 부족해요.")

        st.markdown("---")

        # 채팅 섹션
        st.subheader("💬 AI 비서에게 물어보세요")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        st.write("자주 묻는 질문 (버튼을 누르면 바로 답해드려요!)")
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        user_input = None
        
        if btn_col1.button("💰 이번 달 순수익은?"):
            user_input = f"{selected_year}년의 월별 순수익을 알려줘."
        if btn_col2.button("📉 지출이 제일 큰 달은?"):
            user_input = f"{selected_year}년 중 지출이 가장 컸던 달과 이유를 분석해줘."
        if btn_col3.button("📊 일년 총 결산 해줘"):
            user_input = f"{selected_year}년 전체 수입과 지출을 요약해주고, 잘한 점을 칭찬해줘."

        chat_input = st.chat_input("궁금한 내용을 입력하세요...")
        if chat_input:
            user_input = chat_input

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                container = st.empty()
                container.markdown("장부를 분석하고 있습니다... ⏳")
                try:
                    llm = initialize_llm(api_key)
                    tools = [analyze_financial_data]
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "당신은 약국 운영을 돕는 따뜻하고 유능한 비서입니다. 어르신이 보기 편하게 금액에 콤마를 찍고, 중요한 내용은 **굵게** 표시하세요."),
                        ("human", "{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad"),
                    ])
                    agent = create_tool_calling_agent(llm, tools, prompt)
                    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
                    
                    response = agent_executor.invoke({"input": user_input})
                    final_ans = response['output']
                    
                    container.markdown(final_ans)
                    st.session_state.messages.append({"role": "assistant", "content": final_ans})
                except Exception as e:
                    container.error(f"오류가 발생했어요: {e}")

    except Exception as e:
        st.error(f"파일을 읽는 중 문제가 생겼어요: {e}")

else:
    # 파일 업로드 전 안내 화면
    c1, c2 = st.columns([1, 2])
    with c1:
        st.image("https://cdn-icons-png.flaticon.com/512/3022/3022709.png", width=150)
    with c2:
        st.markdown("""
        ## 환영합니다! 👋
        어머니, 약국 운영하시느라 정말 고생 많으셨습니다.
        
        **1. 왼쪽의 'Browse files' 버튼을 눌러주세요.**
        **2. 엑셀 장부 파일을 선택하면 제가 분석해 드릴게요.**
        """)
