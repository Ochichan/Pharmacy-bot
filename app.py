import os
import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
# Pandas DataFrame Agent: AI가 데이터프레임을 직접 조작하게 하는 핵심 라이브러리
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import AgentType
import altair as alt

# ---------------------------------------------------------
# 0. API KEY 설정
# ---------------------------------------------------------
# Streamlit Cloud 배포 시 secrets 관리 필수
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    # 로컬 테스트용 (필요시 직접 입력)
    api_key = "YOUR_API_KEY_HERE"

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
# 2. UI 디자인 (어머니를 위한 가독성 중심 CSS)
# ---------------------------------------------------------
def inject_custom_css():
    st.markdown("""
    <style>
    /* 전체 폰트 설정 */
    html, body, [class*="css"] {
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif;
        font-size: 18px; 
    }
    
    .stApp { background-color: #0f172a; color: #ffffff !important; }

    /* 사이드바 스타일 */
    [data-testid="stSidebar"] { background-color: #1e293b; color: #ffffff; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #60a5fa !important;
    }

    /* 메트릭 카드 (숫자 박스) 디자인 업그레이드 */
    div[data-testid="stMetric"] {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: #60a5fa;
    }
    div[data-testid="stMetric"] label { color: #94a3b8 !important; font-size: 1.1rem !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #3b82f6 !important; font-size: 2.2rem !important; font-weight: bold; }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] { font-size: 1.0rem !important; }

    /* 데이터프레임 스타일 */
    .stDataFrame { background-color: #ffffff; border-radius: 10px; padding: 10px; }
    [data-testid="stTable"] { background-color: #ffffff !important; color: #000000 !important; }

    /* 채팅 메시지 스타일 */
    .stChatMessage { background-color: #1e293b; border-radius: 20px; padding: 15px; margin-bottom: 10px; border: 1px solid #475569; }
    
    /* 버튼 스타일 */
    .stButton > button {
        background-color: #2563eb; color: white !important; border-radius: 30px;
        padding: 12px 24px; font-weight: bold; font-size: 1.2rem;
        border: 1px solid #60a5fa;
        width: 100%;
    }
    .stButton > button:hover { background-color: #1d4ed8; transform: scale(1.02); }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ---------------------------------------------------------
# 3. LangChain 및 로직 (업그레이드된 부분)
# ---------------------------------------------------------
@st.cache_resource
def get_agent(df):
    """
    Pandas DataFrame Agent 생성
    이 에이전트는 Python 코드를 내부적으로 실행하여 DataFrame을 직접 분석합니다.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp", # 최신 모델 권장
        api_key=api_key,
        temperature=0 # 분석은 정확해야 하므로 창의성 0
    )
    
    # 🌟 핵심: create_pandas_dataframe_agent
    # 데이터프레임 전체를 AI에게 도구로 쥐어줍니다.
    agent = create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True, # 로그 출력
        allow_dangerous_code=True, # 코드를 실행하여 분석하도록 허용 (로컬/안전한 환경)
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True # 파싱 에러 자동 처리
    )
    return agent

# ---------------------------------------------------------
# 4. 메인 화면 구성
# ---------------------------------------------------------

# 사이드바
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3022/3022709.png", width=80)
    st.title("💊 약국 비서")
    st.markdown("---")
    st.markdown("### ⚙️ 사용 방법")
    st.info("1. 엑셀 파일을 업로드하세요.\n2. 보고 싶은 연도를 선택하세요.\n3. AI 비서에게 질문하세요!")
    
    uploaded_file = st.file_uploader("📂 가계부 엑셀 업로드", type=['xlsx', 'csv'])
    
    st.markdown("---")
    st.markdown("### 💡 팁")
    st.caption("작년과 비교해서 얼마나 더 벌었는지 알려드려요.")

# 메인 타이틀
st.title("💊 엄마를 위한 약국 똑똑이 비서")

if uploaded_file:
    try:
        # 파일 로드 및 캐싱
        if 'df' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # 컬럼명 공백 제거 (오류 방지)
            df.columns = df.columns.str.replace(' ', '')
            
            # 필수 전처리
            df['금액'] = pd.to_numeric(df['금액'], errors='coerce').fillna(0)
            st.session_state['df'] = df
            st.session_state['file_name'] = uploaded_file.name
        else:
            df = st.session_state['df']

        # ---------------------------------------------------------
        # [기능 1] 연도 선택 및 데이터 필터링
        # ---------------------------------------------------------
        all_years = sorted(df['년'].unique(), reverse=True)
        
        col_filter1, col_filter2 = st.columns([1, 3])
        with col_filter1:
            selected_year = st.selectbox("📅 연도 선택", all_years)
        
        # 선택된 연도 데이터
        df_year = df[df['년'] == selected_year]
        # 작년 데이터 (비교용)
        df_last_year = df[df['년'] == (selected_year - 1)]

        # ---------------------------------------------------------
        # [기능 2] 데이터 집계 및 계산
        # ---------------------------------------------------------
        def calculate_profit(dataframe):
            if dataframe.empty: return 0
            # 수입 - (고정비용 + 의약품구입비 + 기타지출)
            # 여기서는 대분류가 '수입'인 것과 아닌 것으로 나눔
            income = dataframe[dataframe['대분류'] == '수입']['금액'].sum()
            expense = dataframe[dataframe['대분류'] != '수입']['금액'].sum()
            return income - expense

        current_profit = calculate_profit(df_year)
        last_profit = calculate_profit(df_last_year)
        
        # 전년 대비 증감
        diff = current_profit - last_profit
        diff_percent = (diff / last_last_profit * 100) if last_profit != 0 else 0

        # ---------------------------------------------------------
        # [기능 3] KPI 대시보드 (전년 대비 기능 추가)
        # ---------------------------------------------------------
        st.markdown(f"### 🏆 {selected_year}년 성적표")
        kpi1, kpi2, kpi3 = st.columns(3)

        kpi1.metric(
            label="총 순수익", 
            value=f"{current_profit:,.0f}원", 
            delta=f"{diff:,.0f}원 (작년 대비)" if not df_last_year.empty else None
        )
        
        # 월 평균 계산
        months_count = df_year['월'].nunique()
        avg_monthly = current_profit / months_count if months_count > 0 else 0
        kpi2.metric(label="월 평균 순수익", value=f"{avg_monthly:,.0f}원")
        
        # 가장 수익 좋은 달
        monthly_profit = []
        for m in range(1, 13):
            m_df = df_year[df_year['월'] == m]
            if not m_df.empty:
                monthly_profit.append({'월': m, '순수익': calculate_profit(m_df)})
        
        profit_df = pd.DataFrame(monthly_profit)
        if not profit_df.empty:
            best_month = profit_df.loc[profit_df['순수익'].idxmax()]
            kpi3.metric(label="최고의 달", value=f"{int(best_month['월'])}월", delta="수고하셨어요!👏", delta_color="off")
        else:
            kpi3.metric(label="데이터 없음", value="-")

        st.divider()

        # ---------------------------------------------------------
        # [기능 4] 시각화 업그레이드 (막대 + 도넛 차트)
        # ---------------------------------------------------------
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.subheader("📈 월별 순수익 추이")
            if not profit_df.empty:
                bar_chart = alt.Chart(profit_df).mark_bar(cornerRadius=10).encode(
                    x=alt.X('월:O', title='월'),
                    y=alt.Y('순수익:Q', title='순수익 (원)'),
                    color=alt.condition(
                        alt.datum.순수익 > 0,
                        alt.value("#3b82f6"),  # 양수일 때 파란색
                        alt.value("#ef4444")   # 적자일 때 빨간색
                    ),
                    tooltip=['월', alt.Tooltip('순수익', format=',')]
                ).properties(height=350)
                st.altair_chart(bar_chart, use_container_width=True)
            else:
                st.info("차트를 그릴 데이터가 부족해요.")

        with col_chart2:
            st.subheader("🍩 지출 분석 (어디에 많이 썼을까?)")
            # 지출 데이터만 필터링
            expense_df = df_year[df_year['대분류'] != '수입']
            if not expense_df.empty:
                # 대분류별 합계
                expense_sum = expense_df.groupby('대분류')['금액'].sum().reset_index()
                
                pie_chart = alt.Chart(expense_sum).mark_arc(innerRadius=60).encode(
                    theta=alt.Theta(field="금액", type="quantitative"),
                    color=alt.Color(field="대분류", type="nominal", legend=alt.Legend(title="지출 항목")),
                    tooltip=['대분류', alt.Tooltip('금액', format=',')]
                ).properties(height=350)
                st.altair_chart(pie_chart, use_container_width=True)
            else:
                st.info("지출 내역이 없어요.")

        # ---------------------------------------------------------
        # [기능 5] AI 채팅 비서 (Pandas Agent)
        # ---------------------------------------------------------
        st.divider()
        st.subheader("💬 우리 약국 AI 비서")
        st.caption("💡 팁: '가장 지출이 큰 항목이 뭐야?', '3월 순수익 알려줘', '약값 지출 추세가 어때?' 등 자유롭게 물어보세요.")

        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 장부 엑셀을 다 읽었어요. 무엇이든 물어보세요! 😊"}]

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input("질문을 입력하세요..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("장부를 꼼꼼히 살펴보고 있어요...🕵️‍♀️"):
                    try:
                        # 전체 데이터를 agent에게 넘겨서 분석하게 함
                        agent = get_agent(df)
                        
                        # 프롬프트 엔지니어링: 엄마가 이해하기 쉽게 답변하도록 유도
                        system_prefix = f"""
                        당신은 약국을 운영하는 '어머니'를 돕는 친절하고 똑똑한 비서입니다.
                        데이터프레임(`df`)에는 '년', '월', '대분류', '소분류'(있을 경우), '내역', '금액' 정보가 있습니다.
                        
                        사용자의 질문: {prompt}
                        
                        답변 가이드:
                        1. 숫자는 반드시 3자리마다 쉼표(,)를 찍어주세요. (예: 1,000,000원)
                        2. 너무 전문적인 용어보다는 '순수익', '지출', '가장 많이 쓴 돈' 같이 쉬운 말을 사용하세요.
                        3. 결론부터 말하고, 필요하면 이유를 설명해주세요.
                        4. 한국어로 정중하고 상냥하게 답변하세요.
                        """
                        
                        response = agent.run(system_prefix)
                        st.write(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        error_msg = "죄송해요. 질문 내용을 데이터에서 찾기가 조금 어렵네요. 조금 더 구체적으로 질문해 주시겠어요?"
                        st.error(f"기술적 오류: {e}") # 디버깅용
                        st.write(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

    except Exception as e:
        st.error("엑셀 파일을 읽는 중 문제가 발생했어요.")
        st.warning(f"오류 내용: {e}")
        st.info("엑셀 파일의 첫 번째 시트에 '년', '월', '대분류', '금액' 열이 제대로 있는지 확인해주세요!")

else:
    # 초기 안내 화면
    col_intro1, col_intro2 = st.columns([1, 1])
    with col_intro1:
        st.markdown("### 👋 환영합니다!")
        st.markdown("""
        어머니, 약국 운영하시느라 고생 많으셨죠?
        이제 복잡한 계산은 저한테 맡기세요!
        
        **왼쪽 사이드바**에서 엑셀 파일만 올려주시면
        제가 알아서 싹 정리해 드릴게요.
        """)
    with col_intro2:
        st.info("👈 왼쪽의 'Browse files' 버튼을 눌러주세요!")
