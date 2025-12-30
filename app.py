import os
import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import altair as alt

# ---------------------------------------------------------
# 0. API KEY 설정
# ---------------------------------------------------------
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    # 로컬 테스트용
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
# 2. UI 디자인 (가독성 & 편의성)
# ---------------------------------------------------------
def inject_custom_css():
    st.markdown("""
    <style>
    /* 폰트 및 기본 설정 */
    html, body, [class*="css"] {
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif;
        font-size: 18px; 
    }
    .stApp { background-color: #0f172a; color: #ffffff !important; }

    /* 사이드바 */
    [data-testid="stSidebar"] { background-color: #1e293b; color: #ffffff; }
    
    /* 숫자 카드 (Metric) */
    div[data-testid="stMetric"] {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 15px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    div[data-testid="stMetricValue"] { color: #60a5fa !important; font-size: 2rem !important; }

    /* 버튼 스타일 (질문 버튼용) */
    .stButton button {
        background-color: #334155; 
        color: white; 
        border-radius: 10px;
        border: 1px solid #475569;
        font-size: 1rem;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #2563eb;
        border-color: #60a5fa;
    }

    /* 채팅 메시지 */
    .stChatMessage { background-color: #1e293b; border-radius: 15px; border: 1px solid #475569; }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ---------------------------------------------------------
# 3. AI 로직 (Pandas Agent)
# ---------------------------------------------------------
@st.cache_resource
def get_agent(df):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        api_key=api_key,
        temperature=0
    )
    
    agent = create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True,
        allow_dangerous_code=True,
        # 🌟 에러 방지를 위해 문자열로 지정
        agent_type="zero-shot-react-description",
        handle_parsing_errors=True
    )
    return agent

# ---------------------------------------------------------
# 4. 메인 화면 로직
# ---------------------------------------------------------

# 사이드바
with st.sidebar:
    st.title("💊 약국 비서")
    st.caption("어머니를 위한 똑똑한 장부 관리")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📂 엑셀 파일을 올려주세요", type=['xlsx', 'csv'])
    
    st.markdown("---")
    st.info("💡 **사용 팁**\n\n파일을 올리고 연도를 선택하면\n자동으로 분석해 드려요.")

# 메인 타이틀
st.title("💊 우리 엄마 약국 성적표")

if uploaded_file:
    try:
        # 데이터 로드
        if 'df' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            df.columns = df.columns.str.replace(' ', '') # 공백 제거
            df['금액'] = pd.to_numeric(df['금액'], errors='coerce').fillna(0)
            
            st.session_state['df'] = df
            st.session_state['file_name'] = uploaded_file.name
        else:
            df = st.session_state['df']

        # 1. 연도 선택
        all_years = sorted(df['년'].unique(), reverse=True)
        col_year, _ = st.columns([1, 3])
        with col_year:
            selected_year = st.selectbox("📅 몇 년도 장부를 볼까요?", all_years)
        
        df_year = df[df['년'] == selected_year]
        df_last = df[df['년'] == (selected_year - 1)]

        # 2. 계산 로직
        def calc_profit(d):
            if d.empty: return 0
            inc = d[d['대분류'] == '수입']['금액'].sum()
            exp = d[d['대분류'] != '수입']['금액'].sum()
            return inc - exp

        cur_profit = calc_profit(df_year)
        last_profit = calc_profit(df_last)
        diff = cur_profit - last_profit

        # 3. 상단 KPI 카드
        st.markdown("### 🏆 한눈에 보는 요약")
        k1, k2, k3 = st.columns(3)
        k1.metric("올해 총 순수익", f"{cur_profit:,.0f}원", f"{diff:,.0f}원 (작년 대비)" if not df_last.empty else None)
        
        # 월평균
        avg = cur_profit / df_year['월'].nunique() if not df_year.empty else 0
        k2.metric("월 평균 순수익", f"{avg:,.0f}원")

        # 최고의 달 찾기
        monthly_data = []
        for m in sorted(df_year['월'].unique()):
            m_d = df_year[df_year['월'] == m]
            monthly_data.append({'월': m, '순수익': calc_profit(m_d)})
        
        profit_df = pd.DataFrame(monthly_data)
        if not profit_df.empty:
            best = profit_df.loc[profit_df['순수익'].idxmax()]
            k3.metric("가장 장사 잘 된 달", f"{int(best['월'])}월", f"💰 {best['순수익']:,.0f}원", delta_color="off")

        st.divider()

        # 4. 차트 섹션 (업그레이드: 텍스트 라벨 추가)
        c1, c2 = st.columns([1.5, 1])
        
        with c1:
            st.subheader("📈 월별 순수익 그래프")
            if not profit_df.empty:
                # 기본 바 차트
                base = alt.Chart(profit_df).encode(x=alt.X('월:O', title='월'))
                bars = base.mark_bar(cornerRadius=8).encode(
                    y=alt.Y('순수익:Q', title='금액'),
                    color=alt.condition(alt.datum.순수익 > 0, alt.value("#3b82f6"), alt.value("#ef4444"))
                )
                # 🌟 업그레이드: 막대 위에 숫자 표시
                text = base.mark_text(dy=-10, color='white').encode(
                    y=alt.Y('순수익:Q'),
                    text=alt.Text('순수익:Q', format=',')
                )
                st.altair_chart((bars + text).properties(height=350), use_container_width=True)

        with c2:
            st.subheader("🍩 지출 비중")
            exp_df = df_year[df_year['대분류'] != '수입']
            if not exp_df.empty:
                exp_sum = exp_df.groupby('대분류')['금액'].sum().reset_index()
                pie = alt.Chart(exp_sum).mark_arc(innerRadius=60).encode(
                    theta=alt.Theta(field="금액", type="quantitative"),
                    color=alt.Color(field="대분류", legend=alt.Legend(title="항목", orient="bottom")),
                    tooltip=['대분류', alt.Tooltip('금액', format=',')]
                ).properties(height=350)
                st.altair_chart(pie, use_container_width=True)

        # 5. AI 채팅 섹션 (업그레이드: 추천 질문 버튼)
        st.divider()
        st.subheader("💬 AI 비서에게 물어보세요")

        # 채팅 기록 초기화
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "장부 분석이 끝났어요! 궁금한 점을 물어보시거나 아래 버튼을 눌러보세요."}]

        # 🌟 추천 질문 버튼 (누르면 바로 질문됨)
        st.write("👇 **궁금한 내용을 클릭해보세요!**")
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("💰 이번 달 순수익은?"):
            prompt = f"{selected_year}년의 월별 순수익을 표로 보여줘."
            st.session_state.trigger_prompt = prompt
        if b2.button("💸 지출이 제일 큰 항목?"):
            prompt = f"{selected_year}년에 돈을 가장 많이 쓴 지출 항목(대분류) TOP 3를 알려줘."
            st.session_state.trigger_prompt = prompt
        if b3.button("📉 작년보다 나아졌어?"):
            prompt = f"{selected_year}년과 {selected_year-1}년의 총 순수익을 비교해서 설명해줘."
            st.session_state.trigger_prompt = prompt
        if b4.button("💊 약값은 얼마나 썼어?"):
            prompt = f"{selected_year}년에 '의약품' 관련 비용으로 총 얼마를 썼는지 알려줘."
            st.session_state.trigger_prompt = prompt

        # 이전 대화 출력
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        # 입력 처리 (버튼 클릭 or 직접 입력)
        prompt_input = st.chat_input("직접 질문을 입력할 수도 있어요...")
        
        # 버튼이 눌렸거나, 직접 입력이 있거나
        final_prompt = None
        if prompt_input:
            final_prompt = prompt_input
        elif "trigger_prompt" in st.session_state:
            final_prompt = st.session_state.trigger_prompt
            del st.session_state.trigger_prompt # 사용 후 삭제

        if final_prompt:
            st.session_state.messages.append({"role": "user", "content": final_prompt})
            st.chat_message("user").write(final_prompt)

            with st.chat_message("assistant"):
                with st.spinner("장부를 계산기 두드리며 확인 중... 🧮"):
                    try:
                        agent = get_agent(df)
                        # 🌟 프롬프트 강화: 데이터 구조를 더 명확히 설명
                        system_prompt = f"""
                        당신은 약국을 운영하는 어머니의 다정한 회계 비서입니다.
                        
                        [데이터 설명]
                        - `df` 데이터프레임에는 약국 장부 데이터가 들어있습니다.
                        - 컬럼: '년', '월', '대분류', '내역', '금액'
                        - '대분류'가 '수입'이면 매출이고, 나머지는 모두 지출입니다.
                        - 순수익 = 수입 합계 - 지출 합계
                        
                        [사용자 질문]
                        {final_prompt}
                        
                        [답변 원칙]
                        1. 금액은 무조건 '1,234,567원' 처럼 쉼표를 찍으세요.
                        2. 표(Table)가 필요하면 Markdown으로 깔끔하게 그려주세요.
                        3. 말투는 공손하고 다정하게 하세요 (예: "~입니다", "~인 것 같아요").
                        4. 파이썬 코드를 실행해서 정확한 값을 계산해서 답하세요.
                        """
                        response = agent.run(system_prompt)
                        st.write(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        err_msg = "죄송해요. 질문이 조금 복잡해서 계산하다가 실수했네요. 다시 한 번 쉽게 질문해 주시겠어요?"
                        st.write(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": err_msg})

    except Exception as e:
        st.error("파일을 읽는데 실패했어요. 파일 형식을 확인해주세요.")
        st.write(e)

else:
    # 초기 화면
    st.info("👈 왼쪽에서 엑셀 파일을 업로드해주세요!")
    st.markdown("""
    ### 👩‍apothecary 약국 비서 사용법
    1. 왼쪽 **'Browse files'** 버튼 클릭
    2. 약국 가계부 엑셀 파일 선택
    3. 자동으로 만들어지는 **성적표** 확인
    4. 궁금한 건 **채팅**으로 물어보기
    """)
