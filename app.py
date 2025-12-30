import os
import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent
from langchain.agents.agent import AgentExecutor
import altair as alt

# ---------------------------------------------------------
# 0. API KEY 및 페이지 설정
# ---------------------------------------------------------
# st.secrets를 사용하거나 환경변수에서 가져오도록 설정
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = "YOUR_API_KEY_HERE" # 로컬 테스트용 (직접 입력 필요 시)

st.set_page_config(
    page_title="약국 똑똑이 비서 v2.0",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 1. UI 디자인 (가독성 + 부드러운 색감)
# ---------------------------------------------------------
def inject_custom_css():
    st.markdown("""
    <style>
    /* 폰트 설정 */
    html, body, [class*="css"] {
        font-family: 'Pretendard', sans-serif;
        font-size: 18px; 
    }
    
    .stApp { background-color: #f8fafc; color: #1e293b !important; }

    /* 사이드바 */
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
    [data-testid="stSidebar"] h1, h2, h3 { color: #2563eb !important; }

    /* 메트릭 카드 */
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

    /* 채팅 메시지 스타일 */
    .stChatMessage { background-color: #ffffff; border-radius: 15px; padding: 15px; margin-bottom: 10px; border: 1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    [data-testid="stChatMessageAvatarUser"] { background-color: #fbbf24; }
    [data-testid="stChatMessageAvatarAssistant"] { background-color: #3b82f6; }

    /* 헤더 스타일 */
    h1, h2, h3 { color: #1e293b; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ---------------------------------------------------------
# 2. LangChain 도구 및 로직
# ---------------------------------------------------------
@st.cache_resource
def initialize_llm(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        api_key=api_key,
        temperature=0, # 분석은 정확해야 하므로 온도를 낮춤
    )

@tool
def analyze_financial_data(question: str):
    """
    엑셀 데이터를 분석하여 질문에 답합니다. 
    단순 합계뿐만 아니라, 특정 월의 지출 내역(내역/적요 컬럼이 있다면)을 확인하여 상세한 이유를 설명할 수 있습니다.
    """
    try:
        df = st.session_state['df']
        selected_year = st.session_state.get('selected_year', None)
        
        # 데이터 전처리
        df['금액'] = pd.to_numeric(df['금액'], errors='coerce').fillna(0)
        if selected_year:
            df = df[df['년'] == selected_year]

        # 1. 월별 전체 요약 데이터 생성
        income_grp = df[df['대분류'] == '수입'].groupby(['월'])['금액'].sum()
        expense_grp = df[df['대분류'].isin(['고정비용', '의약품_구입비'])].groupby(['월'])['금액'].sum()
        
        summary_text = "### 월별 요약 (단위: 원)\n"
        for month in sorted(income_grp.index):
            inc = income_grp.get(month, 0)
            exp = expense_grp.get(month, 0)
            profit = inc - exp
            summary_text += f"- {month}월: 수입 {inc:,.0f}, 지출 {exp:,.0f}, 순수익 {profit:,.0f}\n"

        # 2. 특이사항 분석 (지출이 큰 내역 Top 5 추출) - AI가 구체적인 이유를 말할 수 있게 함
        # '내역' 혹은 '적요' 컬럼이 있는지 확인
        detail_col = next((col for col in df.columns if col in ['내역', '적요', '상세', '비고']), None)
        
        top_expenses_text = ""
        if detail_col:
            # 고정비용 중 금액이 큰 순서대로 추출
            high_cost_items = df[df['대분류'] == '고정비용'].sort_values(by='금액', ascending=False).head(10)
            top_expenses_text = "\n### 올해의 주요 고정비 지출 내역 (참고용):\n"
            for _, row in high_cost_items.iterrows():
                top_expenses_text += f"- {row['월']}월 [{row[detail_col]}]: {row['금액']:,.0f}원\n"

        final_context = f"{summary_text}\n{top_expenses_text}\n\n사용자 질문: {question}"
        return final_context

    except Exception as e:
        return f"데이터 분석 중 오류가 발생했습니다: {str(e)}"

# ---------------------------------------------------------
# 3. 메인 화면 구성
# ---------------------------------------------------------

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3022/3022709.png", width=80)
    st.title("💊 약국 비서")
    st.markdown("---")
    
    # 파일 업로더
    uploaded_file = st.file_uploader("📂 엑셀 가계부 파일 업로드", type=['xlsx'])
    
    st.markdown("### 💡 팁")
    st.info("""
    **질문 예시:**
    - "이번 달 순수익 얼마야?"
    - "8월에 지출이 왜 이렇게 커?"
    - "일 년 동안 가장 돈 많이 번 달은?"
    """)

st.title("💊 엄마를 위한 약국 똑똑이 비서")

if uploaded_file:
    try:
        # 데이터 로드 및 세션 저장
        if 'df' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
            df = pd.read_excel(uploaded_file)
            st.session_state['df'] = df
            st.session_state['file_name'] = uploaded_file.name
        else:
            df = st.session_state['df']

        # 전처리
        df['금액'] = pd.to_numeric(df['금액'], errors='coerce').fillna(0)
        
        # 연도 선택
        all_years = sorted(df['년'].unique(), reverse=True)
        col_filter, _ = st.columns([1, 3])
        with col_filter:
            selected_year = st.selectbox("📅 연도 선택", all_years)
            st.session_state['selected_year'] = selected_year # 툴에서 쓰기 위해 저장

        # 필터링
        df_year = df[df['년'] == selected_year]

        # 데이터 집계
        income_sum = df_year[df_year['대분류'] == '수입'].groupby('월')['금액'].sum()
        fixed_sum = df_year[df_year['대분류'] == '고정비용'].groupby('월')['금액'].sum()
        drug_sum = df_year[df_year['대분류'] == '의약품_구입비'].groupby('월')['금액'].sum()
        
        summary = pd.concat([income_sum, fixed_sum, drug_sum], axis=1)
        summary.columns = ['수입', '고정비용', '의약품_구입비']
        summary = summary.fillna(0)
        summary['총지출'] = summary['고정비용'] + summary['의약품_구입비']
        summary['순수익'] = summary['수입'] - summary['총지출']

        # --- [KPI 카드 섹션] ---
        st.markdown(f"### 🏆 {selected_year}년 성적표")
        kpi1, kpi2, kpi3 = st.columns(3)
        
        total_profit = summary['순수익'].sum()
        avg_profit = summary['순수익'].mean()
        max_profit_month = summary['순수익'].idxmax()
        max_profit_val = summary['순수익'].max()

        kpi1.metric("총 순수익", f"{total_profit:,.0f}원")
        kpi2.metric("월 평균 순수익", f"{avg_profit:,.0f}원")
        kpi3.metric("최고의 달 (효자달)", f"{max_profit_month}월", f"💰 +{max_profit_val:,.0f}원")

        st.markdown("---")

        # --- [차트 섹션] ---
        # 탭을 사용하여 차트를 깔끔하게 분리
        tab1, tab2 = st.tabs(["📊 수입 vs 지출 흐름", "🍰 고정비용 분석"])

        with tab1:
            st.subheader("들어온 돈(수입) vs 나간 돈(지출)")
            chart_data = summary.reset_index()
            
            # 수입 (막대)
            bar = alt.Chart(chart_data).mark_bar(color='#a7f3d0', cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
                x=alt.X('월:O', title='월'),
                y=alt.Y('수입:Q', title='금액'),
                tooltip=['월', alt.Tooltip('수입', format=',')]
            )
            
            # 지출 (선) - 빨간색으로 경각심
            line = alt.Chart(chart_data).mark_line(color='#ef4444', point=True).encode(
                x='월:O',
                y='총지출:Q',
                tooltip=['월', alt.Tooltip('총지출', format=',')]
            )
            
            # 순수익 (텍스트)
            text = bar.mark_text(dy=-10).encode(
                text=alt.Text('순수익:Q', format=',')
            )

            st.altair_chart((bar + line).interactive(), use_container_width=True)

        with tab2:
            st.subheader("어디에 돈을 많이 썼을까? (고정비용)")
            # 고정비용 상세 항목별 집계 (중분류 혹은 내역 기준)
            # 데이터에 '중분류'가 있다고 가정, 없으면 '내역' 사용
            cat_col = '중분류' if '중분류' in df_year.columns else ('내역' if '내역' in df_year.columns else None)
            
            if cat_col:
                fixed_cost_df = df_year[df_year['대분류'] == '고정비용']
                pie_data = fixed_cost_df.groupby(cat_col)['금액'].sum().reset_index()
                
                pie_chart = alt.Chart(pie_data).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta(field="금액", type="quantitative"),
                    color=alt.Color(field=cat_col, type="nominal", legend=alt.Legend(title="항목")),
                    tooltip=[cat_col, alt.Tooltip('금액', format=',')]
                )
                st.altair_chart(pie_chart, use_container_width=True)
            else:
                st.info("데이터에 '중분류' 또는 '내역' 컬럼이 없어 상세 분석이 어렵습니다.")

        # --- [채팅 섹션] ---
        st.markdown("---")
        st.subheader("💬 엄마를 위한 AI 비서")
        st.caption("궁금한 걸 편하게 물어보세요. (예: 5월달 상세 내역 알려줘)")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 채팅 기록 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 사용자 입력 처리
        if prompt := st.chat_input("질문을 입력하세요..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("비서가 장부를 살펴보고 있어요... 🧐")
                
                try:
                    llm = initialize_llm(api_key)
                    tools = [analyze_financial_data]
                    
                    # 프롬프트 강화: 더 친절하고 상세하게
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", """
                         당신은 사용자의 어머니를 보조하는 친절하고 똑똑한 약국 회계 비서입니다.
                         
                         [지침]
                         1. 답변은 공손하고 다정하게(해요체) 해주세요.
                         2. 금액을 말할 때는 반드시 3자리마다 콤마를 찍어주세요 (예: 1,000,000원).
                         3. 데이터 도구를 통해 얻은 '수입', '지출', '특이사항(고액 지출)'을 바탕으로 분석적인 답변을 주세요.
                         4. 질문에 대한 답을 데이터에서 찾을 수 없다면 솔직하게 모른다고 해주세요.
                         """),
                        ("human", "{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad"),
                    ])
                    
                    agent = create_tool_calling_agent(llm, tools, prompt_template)
                    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # 디버깅용 verbose
                    
                    response = agent_executor.invoke({"input": prompt})
                    full_response = response['output']
                    
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    error_msg = f"죄송해요, 답변을 만드는 중에 문제가 생겼어요.\n\n오류 내용: {e}"
                    message_placeholder.error(error_msg)

    except Exception as e:
        st.error(f"파일을 처리하는 중 오류가 발생했습니다: {e}")
        st.write("엑셀 파일의 컬럼명(년, 월, 대분류, 금액 등)을 확인해주세요.")

else:
    # 초기 안내 화면
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3022/3022709.png", width=150)
    with col2:
        st.markdown("""
        ## 환영합니다! 👋
        
        어머니, 약국 운영하시느라 고생 많으셨죠?
        이 비서가 복잡한 장부 정리를 도와드릴게요.
        
        **👈 왼쪽에서 엑셀 파일을 선택해주세요.**
        """)
