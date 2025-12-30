import os
import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import altair as alt
import io

# ---------------------------------------------------------
# 0. API KEY 설정
# ---------------------------------------------------------
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
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
# 2. UI 디자인 (깔끔한 탭 스타일)
# ---------------------------------------------------------
def inject_custom_css():
    st.markdown("""
    <style>
    /* 1. 전체 폰트 및 기본 글자 색상 (연한 하늘색) */
    html, body, [class*="css"], .stApp {
        font-family: 'Pretendard', -apple-system, system-ui, sans-serif;
        font-size: 18px; 
        color: #bae6fd !important; /* 연한 하늘색으로 변경 */
    }
    
    .stApp { background-color: #0f172a; }

    /* 2. 사이드바 내부 글자 색상 보정 */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #bae6fd !important;
    }

    /* 3. 사이드바 배경 및 테두리 */
    [data-testid="stSidebar"] { 
        background-color: #1e293b; 
        border-right: 1px solid #334155; 
    }
    
    /* 4. 탭 디자인 */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 60px; white-space: pre-wrap; background-color: #1e293b; border-radius: 10px;
        color: #94a3b8; /* 선택되지 않은 탭은 약간 흐리게 */
        font-weight: bold; border: 1px solid #334155; padding: 0 20px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #2563eb; 
        color: #ffffff !important; /* 선택된 탭은 흰색으로 강조 */
        border-color: #60a5fa;
    }

    /* 5. KPI 카드 (Metric) */
    div[data-testid="stMetric"] {
        background-color: #1e293b; padding: 20px; border-radius: 15px;
        border: 1px solid #334155; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;
    }
    /* Metric 라벨(제목) 색상 */
    div[data-testid="stMetricLabel"] > div {
        color: #bae6fd !important;
    }
    /* Metric 수치 색상 */
    div[data-testid="stMetricValue"] { 
        color: #60a5fa !important; 
        font-size: 1.8rem !important; 
    }

    /* 6. 브리핑 박스 커스텀 클래스 */
    .briefing-box {
        background-color: #1e293b; 
        padding: 20px; 
        border-radius: 15px;
        border-left: 5px solid #10b981; 
        margin-bottom: 20px;
        color: #bae6fd; /* 박스 내부 글자색 */
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ---------------------------------------------------------
# 3. AI 에이전트 설정
# ---------------------------------------------------------
@st.cache_resource
def get_agent(df):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        api_key=api_key,
        temperature=0
    )
    return create_pandas_dataframe_agent(
        llm, df, verbose=True, allow_dangerous_code=True,
        agent_type="zero-shot-react-description", handle_parsing_errors=True
    )

# ---------------------------------------------------------
# 4. 메인 로직
# ---------------------------------------------------------

# 사이드바
with st.sidebar:
    st.title("💊 약국 비서")
    st.write("어머니, 오늘도 화이팅하세요! 💪")
    st.markdown("---")
    uploaded_file = st.file_uploader("📂 엑셀 파일을 올려주세요", type=['xlsx', 'csv'])
    st.markdown("---")
    st.info("💡 **사용법**\n1. 파일을 올린다.\n2. 연도를 고른다.\n3. 분석 결과를 본다.")

st.title("💊 우리 엄마 약국 성적표 V3")

if uploaded_file:
    try:
        # 데이터 로드
        if 'df' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            df.columns = df.columns.str.replace(' ', '')
            df['금액'] = pd.to_numeric(df['금액'], errors='coerce').fillna(0)
            st.session_state['df'] = df
            st.session_state['file_name'] = uploaded_file.name
        else:
            df = st.session_state['df']

        # 연도 선택
        all_years = sorted(df['년'].unique(), reverse=True)
        selected_year = st.sidebar.selectbox("📅 확인하고 싶은 연도", all_years)

        # 데이터 필터링
        df_year = df[df['년'] == selected_year]
        df_last = df[df['년'] == (selected_year - 1)]

        # --- 핵심 데이터 계산 (미리 집계) ---
        def summarize_monthly(d):
            if d.empty: return pd.DataFrame()
            monthly = d.groupby(['월', '대분류'])['금액'].sum().unstack(fill_value=0).reset_index()
            if '수입' not in monthly.columns: monthly['수입'] = 0
            
            # 지출 합계 계산 (수입 제외한 모든 컬럼)
            expense_cols = [c for c in monthly.columns if c not in ['월', '수입']]
            monthly['지출'] = monthly[expense_cols].sum(axis=1)
            monthly['순수익'] = monthly['수입'] - monthly['지출']
            return monthly

        summary_df = summarize_monthly(df_year)
        
        # 총계 계산
        total_income = df_year[df_year['대분류'] == '수입']['금액'].sum()
        total_expense = df_year[df_year['대분류'] != '수입']['금액'].sum()
        total_profit = total_income - total_expense
        
        # 작년 비교
        last_profit = 0
        if not df_last.empty:
            l_inc = df_last[df_last['대분류'] == '수입']['금액'].sum()
            l_exp = df_last[df_last['대분류'] != '수입']['금액'].sum()
            last_profit = l_inc - l_exp
        
        diff_profit = total_profit - last_profit

        # ---------------------------------------------------------
        # [탭 구성] 대시보드 vs AI 비서
        # ---------------------------------------------------------
        tab1, tab2 = st.tabs(["📊 우리 약국 현황", "💬 AI 비서에게 물어보기"])

        # === 탭 1: 대시보드 ===
        with tab1:
            # 1. 자동 브리핑 (비서가 말하듯이)
            st.markdown(f"""
            <div class="briefing-box">
                <h3>📢 {selected_year}년 결산 브리핑</h3>
                <p>사장님, 올해 총 순수익은 <b>{total_profit:,.0f}원</b>입니다.<br>
                작년보다 <b>{diff_profit:+,.0f}원</b> {'늘었네요! 🎉' if diff_profit > 0 else '줄었어요. 힘내세요! 💪'}<br>
                총 매출(수입)은 <b>{total_income:,.0f}원</b>, 총 지출은 <b>{total_expense:,.0f}원</b>이었습니다.</p>
            </div>
            """, unsafe_allow_html=True)

            # 2. KPI 카드
            c1, c2, c3 = st.columns(3)
            c1.metric("총 순수익", f"{total_profit:,.0f}원", f"{diff_profit:,.0f}원 (작년 대비)")
            
            best_month_row = summary_df.loc[summary_df['순수익'].idxmax()] if not summary_df.empty else None
            if best_month_row is not None:
                c2.metric("가장 좋았던 달", f"{int(best_month_row['월'])}월", f"💰 {best_month_row['순수익']:,.0f}원")
            
            avg_profit = total_profit / len(summary_df) if not summary_df.empty else 0
            c3.metric("월 평균 순수익", f"{avg_profit:,.0f}원")

            st.divider()

            # 3. 차트 (수입 vs 지출 비교)
            #col_chart, col_data = st.columns([1.5, 1])
            
            #with col_chart:
              st.subheader("📊 수입 vs 지출 비교")
              if not summary_df.empty:
                  # 데이터 변형 (Altair용)
                  chart_data = summary_df.melt(id_vars=['월'], value_vars=['수입', '지출'], var_name='구분', value_name='금액')
                  
                  base = alt.Chart(chart_data).encode(x=alt.X('월:O', title='월'))
                  bar = base.mark_bar(cornerRadius=5).encode(
                      x=alt.X('구분:N', title=None, axis=None), # 그룹화
                      y=alt.Y('금액:Q', title='금액 (원)'),
                      color=alt.Color('구분:N', scale=alt.Scale(domain=['수입', '지출'], range=['#3b82f6', '#ef4444'])),
                      column=alt.Column('월:O', header=alt.Header(titleOrient="bottom", labelOrient="bottom")), # 월별 그룹
                      tooltip=['월', '구분', alt.Tooltip('금액', format=',')]
                  ).properties(width=30, height=300) # 바 너비 조절
                   
                  st.altair_chart(bar)
              else:
                  st.info("데이터가 부족합니다.")

            #with col_data:
              st.subheader("📋 월별 상세표")
              display_cols = ['월', '수입', '지출', '순수익']
              st.dataframe(
                  summary_df[display_cols].style.format("{:,.0f}"), 
                  use_container_width=True, 
                  height=300,
                  hide_index=True
              )
                
                # 📥 다운로드 버튼 추가
              csv_buffer = io.BytesIO()
              summary_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig') # 엑셀 한글 깨짐 방지
              st.download_button(
                  label="📥 이 표를 엑셀(CSV)로 저장하기",
                  data=csv_buffer.getvalue(),
                  file_name=f"{selected_year}_약국_요약표.csv",
                  mime="text/csv"
              )

        # === 탭 2: AI 비서 ===
        with tab2:
            st.subheader("💬 AI 비서에게 물어보세요")
            
            # 추천 질문 버튼
            btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
            if btn_col1.button("💰 지출 항목 순위?"):
                st.session_state.trigger = f"{selected_year}년 지출 항목(대분류)을 금액이 큰 순서대로 5개 보여줘."
            if btn_col2.button("📈 상반기/하반기 비교"):
                st.session_state.trigger = f"{selected_year}년 상반기(1~6월)와 하반기(7~12월) 순수익을 비교해줘."
            if btn_col3.button("🍔 식비는 얼마나 썼어?"):
                st.session_state.trigger = f"{selected_year}년 내역 중에 '식대'나 '간식' 관련 비용이 얼마나 되는지 찾아줘."
            if btn_col4.button("🏥 약품비 비율은?"):
                st.session_state.trigger = f"{selected_year}년 전체 지출 중에서 '의약품' 구입비가 차지하는 비율이 몇 퍼센트야?"

            # 채팅 기록
            if "messages" not in st.session_state:
                st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 장부를 꼼꼼히 살펴보고 있습니다. 궁금한 점을 물어보세요."}]

            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])

            # 입력 처리
            user_input = st.chat_input("예: 8월에 왜 지출이 많아?")
            final_prompt = None
            
            if user_input: final_prompt = user_input
            elif "trigger" in st.session_state:
                final_prompt = st.session_state.trigger
                del st.session_state.trigger

            if final_prompt:
                st.session_state.messages.append({"role": "user", "content": final_prompt})
                st.chat_message("user").write(final_prompt)

                with st.chat_message("assistant"):
                    with st.spinner("장부 계산 중입니다... 🧮"):
                        try:
                            agent = get_agent(df)
                            system_prompt = f"""
                            당신은 '어머니'를 보좌하는 약국 전문 회계 비서입니다.
                            데이터프레임(df) 정보: {selected_year}년도 데이터입니다.
                            
                            질문: {final_prompt}
                            
                            [필수 지침]
                            1. 숫자는 반드시 3자리 콤마(예: 1,500,000원)를 사용하세요.
                            2. 비서처럼 정중하고 다정하게 대답하세요. ("~입니다", "~인 것 같아요")
                            3. 구체적인 수치를 근거로 드세요.
                            """
                            response = agent.run(system_prompt)
                            st.write(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            err = "죄송해요. 질문이 너무 어려워서 계산하다가 막혔어요. 조금 더 쉽게 질문해 주시겠어요?"
                            st.write(err)
                            st.session_state.messages.append({"role": "assistant", "content": err})

    except Exception as e:
        st.error(f"파일을 읽는데 실패했습니다: {e}")

else:
    # 초기 안내 화면
    c1, c2 = st.columns(2)
    with c1:
        st.info("👈 왼쪽에서 파일을 올려주세요.")
        st.markdown("### 🌟 V3 업데이트 기능")
        st.markdown("""
        * **🗂️ 탭 기능**: 대시보드와 채팅이 나눠져서 보기 편해요.
        * **🗣️ 3줄 브리핑**: 엑셀만 올리면 알아서 요약해 줍니다.
        * **📊 비교 차트**: 번 돈(수입)과 쓴 돈(지출)을 나란히 비교해요.
        * **📥 저장 기능**: 정리된 표를 파일로 저장할 수 있어요.
        """)
    with c2:
        st.image("https://cdn-icons-png.flaticon.com/512/3022/3022709.png", width=150)
