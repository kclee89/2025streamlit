import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# Streamlit 앱 제목
st.title("CSV 데이터 분석 및 시각화")

# CSV 파일 경로
file_path = "4DMR.csv"

try:
    # CSV 파일 읽기
    df = pd.read_csv(file_path)
    
    # 데이터프레임 표시
    st.header("데이터 미리보기")
    st.write(df)
    
    # 데이터 정보
    st.header("데이터 정보")
    st.write(f"행 개수: {df.shape[0]}, 열 개수: {df.shape[1]}")
    st.write("컬럼 정보:")
    st.write(df.dtypes)

    # 데이터 통계 요약
    st.header("기초 통계 요약")
    st.write(df.describe())

    # instability 그룹 비교
    st.header("Instability 그룹 간 비교")
    if "instability" in df.columns:
        # instability가 1인 그룹과 0인 그룹 분리
        group_1 = df[df["instability"] == 1]
        group_0 = df[df["instability"] == 0]

        # 사용자로부터 비교할 수치형 열 선택
        numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        numeric_columns.remove("instability")  # instability는 제외
        selected_column = st.selectbox("비교할 수치형 열을 선택하세요:", numeric_columns)

        # 두 그룹의 평균 차이 및 통계적 유의성 확인
        mean_1 = group_1[selected_column].mean()
        mean_0 = group_0[selected_column].mean()
        t_stat, p_value = ttest_ind(group_1[selected_column], group_0[selected_column], nan_policy='omit')

        st.write(f"### {selected_column} 열의 그룹별 평균")
        st.write(f"- Instability 1 (불안정 그룹): {mean_1:.2f}")
        st.write(f"- Instability 0 (안정 그룹): {mean_0:.2f}")
        st.write(f"t-통계량: {t_stat:.2f}, p-값: {p_value:.4f}")

        # 유의미한 차이가 있을 경우
        if p_value < 0.05:
            st.success("두 그룹 간에 통계적으로 유의미한 차이가 있습니다.")
            # 막대 그래프 출력
            st.bar_chart(pd.DataFrame({
                "Group": ["Instability 1", "Instability 0"],
                "Mean": [mean_1, mean_0]
            }).set_index("Group"))
        else:
            st.warning("두 그룹 간에 통계적으로 유의미한 차이가 없습니다.")
    else:
        st.error("데이터에 'instability' 열이 없습니다.")
except FileNotFoundError:
    st.error(f"CSV 파일 '{file_path}'을(를) 찾을 수 없습니다. 경로를 확인하세요.")
except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")

