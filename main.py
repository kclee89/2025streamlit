import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency

# Streamlit 앱 제목
st.title("CSV 데이터 분석 및 시각화")

# CSV 파일 경로
file_path = "4DMR.csv"

try:
    # CSV 파일 읽기
    df = pd.read_csv(file_path)
    
    # 중복된 열 이름 처리
    if df.columns.duplicated().any():
        st.warning("중복된 열 이름이 감지되었습니다. 고유한 이름으로 자동 수정합니다.")
        df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)

    # 괄호로 처리된 글자 제거 (컬럼 이름 정리)
    df.columns = [col.split("(")[0].strip() for col in df.columns]

    # 데이터프레임 표시
    st.header("데이터 미리보기")
    st.write(df)
    
    # 데이터 정보
    st.header("데이터 정보")
    st.write(f"행 개수: {df.shape[0]}, 열 개수: {df.shape[1]}")
    st.write("컬럼 정보:")
    st.write(df.dtypes)

    # Instability 그룹 비교
    st.header("Instability 그룹 간 분석")
    instability_column = "Instability"
    if instability_column in df.columns:
        # 사용자로부터 분석할 열 선택
        numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        numeric_columns.remove(instability_column)
        categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # 분석 대상 열 선택
        selected_column = st.selectbox("분석할 열을 선택하세요:", numeric_columns + categorical_columns)

        # 결측값 처리
        df_clean = df.dropna(subset=[selected_column, instability_column])

        if selected_column in numeric_columns:
            # 수치형 데이터: t-test
            group_1 = df_clean[df_clean[instability_column] == 1][selected_column]
            group_0 = df_clean[df_clean[instability_column] == 0][selected_column]

            t_stat, p_value = ttest_ind(group_1, group_0, equal_var=False)

            st.write(f"### {selected_column} 열의 그룹별 평균")
            st.write(f"- Instability 1 (불안정 그룹): {group_1.mean():.2f}")
            st.write(f"- Instability 0 (안정 그룹): {group_0.mean():.2f}")
            st.write(f"t-통계량: {t_stat:.2f}, p-값: {p_value:.4f}")

            # Boxplot 그래프
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=df_clean, x=instability_column, y=selected_column, ax=ax)
            sns.stripplot(data=df_clean, x=instability_column, y=selected_column, color="black", alpha=0.5, jitter=True, ax=ax)
            ax.set_title(f"{selected_column} 값의 Instability 그룹 간 비교")
            ax.set_xlabel("Instability (0: 안정, 1: 불안정)")
            ax.set_ylabel(selected_column)

            if p_value < 0.05:
                st.success("두 그룹 간에 통계적으로 유의미한 차이가 있습니다.")
                ax.text(0.5, max(df_clean[selected_column]), f"p-value = {p_value:.4f}",
                        horizontalalignment="center", color
