import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency, fisher_exact

# Streamlit 앱 제목
st.title("CSV 데이터 분석 및 시각화")

# CSV 파일 경로
file_path = "4DMR.csv"

try:
    # CSV 파일 읽기
    df = pd.read_csv(file_path)
    
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
                        horizontalalignment="center", color="red", fontsize=12)
            else:
                st.warning("두 그룹 간에 통계적으로 유의미한 차이가 없습니다.")
            st.pyplot(fig)
        else:
            # 범주형 데이터: 카이제곱 검정 또는 Fisher의 정확 검정
            contingency_table = pd.crosstab(df_clean[selected_column], df_clean[instability_column])
            st.write("### 교차표")
            st.write(contingency_table)

            # 카이제곱 검정
            chi2, p_value, _, _ = chi2_contingency(contingency_table)

            st.write(f"카이제곱 통계량: {chi2:.2f}, p-값: {p_value:.4f}")

            # 막대 그래프
            fig, ax = plt.subplots(figsize=(8, 6))
            contingency_table.plot(kind="bar", stacked=True, ax=ax, color=["blue", "orange"])
            ax.set_title(f"{selected_column} 값의 Instability 그룹 간 비교")
            ax.set_xlabel(selected_column)
            ax.set_ylabel("빈도수")
            ax.legend(title="Instability", labels=["안정 (0)", "불안정 (1)"])

            if p_value < 0.05:
                st.success("두 그룹 간에 통계적으로 유의미한 차이가 있습니다.")
                ax.text(0.5, contingency_table.values.max() + 1, f"p-value = {p_value:.4f}",
                        horizontalalignment="center", color="red", fontsize=12)
            else:
                st.warning("두 그룹 간에 통계적으로 유의미한 차이가 없습니다.")
            st.pyplot(fig)
    else:
        st.error(f"'{instability_column}' 열이 없습니다.")
except FileNotFoundError:
    st.error(f"CSV 파일 '{file_path}'을(를) 찾을 수 없습니다. 경로를 확인하세요.")
except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
