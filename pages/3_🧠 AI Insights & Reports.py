import streamlit as st
import pandas as pd
from groq import Groq
from datetime import datetime

st.title("🧠 Get AI-Powered Business Insights")
st.subheader("Ask questions about your dataset and receive AI-generated insights.")
client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)

if "Cleaned_Dataset" in st.session_state:
    df=st.session_state["Cleaned_Dataset"]
    st.info(
    f"📊 Dataset Loaded: **{df.shape[0]} rows × {df.shape[1]} columns**"
)
    st.success("✅ Using the cleaned dataset for AI-powered insights.")
elif "Dataset" in st.session_state:
    df=st.session_state["Dataset"]
    st.warning("⚠️ Dataset has not been cleaned yet. AI insights will be generated using the original uploaded dataset.")
    
else:
    st.info("ℹ️ Please upload a dataset from the 'Upload CSV Dataset' page first.")
    st.stop()


st.header("💬 Ask Your Question")
st.caption("💡 Example questions you can ask:")

st.markdown("""
- Which category has the highest sales?
- What trends do you observe in the dataset?
- Which products are underperforming?
- Are there any anomalies or outliers?
- What business recommendations would you make?
""")
user_question = st.text_area(
    "Enter your question",
    height=150,
    placeholder="Example: Which products generate the highest revenue and what should the company focus on?"
)

btn=st.button("🧠 Generate AI Insights",use_container_width=True,
    disabled=not user_question.strip())

if btn:
   if  user_question.strip() =="":
    st.warning("⚠️ Please enter a question before generating AI insights.")
   else:
      prompt = f"""
You are an expert Data Scientist, Data Analyst, Business Intelligence Consultant, and Machine Learning Engineer.

Your role is to analyze datasets professionally and provide accurate insights.

Dataset Information
===================

Rows: {df.shape[0]}
Columns: {df.shape[1]}

Column Names:
{', '.join(df.columns)}

Data Types:
{df.dtypes.to_string()}

Summary Statistics:
{df.describe(include='all').fillna('').round(2).to_string()}

Missing Values:
{df.isnull().sum().to_string()}

Duplicate Rows:
{df.duplicated().sum()}

Sample Records:
{df.head(5)}

User Question:
{user_question}

Instructions

- Analyze the ENTIRE dataset using the provided statistics, not only the sample rows.
- Use the sample rows only to understand the structure of the dataset.
- Never assume facts that are not supported by the data.
- Never mention "Based on the sample..." unless the requested information truly cannot be inferred.
- If exact calculations are impossible from the provided information, clearly explain WHY.
- Write like a professional Data Scientist preparing an analysis report.
- Avoid generic statements.
- Be concise but insightful.
- Use proper Markdown headings.

Your response should contain the following sections whenever applicable:

## Executive Summary
Provide a concise answer to the user's question.

## Key Findings
Identify meaningful trends, distributions, correlations, anomalies, or observations.

## Data Quality Assessment
Mention:
- Missing values
- Duplicate rows
- Possible outliers
- Class imbalance
- Data consistency issues

## Exploratory Data Analysis (EDA)
Describe:
- Feature distributions
- Relationships
- Interesting observations
- Potential visualizations that would be useful

## Machine Learning Readiness
Evaluate whether the dataset is suitable for ML.

Recommend:
- Target variable (if applicable)
- Feature engineering
- Encoding
- Scaling
- Feature selection
- Train-test split considerations
- Suitable ML algorithms

## Business Insights
Only include this section if business-related insights exist.

## Recommendations
Provide practical next steps.

## Limitations
Mention only genuine limitations.
Do NOT say "only sample data" unless the requested answer truly requires calculations that cannot be derived from the provided statistics.

Do not hallucinate.
Do not invent numbers.
Do not repeat the dataset information unnecessarily.
"""
      
      try:
        with st.spinner("🤖 AI is analyzing your dataset..."):
           response = client.chat.completions.create(
               model="openai/gpt-oss-120b",
               messages=[
                  {
                    "role": "user",
                    "content": prompt
                  }
               ],
               temperature=0.3
            )

           answer = response.choices[0].message.content
            
        st.success("✅ AI Insights Generated Successfully")
        st.session_state["AI_Insights"] =answer
        st.session_state["User_Question"] = user_question
        st.divider()
        st.header("📊 AI Business Insights")
        with st.container(border=True):
            st.markdown(answer)

      except Exception as e:
        st.error(f"❌ Error: {e}")

st.divider()
st.header("⬇️ Download Report")


if "AI_Insights" in st.session_state:
    st.success("✅ AI report is ready for download.")
    ai_report = st.session_state["AI_Insights"]
    if "Cleaned_Dataset" in st.session_state:
        report_df = st.session_state["Cleaned_Dataset"]
    else:
        report_df = st.session_state["Dataset"]

    report = f"""
    =============================
    AI BUSINESS REPORT
    =============================

    Dataset Summary
    ---------------
    Rows    : {report_df.shape[0]}
    Columns : {report_df.shape[1]}
    
    User Question
    -------------
    {st.session_state["User_Question"]}
    
    AI Business Insights
    --------------------
    {ai_report}

    Generated On
    ------------
    {datetime.now().strftime("%d-%m-%Y %H:%M")}
    =============================
    Generated using DataInsight AI
    =============================
    """
    st.caption(
    f"Report Length: {len(ai_report.split())} words"
)
    report_bytes = report.encode("utf-8")
    st.download_button(
        label="⬇️ Download AI Insights Report",
        data=report_bytes,
        file_name="DataInsight_AI_Report.txt",
        mime="text/plain"
)

if "Cleaned_Dataset" not in st.session_state and "AI_Insights" not in st.session_state:
    st.info("ℹ️ Nothing is available to download yet.")
