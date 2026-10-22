import streamlit as st

st.set_page_config(page_title="Home", page_icon="🏡", layout="wide")

st.title("📊 **DataInsight AI**")

st.markdown(
    "### AI-Powered Business Analytics & Intelligence Platform"
)
st.divider()
st.info(
    "👋 Welcome to DataInsight AI! Upload your dataset and transform raw data into executive business intelligence, interactive visualizations, and AI-powered strategic recommendations."
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="📂 Supported Formats",
    value="CSV & Excel"
)

col2.metric(
    label="⚡ Data Cleaning",
    value="Automated"
)

col3.metric(
    label="💼 Business Intelligence",
    value="Executive KPIs"
)

col4.metric(
    label="🤖 AI Engine",
    value="Groq LLM"
)

st.divider()

st.header("Analyze • Visualize • Discover • Decide")

st.caption("Clean your data, track business KPIs, uncover profitability insights, and make smarter strategic decisions.")

st.divider()

st.subheader("🌟 Why Choose DataInsight AI?")
feature_column1, feature_column2, feature_column3, feature_column4 = st.columns(4)

with feature_column1:
    with st.container(border=True):
        st.subheader("🧹 Automatic Cleaning")
        st.write(
            "Automatically removes duplicate records, fills missing values, and prepares high-quality datasets for analysis."
        )

with feature_column2:
    with st.container(border=True):
        st.subheader("💼 Executive Dashboard")
        st.write(
            "Automated KPI calculation (revenue, net profit, margins, loss records), unit economics, and evidence-based recommendations."
        )

with feature_column3:
    with st.container(border=True):
        st.subheader("📊 Visual Analytics")
        st.write(
            "Interactive Plotly visualizations and an advanced visualization studio to explore custom dimensions, metrics, and distributions."
        )

with feature_column4:
    with st.container(border=True):
        st.subheader("🤖 AI Business Assistant")
        st.write(
            "Ask natural language questions and receive executive-level briefings, trend analyses, and downloadable reports."
        )

st.divider()

st.subheader("⚡ How It Works")
st.write("Transform your data into actionable business intelligence in simple steps.")
step1, step2, step3, step4, step5 = st.columns(5)

with step1:
    with st.container(border=True):
        st.subheader("①")
        st.markdown("# 📂")
        st.markdown("### Upload Data")
        st.caption("Upload your business dataset in CSV or Excel format to begin.")

with step2:
    with st.container(border=True):
        st.subheader("②")
        st.markdown("# 🧹")
        st.markdown("### Clean & Audit")
        st.caption("Audit data health, remove duplicates, and impute missing values.")

with step3:
    with st.container(border=True):
        st.subheader("③")
        st.markdown("# 💼")
        st.markdown("### Business KPIs")
        st.caption("Review automated financial KPIs, product margins, and recommendations.")

with step4:
    with st.container(border=True):
        st.subheader("④")
        st.markdown("# 📊")
        st.markdown("### Visual Analytics")
        st.caption("Explore interactive charts, distribution curves, and custom studio plots.")

with step5:
    with st.container(border=True):
        st.subheader("⑤")
        st.markdown("# 🤖")
        st.markdown("### AI & Reports")
        st.caption("Consult the AI assistant and export documented management reports.")

st.info("""
🚀 **Get Started** 
           
1. Click **📂 Data Processing** in the left sidebar to upload and clean your dataset.
2. Head to **💼 Business Dashboard** to instantly explore executive metrics, product rankings, and recommendations!
""")

st.divider()

st.markdown("""
### 📂 Supported Formats

✅ CSV
            
✅ Excel (.xlsx, .xls)
            
🟡 JSON - Coming Soon 

🟡 Parquet - Coming Soon
""")

st.markdown("---")
st.markdown("### 👨‍💻 Developed by **Shivendra Mahato**")

st.caption(
    "Version 2.0 • Powered by Streamlit • Pandas • Plotly • Groq AI"
)
