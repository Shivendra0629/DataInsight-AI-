# 🚀 DataInsight AI

### Commercial AI-Powered Business Analytics & Intelligence Platform

DataInsight AI is an executive data analytics platform built using **Python**, **Streamlit**, **Pandas**, **Plotly**, and **Groq (Llama 3.3 70B Versatile)**.

It enables business leaders, managers, and data analysts to upload datasets, audit data health, automatically calculate executive financial KPIs, explore interactive visual dynamics, analyze product profitability, and receive evidence-based strategic recommendations.

---

# 🌐 Live Demo

👉 https://datainsightai.streamlit.app/

---

# ✨ Core Features

## 💼 Business Performance Dashboard (New)

An executive-grade SaaS business intelligence dashboard that automatically maps columns and calculates key commercial metrics:

- **Executive KPI Cards**: Total Revenue, Net Profit, Profit Margin (%), Total Loss from Negative-Profit Records, Average Order Value (AOV), and Growth Trends (MoM/WoW).
- **Automatic Column Detection**: Pattern-matches date, product, category, revenue, cost, profit, quantity, price, order ID, and customer fields with an intuitive user override drawer.
- **Smart Profit Engine**: Directly handles precomputed profit or dynamically computes `Net Profit = Revenue − Cost`. Clearly documents the calculation method and assumptions.
- **Automatic Visual Analytics**: 2 to 4 responsive Plotly charts answering core business questions (e.g., Performance Trajectory, Product Profitability Diverging Bar, Revenue vs. Profit Bubble Quadrants, Category Comparison) with plain-language explanations.
- **Product Profitability & Unit Economics**: Interactive table classifying products as *Top Performer*, *Profitable*, *Loss-Maker 🔴*, or *High Revenue / Low Margin ⚠️*, with CSV export.
- **Evidence-Based Recommendations Engine**: 4-part transparent strategic action items (Observation, Supporting Evidence, Business Impact, Recommended Next Steps).
- **AI Executive Briefing**: Groq LLM synthesis of verified pandas-calculated metrics with automatic deterministic fallback if the API is offline.
- **Advanced Visualization Studio**: Analyst sandbox for custom dimensions, measures, aggregations (Sum, Mean, Median, Count, Min, Max), color groupings, and chart types.

---

## 📂 Data Processing

Clean and preprocess datasets without writing code:

- Upload CSV and Excel (`.xlsx`, `.xls`) datasets
- Preview uploaded data and schema inspection
- Identify missing values and duplicate records
- Automated imputation and deduplication
- Instant download of cleaned CSV and Excel files

---

## 📈 Visual Analytics

Interactive data exploration using Plotly:

- 📊 Bar Chart & Horizontal Bar
- 📈 Histogram & Distribution Curves
- 🔵 Scatter Plot
- 🥧 Pie / Donut Chart
- 📦 Box Plot
- 🔥 Correlation Heatmap

---

## 🧠 AI Insights & Reports

Powered by **Groq Cloud** using **Llama 3.3 70B Versatile**:

- Ask questions in natural language
- Statistical summaries, trend observations, and anomaly detection
- Machine Learning readiness evaluation
- Downloadable management reports

---

# 🛠 Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.12+ / 3.14 | Core Programming Language |
| Streamlit | Web Application Framework |
| Pandas | Data Manipulation & Financial Calculations |
| Plotly | Interactive Visual Analytics |
| Groq Cloud API | Llama 3.3 70B Versatile LLM Inference |
| OpenPyXL | Excel File Processing |

---

# 📂 Project Structure

```text
DataInsight-AI/
│
├── Home.py                               # Landing page & navigation hub
│
├── pages/
│   ├── 1_📂 Data Processing.py          # Data upload, inspection & cleaning
│   ├── 2_💼 Business Dashboard.py        # Executive Business Performance Dashboard
│   ├── 3_📈 Visual Analytics.py         # Interactive manual visual explorer
│   └── 4_🧠 AI Insights & Reports.py    # Natural language AI query assistant
│
├── utils/
│   ├── analytics.py                      # Column mapping, KPI math, product & rec engine
│   ├── charts.py                         # Professional SaaS Plotly chart builders
│   ├── ai_insights.py                    # Verified analytical summary for Groq LLM
│   └── helper.py                         # Currency, percentage, and number formatters
│
├── tests/
│   └── test_dashboard.py                 # Automated unit tests covering all 10 edge cases
│
├── assets/
├── .streamlit/
│   └── secrets.toml                      # Groq API configuration
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🚀 Installation & Running

### 1. Clone the repository

```bash
git clone https://github.com/Shivendra0629/DataInsight-AI.git
cd DataInsight-AI
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Groq API Key

Create or edit `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

*(Note: The dashboard functions deterministically even without an API key).*

### 5. Run the application

```bash
streamlit run Home.py
```

---

# 🧪 Running Automated Tests

Run the full test suite verifying all 10 business edge cases (product/revenue/cost, direct profit, date trends, non-financial data, missing values, negative profit, zero revenue, ambiguous headers, empty datasets, and high-cardinality catalogs):

```bash
python -m unittest tests/test_dashboard.py
```

---

# 👨‍💻 Developed by

**Shivendra Mahato** — *DataInsight AI*
