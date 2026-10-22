import streamlit as st 
import pandas as pd 
from io import BytesIO

st.title("📂 Dataset Preprocessing")
st.subheader("Upload your CSV dataset, review its structure, clean it automatically, and download the processed file.")

uploaded_file =st.file_uploader("Choose your file",type=["csv","xlsx", "xls"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    st.session_state["Dataset"]=df
    st.success("✅ File uploaded successfully")
    st.divider()

    st.subheader("📊 Dataset Overview")

    missing=df.isnull().sum().sum()
    duplicate=df.duplicated().sum()
    
    overview_col1,overview_col2,overview_col3,overview_col4=st.columns(4)

    overview_col1.metric(
    label="📄 Rows",
    value=df.shape[0]
)

    overview_col2.metric(
    label="📑 Columns",
    value=df.shape[1]
)

    overview_col3.metric(
    label="⚠️ Missing",
    value=missing
)

    overview_col4.metric(
    label="🔁 Duplicates",
    value=duplicate
)
    st.subheader("📋 Dataset Information")
   
    structure_df= pd.DataFrame({
        "Column Name": df.columns,

        "Data Types":df.dtypes.astype(str),

        "Unique":df.nunique()
})
   
    st.dataframe(structure_df, use_container_width=True)

    st.divider() 
    st.header("👀 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.divider()
    if missing == 0 and duplicate == 0:
        st.success("""🟢 Dataset Quality 
                   
        Status : Ready for Analysis
        
        No missing values or duplicate records detected.
            """)
    else:
        st.error(f""" 🔴 Dataset Quality
                 
        Status : Cleaning Recommended
                 
         {duplicate} duplicate rows and {missing} missing values detected.

        """)
else:
    st.info("ℹ️ Please upload a CSV file to begin analysis.")


if "Dataset" in st.session_state:
    cleaned_df = st.session_state["Dataset"].copy()
    st.divider()
    st.subheader("🤖 Automatic Data Cleaning")
    with st.spinner("Cleaning your dataset..."):
        before_rows = cleaned_df.shape[0]
        before_columns = cleaned_df.shape[1]
        before_missing = cleaned_df.isnull().sum().sum()
        before_duplicates = cleaned_df.duplicated().sum()
        
        cleaned_df.drop_duplicates(inplace=True)
        cleaned_df.dropna(how="all", inplace=True)
        cleaned_df.dropna(axis=1, how="all", inplace=True)
        for col in cleaned_df.columns:
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                median_val = cleaned_df[col].median()
                cleaned_df[col] = cleaned_df[col].fillna(0 if pd.isna(median_val) else median_val)
            else:
                modes = cleaned_df[col].mode()
                cleaned_df[col] = cleaned_df[col].fillna(modes.iloc[0] if not modes.empty else 'Unknown')
    
    st.session_state["Cleaned_Dataset"] = cleaned_df
    st.success("✅ Dataset cleaned successfully!")
    st.divider()



if "Cleaned_Dataset" in st.session_state:

    st.subheader("📥 Download Clean Dataset")

    cleaned_df = st.session_state["Cleaned_Dataset"]

    col1, col2 = st.columns(2)

    # CSV Download
    csv = cleaned_df.to_csv(index=False).encode("utf-8")

    with col1:
        st.download_button(
            label="📄 Download CSV",
            data=csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Excel Download
    buffer = BytesIO()  #Create an empty temporary file in memory.

                            #.ExcelWriter(buffer)---> writes into memory.

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:   #engine --> excelwriter doesn't know how to create excel file so uses openpyxl where pandas say to create file 
        cleaned_df.to_excel(writer, index=False)

    xlsx = buffer.getvalue()

    with col2:
        st.download_button(
            label="📊 Download Excel",
            data=xlsx,
            file_name="cleaned_dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
         
    after_rows = cleaned_df.shape[0]
    after_columns = cleaned_df.shape[1]
    after_missing = cleaned_df.isnull().sum().sum()
    after_duplicates = cleaned_df.duplicated().sum()

    
    st.header("👀 Cleaned Dataset Preview")
    st.dataframe(cleaned_df.head(10),use_container_width=True)
    st.divider()
    st.header("📋 Cleaning Summary")
    
    summary_1,summary_2,summary_3,summary_4=st.columns(4)

    summary_1.metric(
        label="🗑 Rows Removed",
        value=before_rows-after_rows
    )
    summary_2.metric(
        label="📑 Columns Removed",
        value=before_columns-after_columns
    )
    summary_3.metric(
        label="🩹 Missing Values Filled",
        value=before_missing-after_missing
    )
    summary_4.metric(
        label="♻️ Duplicate Rows Removed ",
        value=before_duplicates-after_duplicates
    )
   
    

    
         




