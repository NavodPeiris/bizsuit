import streamlit as st
import pandas as pd
from churn.churn import churn_analyze, churn_train
from recommender.recommender import train_recommender, generate_recommendations
from wco.wco import train_wco_models, use_wco_models
import os

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from umap import UMAP
import plotly.express as px

def churn_inf(file): 
    df = None
    # Check the file type and read it appropriately
    if file.name.endswith('.xlsx') or file.name.endswith('.xls'):
        df = pd.read_excel(file)  # Read Excel file
    else:
        df = pd.read_csv(file)  # Read CSV file

    churn_list = churn_analyze(df)
    return churn_list

def churn_train_fn(file): 
    df = None
    # Check the file type and read it appropriately
    if file.name.endswith('.xlsx') or file.name.endswith('.xls'):
        df = pd.read_excel(file)  # Read Excel file
    else:
        df = pd.read_csv(file)  # Read CSV file

    message = churn_train(df)
    return message

def recommender_train_fn(file):
    df = None
    # Check the file type and read it appropriately
    if file.name.endswith('.xlsx') or file.name.endswith('.xls'):
        df = pd.read_excel(file)  # Read Excel file
    else:
        df = pd.read_csv(file)  # Read CSV file

    message = train_recommender(df)
    return message

def recommender_generation_fn(product_id):
    message = generate_recommendations(product_id)
    return message

def wco_train_fn(ar_file, ap_file):
    ar_df = None
    # Check the file type and read it appropriately
    if ar_file.name.endswith('.xlsx') or ar_file.name.endswith('.xls'):
        ar_df = pd.read_excel(ar_file)  # Read Excel file
    else:
        ar_df = pd.read_csv(ar_file)  # Read CSV file

    ap_df = None
    # Check the file type and read it appropriately
    if ap_file.name.endswith('.xlsx') or ap_file.name.endswith('.xls'):
        ap_df = pd.read_excel(ap_file)  # Read Excel file
    else:
        ap_df = pd.read_csv(ap_file)  # Read CSV file

    message = train_wco_models(ar_df, ap_df)
    return message

def wco_inf(ar_file, ap_file):
    ar_df = None
    # Check the file type and read it appropriately
    if ar_file.name.endswith('.xlsx') or ar_file.name.endswith('.xls'):
        ar_df = pd.read_excel(ar_file)  # Read Excel file
    else:
        ar_df = pd.read_csv(ar_file)  # Read CSV file

    ap_df = None
    # Check the file type and read it appropriately
    if ap_file.name.endswith('.xlsx') or ap_file.name.endswith('.xls'):
        ap_df = pd.read_excel(ap_file)  # Read Excel file
    else:
        ap_df = pd.read_csv(ap_file)  # Read CSV file

    return_df = use_wco_models(ar_df, ap_df)
    return return_df


# Set page configuration
st.set_page_config(
    page_title="BizSuit",
    initial_sidebar_state="expanded",
    page_icon="bizsuite_logo_no_background.ico", 
    layout="wide",  # Use wide layout
)

# Inject custom CSS for layout adjustments
st.markdown("""
    <style>
        /* Increase padding around the main container */
        .main {
            padding: 2rem 2rem; /* Adjusted padding values */
        }

        /* Adjust the page width */
        section.main > div {
            max-width: 95%;
            padding: 2rem; /* Adjusted padding values */
        }

        /* Increase padding for block container */
        .block-container {
            padding: 2rem 1rem; /* Adjusted padding values */
        }
    </style>
""", unsafe_allow_html=True)

# Inject custom CSS to hide the Streamlit state
hide_streamlit_style = """
            <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Add Bootstrap for styling
st.markdown('<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-T3c6CoIi6uLrA9TneNEoa7RxnatzjcDSCmG1MXxSR1GAsXEV/Dwwykc2MPK8M2HN" crossorigin="anonymous">', unsafe_allow_html=True)

# Create a sidebar for navigation
page = st.sidebar.selectbox("Select Page", ["Home", "Models", "Train", "Dashboards"])

# Page content based on selection
if page == "Home":
    st.markdown(
        """
        <div style="text-align: center;">
            <h1>BizSuite</h1>
            <h3>Welcome to Bizsuite!</h3>
            <h4>Start Training Models and Gain Valuable Insights</h4>
        </div>
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; height: 100%;">
            <h5>Services We Offer</h5>
            <br>
            <h6>● Churn Prediction</h6>
            <h6>● Recommendation Generation</h6>
            <h6>● Working Capital Optimization</h6>
            <h6>● Realtime Customer Analytics Dashboard</h6>
        </div>
        """,
        unsafe_allow_html=True
    )


elif page == "Models":
    st.title("Models Available")
    
    st.subheader("Churn Prediction Model", divider="gray")
    # churn model
    if(os.path.exists(f"./churn/RF_model.joblib")):
        @st.dialog("Upload The Input Data")
        def churn_pred_dialog():
            # File uploader for Home page
            file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])

            if st.button("Submit"):
                if file is not None:
                    churn_list = churn_inf(file)
                    st.write("Churn Analysis Results:") 
                    st.dataframe(churn_list)

        if st.button(key="churns-pred", label="Use Model"):
            churn_pred_dialog()
    else:
        st.write("Model not available. Please train the model first.")


    st.subheader("Working Capital Optimization Model", divider="gray")
    # churn model
    if(os.path.exists(f"./wco/models/ar_model.joblib") and os.path.exists(f"./wco/models/ap_model.joblib")):
        @st.dialog("Upload The Input Data")
        def wco_pred_dialog():
            # File uploader for Home page
            ar_file = st.file_uploader("Upload Excel or CSV file of Account Receivables", type=["xlsx", "xls", "csv"])
            ap_file = st.file_uploader("Upload Excel or CSV file of Account Payables", type=["xlsx", "xls", "csv"])

            if st.button("Submit"):
                if ar_file is not None and ap_file is not None:
                    wco_rslt_df = wco_inf(ar_file, ap_file)
                    st.write("Working Capital Optimization Results:") 
                    st.dataframe(wco_rslt_df)

        if st.button(key="wco-pred", label="Use Model"):
            wco_pred_dialog()
    else:
        st.write("Model not available. Please train the model first.")


    st.subheader("Recommendation Model API", divider="gray")
    if(os.path.exists(f"./recommender/Model/graph_embeddings.model")):
        # API URL with copy button
        api_url = "http://127.0.0.1:8000/get_recommendation/{product_id}"
        st.code(api_url, language="plaintext")

        # Response format example
        st.write("### Example Response:")
        response_example = """
        {
            "prod_id": "28717037",
            "category_code": "apparel.shoes.keds"
        }
        """
        st.code(response_example, language="json")
    else:
        st.write("Model not available. Please train the model first.")


elif page == "Train":
    st.title("Train")

    st.subheader("Churn Prediction Model", divider="gray")
    churn_df = pd.read_excel("./churn/E Commerce Dataset Test.xlsx")
    st.text("Sample Data:")
    st.dataframe(churn_df.head())

    @st.dialog("Upload The Input Data")
    def churn_train_dialog():
        # File uploader for Home page
        file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])

        if st.button("Submit"):
            if file is not None:
                message = churn_train_fn(file)
                st.write("Model Train Status: ", message) 

    if st.button(key="churns-train", label="Train Model"):
        churn_train_dialog()


    st.subheader("Working Capital Optimization Model", divider="gray")
    wco_payables_df = pd.read_excel("./wco/data/payables_data.xls")
    wco_receivables_df = pd.read_excel("./wco/data/receivables_data.xls")

    st.text("Sample Receivables Data:")
    st.dataframe(wco_receivables_df.head())
    st.text("Sample Payables Data:")
    st.dataframe(wco_payables_df.head())
    

    @st.dialog("Upload The Input Data")
    def wco_train_dialog():
        # File uploader for Home page
        ar_file = st.file_uploader("Upload Excel or CSV file for Account Receivables", type=["xlsx", "xls", "csv"])
        ap_file = st.file_uploader("Upload Excel or CSV file for Account Payables", type=["xlsx", "xls", "csv"])

        if st.button("Submit"):
            if ar_file is not None and ap_file is not None:
                message = wco_train_fn(ar_file, ap_file)
                st.write("Model Train Status: ", message) 

    if st.button(key="wco-train", label="Train Model"):
        wco_train_dialog()


    st.subheader("Recommendation Model", divider="gray")
    recommender_df = pd.read_csv("./recommender/raw_data.csv")
    st.text("Sample Data:")
    st.dataframe(recommender_df.head())

    @st.dialog("Upload The Input Data")
    def recommender_train_dialog():
        # File uploader for Home page
        file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])

        if st.button("Submit"):
            if file is not None:
                message = recommender_train_fn(file)
                st.write("Model Train Status: ", message) 
                
                GRAPH_FILE_NAME = 'undirected_weighted_product_views_graph.parquet'
                
                # Load data
                df_full = pd.read_parquet('./recommender/Data/optimised_raw_data.parquet')
                df_node2vec = pd.read_parquet('./recommender/Data/Embedding_Data/node2vec_embedding_df_{}.parquet'.format(GRAPH_FILE_NAME.split('.')[0]))

                # Preprocess
                df_full = df_full[['product_id', 'category_code']]
                df_full.drop_duplicates(subset=['product_id'], inplace=True)
                df_full.columns = ['pid', 'category_code']
                df_node2vec = df_node2vec.merge(df_full, on=['pid'], how='left')

                # Function to split category levels
                def level_split(category):
                    result = [None] * 3
                    if not category or type(category) != str or category == '':
                        return result
                    try:
                        d = category.split('.')
                        result[:0] = d
                    except:
                        print("Error", category)
                        pass
                    return result[:3]

                # Apply the level splitting function in parallel
                result_level = Parallel(n_jobs=-1, verbose=0)(delayed(level_split)(x) for x in tqdm(df_node2vec.category_code.values))
                category_split_df = pd.DataFrame(result_level, columns=['L1', 'L2', 'L3'])

                # Add category levels to DataFrame
                df_node2vec[['L1', 'L2', 'L3']] = category_split_df[['L1', 'L2', 'L3']]

                # Create embedding matrix
                embedding = np.stack(df_node2vec[~df_node2vec.L1.isna()].embedding_vector.values.tolist())

                # UMAP projection
                umap_2d = UMAP(n_components=2, init='random', random_state=0, n_jobs=-1, verbose=True, metric='cosine', low_memory=False)
                umap_3d = UMAP(n_components=3, init='random', random_state=0, n_jobs=-1, verbose=True, metric='cosine', low_memory=False)

                proj_2d = umap_2d.fit_transform(embedding)
                proj_3d = umap_3d.fit_transform(embedding)

                # 2D Plot
                fig_2d = px.scatter(
                    proj_2d[:30000], x=0, y=1,
                    color=df_node2vec[~df_node2vec.L1.isna()].head(30000).L1, labels={'color': 'L1'}
                )

                # 3D Plot
                fig_3d = px.scatter_3d(
                    proj_3d[:30000], x=0, y=1, z=2,
                    color=df_node2vec[~df_node2vec.L1.isna()].head(30000).L1, labels={'color': 'L1'}
                )
                fig_3d.update_traces(marker_size=5)

                # Display in Streamlit
                st.title('UMAP Product Embedding Visualization')

                # Show 2D UMAP
                st.subheader('2D UMAP Visualization')
                st.plotly_chart(fig_2d)

                # Show 3D UMAP
                st.subheader('3D UMAP Visualization')
                st.plotly_chart(fig_3d)

    if st.button(key="recommendation-train", label="Train Model"):
        recommender_train_dialog()

elif page == "Dashboards":
    # URL of the Grafana dashboard or panel (make sure it's publicly accessible or authenticated)
    grafana_url = "http://localhost:3003"

    # Embed Grafana view in the Streamlit app
    st.markdown(f"""
        <iframe src="{grafana_url}" width="100%" height="1080px"></iframe>
    """, unsafe_allow_html=True)

