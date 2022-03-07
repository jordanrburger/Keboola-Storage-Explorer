import numpy as np
import pandas as pd
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import kbcstorage as kbc
from kbcstorage.client import Client


    
# Web App Title
st.markdown('''
# **The Keboola Storage EDA App**

This is the **EDA App** created in Streamlit using the **pandas-profiling** library.

Simply select a Keboola Storage Bucket and a Keboola Storage Table and the app will generate a profile report.

---
''')
# Enter your Keboola Storage connection details
with st.sidebar.markdown('''
# Enter your Keboola Storage connection details
'''):


# Enter your Keboola Storage connection details
 connection_url = st.sidebar.selectbox('Connection URL', ['https://connection.keboola.com/', 'https://connection.north-europe.azure.keboola.com/', 'https://connection.eu-central-1.keboola.com/'])
 api_token = st.sidebar.text_input('API Token', 'Enter Password', type="password")
# Create a Keboola Storage Client
#click to connect

st.session_state['client'] = Client(connection_url, api_token)
#client = Client(connection_url, api_token)
#try:
#    client.buckets.list()
#    st.sidebar.success('Connected to Keboola Storage')
#except Exception as e:
#    st.sidebar.error('Could not connect to Keboola Storage')
with st.sidebar.header('Select a bucket from storage'):
    def get_buckets():
        buckets = st.session_state['client'].buckets.list()
        buckets_df = pd.DataFrame(buckets)
        bucket = st.sidebar.selectbox('Bucket', buckets_df['name'])
        bucket_id = buckets_df[buckets_df['name'] == bucket]['id'].values[0]
        return bucket_id
    bucket_id = get_buckets()
    
with st.sidebar.header('Select a table from the bucket'):
                # Select a table from the bucket
    def get_tables():
        tables = st.session_state['client'].buckets.list_tables(bucket_id=bucket_id)

        tables_df = pd.DataFrame(tables)
        table = st.sidebar.selectbox('Table', tables_df['name'], on_change=get_tables)
        table_id = tables_df[tables_df['name'] == table]['id'].values[0]
        return table_id
    id = get_tables()
    
    uploaded_file = st.session_state['client'].tables.export_to_file(table_id=id, path_name='.')
    
  
if st.sidebar.button('Generate Pandas Profiling Report'):    
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header('**Input DataFrame**')
    st.write(df)
    st.write('---')
    st.header('**Pandas Profiling Report**')
    st_profile_report(pr)

