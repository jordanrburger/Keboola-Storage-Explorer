import numpy as np
import pandas as pd
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import kbcstorage as kbc
from kbcstorage.client import Client
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
import plotly.express as px
import datetime as dt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

    
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
 st.session_state['api_token'] = st.sidebar.text_input('API Token', 'Enter Password', type="password")
# Create a Keboola Storage Client
#click to connect

st.session_state['client'] = Client(connection_url, st.session_state['api_token'] )
#client = Client(connection_url, api_token)
#try:
#    client.buckets.list()
#    st.sidebar.success('Connected to Keboola Storage')
#except Exception as e:
#    st.sidebar.error('Could not connect to Keboola Storage')
with st.sidebar.header('Select a bucket from storage'):
    def get_buckets():
        st.session_state['buckets'] = st.session_state['client'].buckets.list()
        st.session_state['buckets_df'] = pd.DataFrame(st.session_state['buckets'])
        st.session_state['bucket'] = st.sidebar.selectbox('Bucket', st.session_state['buckets_df']['name'])
        st.session_state['bucket_id'] = st.session_state['buckets_df'][st.session_state['buckets_df']['name'] == st.session_state['bucket']]['id'].values[0]
        return st.session_state['bucket_id']
    st.session_state['bucket_id'] = get_buckets()
    
with st.sidebar.header('Select a table from the bucket'):
                # Select a table from the bucket
    def get_tables():
        st.session_state['tables'] = st.session_state['client'].buckets.list_tables(bucket_id=st.session_state['bucket_id'])

        tables_df = pd.DataFrame(st.session_state['tables'])
        table = st.sidebar.selectbox('Table', tables_df['name'])
        table_id = tables_df[tables_df['name'] == table]['id'].values[0]
        return table_id
    id = get_tables()
    
    st.session_state['uploaded_file'] = st.session_state['client'].tables.export_to_file(table_id=id, path_name='.')
    
query_df = pd.read_csv(st.session_state['uploaded_file'] )
########################################################################################################################
######################################### INTERACTIVE TABLE ############################################################
st.header("📊 Explore")
def aggrid_interactive_table(df: pd.DataFrame):
    """Creates an st-aggrid interactive table based on a dataframe.
    Args:
        df (pd.DataFrame]): Source dataframe
    Returns:
        dict: The selected row
    """
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )

    options.configure_side_bar()
    options.configure_selection("single")
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        theme="dark",
        height=800,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )

    return selection  # return the selected row


selection = aggrid_interactive_table(df=query_df)  # create the table

########################################################################################################################
######################################### MORE OPTIONS SECTION ########################################################

st.write('---')

with st.expander("More Options"):
    with st.container():
        st.download_button(  # add a download button
            label="Download data as CSV",
            data=st.session_state['uploaded_file'] ,
            mime='text/csv',
        )
        if st.button('Generate Pandas Profiling Report'):    # add a button to the sidebar
            pr = ProfileReport(query_df, explorative=True)
            st.header('**Input DataFrame**')
            st.write(query_df)
            st.write('---')
            st.header('**Pandas Profiling Report**')
            st_profile_report(pr)


st.write('---')

########################################################################################################################
######################################### VISUALIZATION SECTION ########################################################

with st.expander("Visualization Builder"):
    x = st.selectbox("Select a dimension", query_df.columns, index=0)  # add a selectbox to the sidebar
    z = st.selectbox("Select a breakdown dimension", query_df.columns, index=1)  # add a selectbox to the sidebar
    y = st.selectbox("Select a metric", query_df.columns, index=2)  # add a selectbox to the sidebar
    viz_type = st.selectbox("Select a visualization type", ["line", "bar", "pie", "scatter"])  # add a selectbox to the sidebar

     # create a dataframe with the selected dimensions and metrics

    if st.button("Generate Visualization"):
        if viz_type == 'line':
            fig = px.line(query_df, x = query_df[x], y = query_df[y], color = query_df[z])
            st.plotly_chart(fig)
        elif viz_type == 'bar':
            fig = px.bar(query_df, x=x, y=y, color=z)
            st.plotly_chart(fig)
        elif viz_type == 'pie':
            fig = px.pie(query_df, values= query_df[y], names = query_df[z])
            st.plotly_chart(fig)
        else:
            fig = px.scatter(query_df, x=x, y=y, color=z)  # create a scatter plot
            st.plotly_chart(fig)  # plot the figure
    


st.write('---')

########################################################################################################################
######################################### FORECASTING SECTION ########################################################

with st.expander("Forecasting Area"):
    st.header('**Forecasting**')
    with st.container():
        ds_selection = st.selectbox("Select a date column", query_df.columns)

        # create 3 containers for the forecasting area
        container1_col1, container1_col2, container1_col3 = st.columns(3)

        # Container 1: select the frequency
        with container1_col1:
            freq = st.selectbox(
                "Is the data period monthly or daily?", ("Monthly", "Weekly", "Daily"))
        # Container 2: select the number of periods
        with container1_col2:
            period = st.number_input(
                "Number of periods to forecast", value=1, min_value=1, max_value=365)
        # Container 3: select the seasonality mode
        with container1_col3:
            seasonality = st.selectbox(
                "Seasonality Mode", ("additive", "multiplicative"))

    with st.container():
        container2_col1, container2_col2 = st.columns(2)
        with container2_col1:
            changepoint_prior_scale = st.slider(
                "changepoint_prior_scale", 0.01, 0.99, 0.05)
        with container2_col2:
            changepoint_range = st.slider(
                "changepoint_range", 0.01, 0.99, 0.05)

    if freq == "Monthly":
        freq = "MS"
    elif freq == "Weekly":
        freq = "W"
    else:
        freq = "D"

    y = st.selectbox(
        "Select a target column (it must be numeric)", query_df.columns, index=1)

    if st.button("Forecast"):
        with st.spinner('Forecasting...'):
            ds = query_df[ds_selection]
            # group ds by index and sum
            #ds = ds.groupby(ds.index).sum()
            y = query_df[y]
            # cast y to int
            y = y.astype(int)
            df = pd.DataFrame({'ds': ds, 'y': y})
            df = df.groupby('ds').agg('sum')
            df = df.reset_index()
            st.write(df)
            today = dt.datetime.today()
            today = pd.to_datetime(today)
            today = today.strftime('%Y-%m-%d')
            df = df.query('ds <= @today')
            m = Prophet(seasonality_mode=seasonality,
                        changepoint_prior_scale=changepoint_prior_scale, changepoint_range=changepoint_range)
            m.fit(df)
            future = m.make_future_dataframe(periods=period, freq=freq)
            forecast = m.predict(future)
            st.write(forecast)
            fig1 = m.plot(forecast)
            fig2 = m.plot_components(forecast)
            st.write('---')
            st.header('**Forecast**')
            st.pyplot(fig1)
            st.write('---')
            st.header('**Forecast Components**')
            st.write(fig2)

            forecasting_params = {
                'y': y, 
                'ds': ds,
                'df': df,
                'm': m,
                'freq': freq,
                'period': period,
                'seasonality': seasonality,
                'changepoint_prior_scale': changepoint_prior_scale,
                'changepoint_range': changepoint_range
            }
            def forecasting_parameters(forecasting_params):
                for key, value in forecasting_params.items():
                    st.session_state[key] = value
            forecasting_parameters(forecasting_params)
            
########################################################################################################################
######################################### CROSS VALIDATION SECTION ########################################################

    st.subheader("Cross-validation Parameters")
    with st.container():
        cv_col1, cv_col2, cv_col3 = st.columns(3)
        with cv_col1:
            initial = st.number_input("Initial Period", value=4,
                            min_value=2, max_value=200)
        with cv_col2:
            period = st.number_input("Number of periods for training",
                            value=4, min_value=2, max_value=200)
        with cv_col3:
            horizon = st.number_input("Number of periods for validation",
                            value=4, min_value=2, max_value=200)

        if st.button("Run Cross Validation"):
            "st.session_state object:", st.session_state
            with st.spinner('Running Cross Validation...'):
                def mean_absolute_percentage_error(y_true, y_pred):
                    '''
                    Calculates the mean absolute percentage error between two arrays'''
                    y_true, y_pred = np.array(y_true), np.array(y_pred)
                    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

                m = Prophet(seasonality_mode=st.session_state.seasonality, changepoint_prior_scale=st.session_state.changepoint_prior_scale, changepoint_range=st.session_state.changepoint_range)
                m.fit(st.session_state.df)
                cv_results = cross_validation(
                    m, initial=(str(initial) + ' ' + freq), period=(str(period) +' ' +freq), horizon=(str(horizon) + ' ' + freq))
                mape_baseline = mean_absolute_percentage_error(
                cv_results.y, cv_results.yhat)
                st.metric('MAPE', mape_baseline)
                df_p = performance_metrics(cv_results)
                st.dataframe(df_p)
                mape_fig = plot_cross_validation_metric(cv_results, metric='mape')
                mse_fig = plot_cross_validation_metric(cv_results, metric='mse')
                st.pyplot(mape_fig)
                st.pyplot(mse_fig)
                st.write(cv_results)
                st.write('---')

