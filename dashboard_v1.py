import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from vis import read_df, plot_interactive_boxplots_with_outliers, find_outliers, making_array, plot_img
import pandas as pd
import numpy as np
from vis import JsonProcessor, find_paths, extract_name, display_coordination_info, load_css
import matplotlib.pyplot as plt
import pickle

# Set page config
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# Main layout
st.markdown("<div class='centered-title' style='color:#04028b;'>DBP Dashboard</div>", unsafe_allow_html=True)
st.markdown('---')  # This adds a horizontal line

# Call the function to load the CSS
load_css("style.css")

# Load the saved DataFrame
df_filtered_tagged_losses = pd.read_pickle('selected_folders.pkl')

# List of available root folders based on 'tag' column
available_roots = df_filtered_tagged_losses['tag'].tolist()

main_root = '/data/bahrdoh/Deep_Learning_Pipeline_Dose/experiments_dose/test/'

with st.sidebar:
    # st.sidebar.empty()
    st.sidebar.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='bold' style='font-size:24px; color:#04028b;'>Dashboard Controls</div>", unsafe_allow_html=True)
    st.divider()
    # Dropdown for root folder selection
    selected_tag = st.sidebar.selectbox("Select DL Model", options=available_roots)
    selected_folder = df_filtered_tagged_losses[df_filtered_tagged_losses['tag'] == selected_tag]['Folder'].values[0]
    selected_root = main_root + selected_folder
    test_path = selected_root + '/Dual_DCNN_LReLu_test_outputs.csv'
    val_path = selected_root + '/Dual_DCNN_LReLu_val_outputs.csv'
    train_path = selected_root + '/Dual_DCNN_LReLu_train_outputs.csv'

    # Dropdown for dataset selection
    dataset_option = st.sidebar.selectbox("Select Mode", options=['Test', 'Val', 'Train'])

    # Load DataFrames
    df_test = read_df(test_path)
    df_train = read_df(train_path)
    df_val = read_df(val_path)

    # Load the selected DataFrame
    if dataset_option == 'Train':
        df = df_train
    elif dataset_option == 'Val':
        df = df_val
    else:
        df = df_test

    # Usage example:
    json_processor = JsonProcessor(json_path=selected_root + '/Data_dict_0.json')
    json_df = json_processor.get_dataframe()

    # Finding matched paths using the function
    df = find_paths(df, json_df)
    df['PatientID'] = df.apply(lambda row: f"{row['PatientID']}_{extract_name(row['moving'])}", axis=1)

    # Find outliers for each difference and Euclidean distance
    outliers_diff_0 = find_outliers(df, 'diff_0')
    outliers_diff_1 = find_outliers(df, 'diff_1')
    outliers_diff_2 = find_outliers(df, 'diff_2')
    outliers_euclidean_dist = find_outliers(df, 'euclidean_dist')

    # Add a new column to each DataFrame indicating the type of difference
    outliers_diff_0['type'] = '0_dist'
    outliers_diff_1['type'] = '1_dist'
    outliers_diff_2['type'] = '2_dist'
    outliers_euclidean_dist['type'] = 'euclidean_dist'

    # Concatenate all DataFrames into one unique DataFrame
    outliers_df = pd.concat([outliers_diff_0, outliers_diff_1, outliers_diff_2, outliers_euclidean_dist])

    # Reset the index for the combined DataFrame
    outliers_df.reset_index(drop=True, inplace=True)

    # Dropdown for difference mode selection
    mode_dict = {
        '0_dist': 'X Axis Distance',
        '1_dist': 'Y Axis Distance',
        '2_dist': 'Z Axis Distance',
        'euclidean_dist': 'Euclidean Distance'
    }

    # Extracting the keys from the dictionary for the dropdown options
    mode_options = list(mode_dict.keys())

    # Dropdown for difference mode selection with user-friendly names
    selected_mode_key = st.sidebar.selectbox("Select Difference in Axes", options=mode_options, format_func=lambda x: mode_dict[x])

    # Use the selected_mode_key for further processing
    filtered_outliers_df = outliers_df[outliers_df['type'] == selected_mode_key]

    # Filter outliers based on the selected dataset and mode
    filtered_outliers_df = outliers_df[(outliers_df['type'] == selected_mode_key)]
    outlier_options = filtered_outliers_df['PatientID'].unique()
    selected_outlier_id = st.sidebar.selectbox("Select Outlier", options=outlier_options)

    # Displaying the coordination information
    st.divider()
    st.markdown("<h2 style='text-align: center;'>Coordination Info</h2>", unsafe_allow_html=True)

    # Filter based on patient_id and mode
    filtered_data = filtered_outliers_df[(filtered_outliers_df['PatientID'] == selected_outlier_id)]

    if not filtered_data.empty:
        outlier = filtered_data.iloc[0].to_dict()
        coord_html = display_coordination_info(outlier)
        st.markdown(coord_html, unsafe_allow_html=True)
    else:
        st.write("No outlier is selected to compare.")

    compare_clicked = st.sidebar.button('Compare Images')

    st.sidebar.markdown('''
    ---
    <div class="sidebar-footer">
    Created by <a href="https://www.linkedin.com/in/zohreh-shahpouri/" target="_blank">Sama Shahpouri</a><br>at 
    <a href="https://umcgprotonentherapiecentrum.nl/" target="_blank">UMCG Protonentherapiecentrum</a>.
    </div>
    ''', unsafe_allow_html=True)

# Use columns with specific ratios for better spacing
col1, col2 = st.columns([2, 2])

with col1:
    st.markdown("<h1 style='font-size:24px; text-align: center; '>Distances within true and predicted coordinations</h1>", unsafe_allow_html=True)
    fig = plot_interactive_boxplots_with_outliers(df)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("<h1 style='font-size:24px; text-align: center;'>Dose images</h1>", unsafe_allow_html=True)
    if compare_clicked and not filtered_data.empty:
        data = making_array(filtered_data.iloc[0].to_dict())
        fig = plot_img(data)
        st.pyplot(fig)
    else:
        st.markdown("<p class='centered-lowered-text'>No data available to show.<br>Click on 'Compare' for displaying!</p>", unsafe_allow_html=True)
