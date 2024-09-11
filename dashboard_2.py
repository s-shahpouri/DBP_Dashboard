import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from vis import read_df, plot_interactive_boxplots_with_outliers, find_outliers, making_array, plot_img, plot_colored_wordcloud
import pandas as pd
import numpy as np
from vis import JsonProcessor, find_paths, extract_name, display_coordination_info, load_css
import matplotlib.pyplot as plt
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


# Set page config
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# Main layout
st.markdown("<div class='centered-title' style='color:#04028b;'>DBP Dashboard</div>", unsafe_allow_html=True)
# st.markdown('---')  # This adds a horizontal line

# Call the function to load the CSS
load_css("style.css")


df_filtered_tagged_losses = pd.read_pickle('selected_folders_3.pkl')
main_root = '/data/bahrdoh/Deep_Learning_Pipeline_Test/experiments_dose/test/'


# df_filtered_tagged_losses = pd.read_pickle('selected_folders.pkl')
# main_root = '/data/bahrdoh/Deep_Learning_Pipeline_Dose/experiments_dose/test/'

available_roots = df_filtered_tagged_losses['tag'].tolist()





with st.sidebar:
    st.sidebar.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='bold' style='font-size:24px; color:#04028b;'>Dashboard Controls</div>", unsafe_allow_html=True)
    st.divider()
    
    # Dropdown for root folder selection
    selected_tag = st.sidebar.selectbox("Select DL Model", options=available_roots)
    selected_folder = df_filtered_tagged_losses[df_filtered_tagged_losses['tag'] == selected_tag]['Folder'].values[0]
    # st.write(f"Selected Folder: {selected_folder}")
    
    selected_root = main_root + selected_folder
    selected_root = main_root + selected_folder
    # if not os.path.exists(selected_root):
    #     st.write(f"Folder does not exist: {selected_root}")
    # else:
    #     st.write(f"Folder exists: {selected_root}")



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
    outliers_0_dis = find_outliers(df, '0_dis')
    outliers_1_dis = find_outliers(df, '1_dis')
    outliers_2_dis = find_outliers(df, '2_dis')
    outliers_euclidean_dist = find_outliers(df, 'Euc_dis')
    outliers_L1_dis = find_outliers(df, 'L1_dis')
    outliers_L2_dis = find_outliers(df, 'L2_dis')

    # Add a new column to each DataFrame indicating the type of difference
    outliers_0_dis['type'] = '0_dis'
    outliers_1_dis['type'] = '1_dis'
    outliers_2_dis['type'] = '2_dis'
    outliers_euclidean_dist['type'] = 'Euc_dis'
    outliers_L1_dis['type'] = 'L1_dis'
    outliers_L2_dis['type'] = 'L2_dis'

    # Concatenate all DataFrames into one unique DataFrame
    outliers_df = pd.concat([outliers_0_dis, outliers_1_dis, outliers_2_dis, outliers_euclidean_dist, outliers_L1_dis, outliers_L2_dis])

    # Reset the index for the combined DataFrame
    outliers_df.reset_index(drop=True, inplace=True)

    # Dropdown for difference mode selection
    mode_dict = {
        '0_dis': 'X Axis Distance',
        '1_dis': 'Y Axis Distance',
        '2_dis': 'Z Axis Distance',
        'Euc_dis': 'Euclidean Distance',
        'L1_dis': 'L1 Distance',
        'L2_dis': 'L2 Distance'
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
    st.divider()
    selected_outlier_id = st.sidebar.selectbox("Select Outlier", options=outlier_options)

    # Displaying the coordination information
    st.markdown("<h3 style='text-align: center;'>Info of selected point</h3>", unsafe_allow_html=True)

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

# Create two tabs
tab1, tab2, tab3 = st.tabs(["Overview", "Details", "Statistics"])

with tab1:
    # st.markdown("<h2 style='text-align: center;'>Overview</h2>", unsafe_allow_html=True)

    # Define the radio button options
    axis_options = {
        'X Axis': '0_dis',
        'Y Axis': '1_dis',
        'Z Axis': '2_dis',
        'Euclidean': 'Euc_dis',
        'L1 Distance': 'L1_dis',
        'L2 Distance': 'L2_dis'
    }
  
    # Add the radio button for axis selection
    selected_axis = st.radio("Select Axis:", list(axis_options.keys()))




    # Calculate the Mean Absolute Difference for each patient across coordinates
    mean_abs_0_dis = df.groupby('PatientID')['0_dis'].apply(lambda x: abs(x).mean()).to_dict()
    mean_abs_1_dis = df.groupby('PatientID')['1_dis'].apply(lambda x: abs(x).mean()).to_dict()
    mean_abs_2_dis = df.groupby('PatientID')['2_dis'].apply(lambda x: abs(x).mean()).to_dict()
    mean_abs_diff_euclidean = df.groupby('PatientID')['Euc_dis'].apply(lambda x: abs(x).mean()).to_dict()
    mean_abs_diff_L1 = df.groupby('PatientID')['L1_dis'].apply(lambda x: abs(x).mean()).to_dict()
    mean_abs_diff_L2 = df.groupby('PatientID')['L2_dis'].apply(lambda x: abs(x).mean()).to_dict()

    # Calculate the Maximum for each patient across coordinates
    max_0_dis = df.groupby('PatientID')['0_dis'].apply(lambda x: max(x)).to_dict()
    max_1_dis = df.groupby('PatientID')['1_dis'].apply(lambda x: max(x)).to_dict()
    max_2_dis = df.groupby('PatientID')['2_dis'].apply(lambda x: max(x)).to_dict()
    max_diff_euclidean = df.groupby('PatientID')['Euc_dis'].apply(lambda x: max(x)).to_dict()
    max_diff_L1 = df.groupby('PatientID')['L1_dis'].apply(lambda x: max(x)).to_dict()
    max_diff_L2 = df.groupby('PatientID')['L2_dis'].apply(lambda x: max(x)).to_dict()

    # Mapping from axis to data dictionaries
    mean_diff_map = {
        '0_dis': mean_abs_0_dis,
        '1_dis': mean_abs_1_dis,
        '2_dis': mean_abs_2_dis,
        'Euc_dis': mean_abs_diff_euclidean,
        'L1_dis': mean_abs_diff_L1,
        'L2_dis': mean_abs_diff_L2
    }

    max_diff_map = {
        '0_dis': max_0_dis,
        '1_dis': max_1_dis,
        '2_dis': max_2_dis,
        'Euc_dis': max_diff_euclidean,
        'L1_dis': max_diff_L1,
        'L2_dis': max_diff_L2
    }

    # Plot the selected word clouds side by side
    col1, col2 = st.columns(2)
    with col1:
        plot_colored_wordcloud(mean_diff_map[axis_options[selected_axis]], f'Mean Absolute Differences ({selected_axis})')
    with col2:
        plot_colored_wordcloud(max_diff_map[axis_options[selected_axis]], f'Max Differences ({selected_axis})')

with tab2:
    # Use columns with specific ratios for better spacing
    col1, col2 = st.columns([4, 2])

    with col1:
        st.markdown("<h3 style='font-size:14x; text-align: center;'>Distances within true and predicted coordinations</h3>", unsafe_allow_html=True)
        fig = plot_interactive_boxplots_with_outliers(df)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<h3 style='font-size:20px; text-align: center;'>Dose images</h3>", unsafe_allow_html=True)
        if compare_clicked and not filtered_data.empty:
            data = making_array(filtered_data.iloc[0].to_dict())
            fig = plot_img(data)
            st.pyplot(fig)
        else:
            st.markdown("<p class='centered-lowered-text'>No data available to show.<br>Click on 'Compare' for displaying!</p>", unsafe_allow_html=True)


with tab3:
    st.markdown("Descriptive Statistics")
    desc_stats = df[['0_dis', '1_dis', '2_dis', 'Euc_dis', 'L1_dis', 'L2_dis']].describe().T
    st.dataframe(desc_stats)


###########################################################################################################

    st.divider()
    st.markdown("""
        ### Z-Score Analysis Table

        The Z-Score analysis for each of the distance metrics.
        
        - **The Z-Score** is a statistical measure that indicates how far each measurement deviates from the average value across all data points.

        - **Filtering Outliers**: Any data points with a Z-score greater than 3 or less than -3 are considered statistical outliers.
        
        - **Z-Score Formula**: 
    """)

    # Display the Z-score formula using st.latex
    st.latex(r'''Z = \frac{X - \mu}{\sigma}''')

    st.markdown("""
        Where:
        - \( X \) is the value in the dataset. mu is the mean of the dataset. sigma is the standard deviation of the dataset.
    """)

    # Calculate Z-Scores for each distance metric
    for col in ['0_dis', '1_dis', '2_dis', 'Euc_dis', 'L1_dis', 'L2_dis']:
        df[f'z_{col}'] = (df[col] - df[col].mean()) / df[col].std()

    # Filter outliers where any Z-score exceeds 3
    outliers = df[(abs(df['z_0_dis']) > 3) | 
                (abs(df['z_1_dis']) > 3) | 
                (abs(df['z_2_dis']) > 3) | 
                (abs(df['z_Euc_dis']) > 3) | 
                (abs(df['z_L1_dis']) > 3) | 
                (abs(df['z_L2_dis']) > 3)]

    # Reorder columns: Z-score columns first, then other relevant columns
    columns_order = [f'z_{col}' for col in ['0_dis', '1_dis', '2_dis', 'Euc_dis', 'L1_dis', 'L2_dis']] + \
                    [col for col in df.columns if col not in [f'z_{col}' for col in ['0_dis', '1_dis', '2_dis', 'Euc_dis', 'L1_dis', 'L2_dis']] + ['fixed', 'PatientID', 'pred_0', 'pred_1', 'pred_2', 'true_0', 'true_1', 'true_2']]

    outliers = outliers[columns_order]

    # Apply a function to color the outlier cells
    def highlight_outliers(val):
        color = 'orange' if abs(val) > 3 else ''
        return f'background-color: {color}'

    # Apply the style to the filtered outliers DataFrame
    styled_outliers = outliers.style.applymap(highlight_outliers, subset=[f'z_{col}' for col in ['0_dis', '1_dis', '2_dis', 'Euc_dis', 'L1_dis', 'L2_dis']])

    # Display the styled DataFrame as HTML
    st.write(styled_outliers.to_html(), unsafe_allow_html=True)




################################################################################################################

    st.divider()
    st.subheader("Correlation Matrix")

    # Compute the correlation matrix
    corr_matrix = df[['0_dis', '1_dis', '2_dis', 'Euc_dis', 'L1_dis', 'L2_dis']].corr()

    # Apply a colormap to the DataFrame
    corr_styled = corr_matrix.style.background_gradient(cmap='coolwarm', axis=None).format(precision=2)

    # Display the styled correlation matrix in Streamlit
    st.dataframe(corr_styled)

################################################################################################################

