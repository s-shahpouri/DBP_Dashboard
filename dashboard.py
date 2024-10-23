import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from vis import read_from_pickle, plot_interactive_boxplots_with_outliers, find_outliers, making_array, plot_img, plot_colored_wordcloud
import pandas as pd
import numpy as np
from vis import DataProcessor, display_coordination_info,plot_ensemble, plot_single, plot_side_by_side, load_css
import matplotlib.pyplot as plt
import pickle
import os
from wordcloud import WordCloud
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
import matplotlib as mpl
from assess import OutlierDetector, PathName, folder_approach, DoseGenerator

from vis import calculate_ensemble_average, plot_ensemble_average_boxplots, plot_overview, plot_images_in_grid
from back_dash import BackDash
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np

import plotly.graph_objects as go
import streamlit as st
import numpy as np




# Set page config
st.set_page_config(
    layout='wide', initial_sidebar_state='expanded',
    page_title="DPB Dashboard",
    page_icon="logo.png"
)

# Main layout
st.markdown("<div class='centered-title' style='color:#04028b;'>DBP Dashboard</div>", unsafe_allow_html=True)
# st.markdown('---')  # This adds a horizontal line

# Call the function to load the CSS
load_css("/data/sama/DBP_Dashboard/style.css")

ensemble_data_path = 'df_best_esms.pkl'
single_data_path = 'df_single.pkl'
user_data_path = 'df_user.pkl'
current_dir = '/data/bahrdoh/Deep_Learning_Pipeline_Test/experiments_dose/test'

# Define the paths to the pickle files
pickle_files = {
    "Ensemble": ensemble_data_path,
    "Single": single_data_path,
    "User": user_data_path
}

df_filtered_tagged_losses = None  # Initialize df_filtered_tagged_losses as None

# Add a radio button to the sidebar for selecting the pickle file
with st.sidebar:
    st.sidebar.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
    main_root = st.text_input("Current Directory ↘️ ", current_dir)
    selected_approach = st.radio("Evaluation Approaches", ["Ensemble", "Single", "User"])



    if selected_approach == "Ensemble":
        back_dash = BackDash(main_root)
        df_cleaned = back_dash.process()
        df_best_esms = back_dash.process_ensemble(df_cleaned)
        back_dash.save_pickle(df_best_esms, ensemble_data_path)
        main_df = pd.read_pickle(pickle_files[selected_approach])
        # df_filtered_tagged_losses = main_df
        df_filtered_tagged_losses = main_df.iloc[:-1]
        df_avg = main_df.iloc[-1]
        # print("@@@@@")
        # print(main_df.iloc[-1])


    elif selected_approach == "Single":
        back_dash = BackDash(main_root)
        df_cleaned = back_dash.process()
        df_single = back_dash.process_single(df_cleaned)
        back_dash.save_pickle(df_single, single_data_path)
        main_df = pd.read_pickle(pickle_files[selected_approach])
        df_filtered_tagged_losses = main_df.iloc[:-1]

    elif selected_approach == "User":
        folder_path = st.text_input("Enter the path of the folder:")
        if folder_path and os.path.exists(folder_path):

            back_dash = BackDash(main_root)
            df_cleaned = back_dash.process()
            df_user = back_dash.process_user(folder_path)
            back_dash.save_pickle(df_user, user_data_path)
            main_df = pd.read_pickle(pickle_files[selected_approach])
            df_filtered_tagged_losses = main_df
            st.success("Processed and saved the user folder.")
        else:
            st.error("Enter a valid folder path!")



    st.divider()
    
    try:

        available_roots = df_filtered_tagged_losses['tag'].tolist()
        selected_tag = st.sidebar.selectbox("Select DL Model", options=available_roots)

        # Get the corresponding row for the selected tag
        selected_row = df_filtered_tagged_losses[df_filtered_tagged_losses['tag'] == selected_tag].iloc[0]

        # Dropdown for selecting dataset mode (Test, Val, or Train)
        mode = st.sidebar.selectbox("Select Mode", options=['test', 'val', 'train'])

        df = read_from_pickle(selected_row[f'{mode}_output'])
        data_dict = selected_row[f'{mode}_dict']


        # Initialize the DataProcessor and PathName
        data_processor = DataProcessor(data_dict)
        json_df = data_processor.get_dataframe()

        # After initializing df and json_df
        # Initialize the PathFinder with df and json_df
        pathname = PathName(df, json_df)

        # Process the DataFrame to find paths and append moving path extraction
        df = pathname.process()

        # Now initialize the OutlierDetector with df and json_df
        outlier_detector = OutlierDetector(df, json_df)  # Instantiate the class

        # Dropdown for difference mode selection
        mode_dict = {
            '0_dis': 'X',
            '1_dis': 'Y',
            '2_dis': 'Z',
            'Euc_dis': 'R',
            'L1_dis': 'L1',
            'L2_dis': 'L2'
        }

        # Extracting the keys from the dictionary for the dropdown options
        mode_options = list(mode_dict.keys())


        # Sidebar to select mode
        selected_mode_key = st.sidebar.selectbox("Select Axes", options=mode_options, format_func=lambda x: mode_dict[x])

        # Process the outliers based on the selected mode key
        outlier_detector.process(selected_mode_key)

        # Filter outliers based on mode
        filtered_outliers_df = outlier_detector.filter_outliers_by_mode(selected_mode_key)
        
        # Get the options from the new 'PatientID_with_diffs'
        outlier_options = filtered_outliers_df['PatientID_with_diffs'].unique()

        # Sidebar to select the specific outlier with a unique key
        selected_outlier_id = st.sidebar.selectbox("Select Outlier", options=outlier_options, key=f"outlier_{selected_mode_key}")

        # Displaying the coordination information
        st.markdown("<h3 style='text-align: center;'>Info of selected outlier</h3>", unsafe_allow_html=True)


        filtered_data = filtered_outliers_df[filtered_outliers_df['PatientID_with_diffs'] == selected_outlier_id]


        if not filtered_data.empty:
            outlier = filtered_data.iloc[0].to_dict()

            dose_generator = DoseGenerator(filtered_data.iloc[0].to_dict())
            updated_data_row = dose_generator.process()

            coord_html = display_coordination_info(outlier)
            st.markdown(coord_html, unsafe_allow_html=True)
        else:
            st.write("No outlier is selected to compare.")

        # compare_clicked = st.sidebar.button('Compare Images')

        st.markdown("")
        st.sidebar.caption('''
                Created by <a href="https://www.linkedin.com/in/zohreh-shahpouri/" target="_blank">Sama Shahpouri</a><br>at 
                <a href="https://umcgprotonentherapiecentrum.nl/" target="_blank">UMCG Protonentherapiecentrum</a>.        ''', unsafe_allow_html=True)
    except:
        pass

# Create two tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview |", "Plots |", "Images |", "Statistics |"])


with tab1:

    try:
        df_overview = df_cleaned.copy()
        df_overview['tr'] = df_overview['tr'].astype(int)
        df_overview = df_overview.reset_index(drop = True)
        # st.table(df_overview)

        col1, col2, col3 = st.columns([1, 8, 1])

        with col2:

            styled_df = plot_overview(df_overview, mode, selected_approach)

        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
    except:
        pass


with tab2:
    if selected_approach == "Ensemble":
        # df_ensembled = read_from_pickle(df_avg[f'{mode}_output'])

        df_ensembled = read_from_pickle(df_avg[f'test_output'])

        # Create columns for the radio buttons to arrange them horizontally
        col1, col2, col3, col4, col5, col6, col7 , col8 = st.columns(8)

        # Create a placeholder for the selected option
        plot_option = None

        # Assign each button to a separate column and plot in the same column
        with col1:

            if st.button("ΔX"):
                plot_option = "ΔX"
                selected_column = '0_dis'
                fig = plot_side_by_side(df, df_ensembled, selected_column, plot_option, "Avg")
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if st.button("ΔY"):
                plot_option = "ΔY"
                selected_column = '1_dis'
                fig = plot_side_by_side(df, df_ensembled, selected_column, plot_option, "Avg")
                st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            if st.button("ΔZ"):
                plot_option = "ΔZ"
                selected_column = '2_dis'
                fig = plot_side_by_side(df, df_ensembled, selected_column, plot_option, "Avg")
                st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            if st.button("ΔR"):
                plot_option = "ΔR"
                selected_column = 'Euc_dis'
                fig = plot_side_by_side(df, df_ensembled, selected_column, plot_option, "Avg")
                st.plotly_chart(fig, use_container_width=True)
        
        with col5:
            if st.button("L1"):
                plot_option = "L1"
                selected_column = 'L1_dis'
                fig = plot_side_by_side(df, df_ensembled, selected_column, plot_option, "Avg")
                st.plotly_chart(fig, use_container_width=True)
        
        with col6:
            if st.button("L2"):
                plot_option = "L2"
                selected_column = 'L2_dis'
                fig = plot_side_by_side(df, df_ensembled, selected_column, plot_option, "Avg")
                st.plotly_chart(fig, use_container_width=True)
        
        with col7:
            if st.button("All Plots"):
                plot_option = "All Plots"

        # If "All Plots" is selected, display the full ensemble plot below the buttons
        if plot_option == "All Plots":
            # Display the full ensemble plot across the entire width of the screen
            fig = plot_interactive_boxplots_with_outliers(df)
            st.plotly_chart(fig, use_container_width=True)

        with col8:
            if st.button("AVG Plots"):
                plot_option = "AVG Plots"

        # If "All Plots" is selected, display the full ensemble plot below the buttons
        if plot_option == "AVG Plots":
            # Display the full ensemble plot across the entire width of the screen
            fig = plot_ensemble(df_ensembled)
            st.plotly_chart(fig, use_container_width=True)

    else:
        try:
            fig = plot_interactive_boxplots_with_outliers(df)
            st.plotly_chart(fig, use_container_width=True)
            st.divider()
        except:
            pass

with tab3:

        # Use columns with specific ratios for better spacing
    col1, col2, col3 = st.columns([1,0.1, 4])
    try:

        with col1:
            selected_image_type = st.radio("Select Image Type", ["CT","Dose"])
            st.divider()
            available_colormaps = ['viridis', 'gray', 'binary', 'jet', ]
            selected_colormap = st.selectbox("Select Color Palette", available_colormaps)


            data = making_array(updated_data_row, selected_image_type)         
            
            if selected_image_type == "CT":
                scaling_factor = 100.0  # Scale for CT iomages
                filtered_data['fixed'] = filtered_data['fixed'].replace('/nrrd/', '/ct_nrrd/')
                filtered_data['moving'] = filtered_data['moving'].replace('/nrrd/', '/ct_nrrd/')


            elif selected_image_type == "Dose":
                scaling_factor = 1000.0  # Scale for Dose images
                filtered_data['fixed'] = filtered_data['fixed'].replace('/ct_nrrd/', '/nrrd/')
                filtered_data['moving'] = filtered_data['moving'].replace('/ct_nrrd/', '/nrrd/')




            fixed_array, _, _, _, _, _ = data
            max_slice_index  = fixed_array.shape[0] - 1
            st.divider()
            slice_index = st.slider("Select Slice Index", min_value=0, max_value=max_slice_index, value=max_slice_index // 2)

        with col3:
            # fig = plot_img(data, slice_index, colormap=selected_colormap, scaling_factor=scaling_factor)
            # st.pyplot(fig)
            plot_images_in_grid(data, slice_index, selected_colormap, scaling_factor)
            # # Call the function
            # fig = plot_img_plotly(data, slice_index, selected_colormap, scaling_factor)
            # st.plotly_chart(fig)  # Use st.plotly_chart to display Plotly figures

            # plot_images_in_grid(data, slice_index, selected_colormap, scaling_factor)

    except:
        st.markdown("<p class='centered-lowered-text'>No data available to show.<br>Select an outlier!</p>", unsafe_allow_html=True)



with tab4:
    try:
        st.markdown("Descriptive Statistics")
        desc_stats = df[['0_dis', '1_dis', '2_dis', 'Euc_dis', 'L1_dis', 'L2_dis']].describe().T
        st.dataframe(desc_stats)

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


        st.divider()
        st.subheader("Correlation Matrix")

        # Compute the correlation matrix
        corr_matrix = df[['0_dis', '1_dis', '2_dis', 'Euc_dis', 'L1_dis', 'L2_dis']].corr()

        # Apply a colormap to the DataFrame
        corr_styled = corr_matrix.style.background_gradient(cmap='coolwarm', axis=None).format(precision=2)

        # Display the styled correlation matrix in Streamlit
        st.dataframe(corr_styled)

    except:
        pass



