import numpy as np
import streamlit as st
import itk
import pandas as pd
import torch
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import json
import re
from monai.metrics import MAEMetric
from io import StringIO
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



def read_from_pickle(data):
    """Process data directly from pickle or string format."""
    # If the data is in a string format, convert it into a DataFrame by splitting on the semicolon
    if isinstance(data, str):
        data = pd.read_csv(StringIO(data), delimiter=';')
    
    # Ensure that the data is properly split into columns
    if 'PatientID' not in data.columns:
        # Assume that the first column contains 'PatientID;pred_0;pred_1;pred_2;true_0;true_1;true_2;Mode' combined into a single column
        data = data.iloc[:, 0].str.split(';', expand=True)
        data.columns = ['PatientID', 'pred_0', 'pred_1', 'pred_2', 'true_0', 'true_1', 'true_2', 'Mode']
    
    # Drop the 'Mode' column if it exists
    data = data.drop(columns=['Mode'], errors='ignore')

    # Round numeric columns
    numeric_cols = ['pred_0', 'pred_1', 'pred_2', 'true_0', 'true_1', 'true_2']
    if set(numeric_cols).issubset(data.columns):
        data[numeric_cols] = data[numeric_cols].astype(float).round(4)

    # Remove "DBP_" from the 'PatientID' column
    if 'PatientID' in data.columns:
        data['PatientID'] = data['PatientID'].str.replace('DBP_', '')

    # Calculate the distances
    data['0_dis'] = data['pred_0'] - data['true_0']
    data['1_dis'] = data['pred_1'] - data['true_1']
    data['2_dis'] = data['pred_2'] - data['true_2']
    data['Euc_dis'] = np.sqrt(data['0_dis']**2 + data['1_dis']**2 + data['2_dis']**2)
    
    # Assuming 'data' is already loaded and contains the necessary columns
    pred_tensors = torch.tensor(data[['pred_0', 'pred_1', 'pred_2']].values, dtype=torch.float32)
    true_tensors = torch.tensor(data[['true_0', 'true_1', 'true_2']].values, dtype=torch.float32)

    # Initialize L1 Loss without reduction
    l1_loss = torch.nn.L1Loss(reduction="none")  # No reduction
    l1_distances = l1_loss(pred_tensors, true_tensors)  # This will give element-wise differences
    l1_rowwise_distances = l1_distances.mean(dim=1).numpy()  # Calculate the average across dimensions, or use another strategy
    data['L1_dis'] = l1_rowwise_distances

    # Initialize MSE Loss without reduction
    mse_loss = torch.nn.MSELoss(reduction='none')  # No reduction
    mse_elementwise_distances = mse_loss(pred_tensors, true_tensors)  # Element-wise squared differences
    mse_rowwise_distances = mse_elementwise_distances.mean(dim=1).numpy()  # Calculate the average across dimensions, or use another strategy
    data['L2_dis'] = mse_rowwise_distances

    return data



class JsonProcessor:
    def __init__(self, json_path):
        self.json_path = json_path
        self.data_dict = self.load_json()
        self.json_df = self.extract_data()

    def load_json(self):
        """Load the JSON data from the file."""
        with open(self.json_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def clean_position_string(pos_str):
        """Clean and convert position string to a list of floats."""
        pos_str = re.sub(r'\s+', ',', pos_str.strip('[] '))  # Replace spaces with commas and strip brackets
        return list(map(float, pos_str.split(',')))

    def extract_data(self):
        """Extract the relevant fields from the JSON data into a DataFrame."""
        df = pd.DataFrame(self.data_dict['test_dict'])
        
        # Apply cleaning function to the 'pos' column and expand it into separate columns
        df['pos'] = df['pos'].apply(self.clean_position_string)
        df[['true_0', 'true_1', 'true_2']] = pd.DataFrame(df['pos'].tolist(), index=df.index).round(4)

        # Drop the 'pos' column as it's no longer needed
        df = df.drop(columns=['pos'])

        # Clean the 'pat_id' column by removing "DBP_"
        df['PatientID'] = df['pat_id'].str.replace('DBP_', '')

        # Drop the old 'pat_id' column as it's redundant
        df = df.drop(columns=['pat_id'])

        return df

    def get_dataframe(self):
        """Return the processed DataFrame."""
        return self.json_df
    


class DataProcessor:
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.json_df = self.extract_data()

    @staticmethod
    def clean_position_string(pos_str):
        """Clean and convert position string to a list of floats."""
        pos_str = re.sub(r'\s+', ',', pos_str.strip('[] '))  # Replace spaces with commas and strip brackets
        return list(map(float, pos_str.split(',')))

    def extract_data(self):
        """Extract the relevant fields from the data dictionary into a DataFrame."""
        df = pd.DataFrame(self.data_dict)  # Convert the dictionary into a DataFrame

        # Apply cleaning function to the 'pos' column and expand it into separate columns
        if 'pos' in df.columns:
            df['pos'] = df['pos'].apply(self.clean_position_string)
            df[['true_0', 'true_1', 'true_2']] = pd.DataFrame(df['pos'].tolist(), index=df.index).round(4)
            df = df.drop(columns=['pos'])  # Drop the 'pos' column after extracting its data
        df['PatientID'] = df['pat_id'].str.replace('DBP_', '')


        df = df.drop(columns=['pat_id'])

        return df

    def get_dataframe(self):
        """Return the processed DataFrame."""
        return self.json_df



# Define a function to find outliers based on the IQR method and return both values and indices
def find_outliers(data, column_name):
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column_name] < lower_bound) | (data[column_name] > upper_bound)]
    return outliers



@ st.cache_data
def plot_interactive_boxplots_with_outliers(df_test):

    N = 22
    
    # Generate an array of rainbow colors
    colors = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]

    # Create subplots
    fig = make_subplots(rows=1, cols=6, subplot_titles=[
        'ΔX', 'ΔY', 'ΔZ', 'ΔR', 'L1', 'L2'
    ])

    # Add strip plots (scatter plots) for all points with manual jitter
    jitter_amount = 0.1
    fig.add_trace(go.Scatter(
        x=np.random.normal(1, jitter_amount, size=len(df_test['0_dis'])),
        y=df_test['0_dis'], mode='markers', name='All Points (0_dis)', 
        marker=dict(color=colors[0], size=5, opacity=0.5)), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=np.random.normal(2, jitter_amount, size=len(df_test['1_dis'])),
        y=df_test['1_dis'], mode='markers', name='All Points (1_dis)', 
        marker=dict(color=colors[4], size=5, opacity=0.5)), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=np.random.normal(3, jitter_amount, size=len(df_test['2_dis'])),
        y=df_test['2_dis'], mode='markers', name='All Points (2_dis)', 
        marker=dict(color=colors[8], size=5, opacity=0.5)), row=1, col=3)
    fig.add_trace(go.Scatter(
        x=np.random.normal(4, jitter_amount, size=len(df_test['Euc_dis'])),
        y=df_test['Euc_dis'], mode='markers', name='All Points (Euc_dis)', 
        marker=dict(color=colors[12], size=5, opacity=0.5)), row=1, col=4)
    fig.add_trace(go.Scatter(
        x=np.random.normal(5, jitter_amount, size=len(df_test['L1_dis'])),
        y=df_test['L1_dis'], mode='markers', name='All Points (L1_dis)', 
        marker=dict(color=colors[16], size=5, opacity=0.5)), row=1, col=5)
    fig.add_trace(go.Scatter(
        x=np.random.normal(6, jitter_amount, size=len(df_test['L2_dis'])),
        y=df_test['L2_dis'], mode='markers', name='All Points (L2_dis)', 
        marker=dict(color=colors[18], size=5, opacity=0.5)), row=1, col=6)

    # Add box plots for suspected outliers using different colors
    fig.add_trace(go.Box(y=df_test['0_dis'], name='Box (0_dis)', boxpoints='suspectedoutliers',
                         marker=dict(color=colors[0]), boxmean=True), row=1, col=1)
    fig.add_trace(go.Box(y=df_test['1_dis'], name='Box (1_dis)', boxpoints='suspectedoutliers',
                         marker=dict(color=colors[4]), boxmean=True), row=1, col=2)
    fig.add_trace(go.Box(y=df_test['2_dis'], name='Box (2_dis)', boxpoints='suspectedoutliers',
                         marker=dict(color=colors[8]), boxmean=True), row=1, col=3)
    fig.add_trace(go.Box(y=df_test['Euc_dis'], name='Box (Euc_dis)', boxpoints='suspectedoutliers',
                         marker=dict(color=colors[12]), boxmean=True), row=1, col=4)
    fig.add_trace(go.Box(y=df_test['L1_dis'], name='Box (L1_dis)', boxpoints='suspectedoutliers',
                         marker=dict(color=colors[16]), boxmean=True), row=1, col=5)
    fig.add_trace(go.Box(y=df_test['L2_dis'], name='Box (L2_dis)', boxpoints='suspectedoutliers',
                         marker=dict(color=colors[18]), boxmean=True), row=1, col=6)

    # Update layout for aesthetics
    fig.update_layout(height=600, width=1600,
                      showlegend=False)

    max_abs_value = max(abs(df_test[['0_dis', '1_dis', '2_dis', 'Euc_dis', 'L1_dis']].max().max()),
                    abs(df_test[['0_dis', '1_dis', '2_dis', 'Euc_dis', 'L1_dis']].min().min()))


    y_axis_range = [-max_abs_value * 1.0, max_abs_value * 1.3]

    # Set common y-axis range for all subplots
    fig.update_yaxes(range=y_axis_range, row=1, col=1)
    fig.update_yaxes(range=y_axis_range, row=1, col=2)
    fig.update_yaxes(range=y_axis_range, row=1, col=3)
    fig.update_yaxes(range=y_axis_range, row=1, col=4)
    fig.update_yaxes(range=y_axis_range, row=1, col=5)



    fig.update_xaxes(showticklabels=False)
    return fig


@ st.cache_data
def plot_ensemble(df_test):

    N = 22
    
    # Generate an array of rainbow colors
    colors = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]

    # Create subplots
    fig = make_subplots(rows=1, cols=6, subplot_titles=[
        'ΔX', 'ΔY', 'ΔZ', 'ΔR', 'L1', 'L2'
    ])

    # Add strip plots (scatter plots) for all points with manual jitter
    jitter_amount = 0.1
    fig.add_trace(go.Scatter(
        x=np.random.normal(1, jitter_amount, size=len(df_test['0_dis'])),
        y=df_test['0_dis'], mode='markers', name='All Points (0_dis)', 
        marker=dict(color=colors[0], size=5, opacity=0.5)), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=np.random.normal(2, jitter_amount, size=len(df_test['1_dis'])),
        y=df_test['1_dis'], mode='markers', name='All Points (1_dis)', 
        marker=dict(color=colors[4], size=5, opacity=0.5)), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=np.random.normal(3, jitter_amount, size=len(df_test['2_dis'])),
        y=df_test['2_dis'], mode='markers', name='All Points (2_dis)', 
        marker=dict(color=colors[8], size=5, opacity=0.5)), row=1, col=3)
    fig.add_trace(go.Scatter(
        x=np.random.normal(4, jitter_amount, size=len(df_test['Euc_dis'])),
        y=df_test['Euc_dis'], mode='markers', name='All Points (Euc_dis)', 
        marker=dict(color=colors[12], size=5, opacity=0.5)), row=1, col=4)
    fig.add_trace(go.Scatter(
        x=np.random.normal(5, jitter_amount, size=len(df_test['L1_dis'])),
        y=df_test['L1_dis'], mode='markers', name='All Points (L1_dis)', 
        marker=dict(color=colors[16], size=5, opacity=0.5)), row=1, col=5)
    fig.add_trace(go.Scatter(
        x=np.random.normal(6, jitter_amount, size=len(df_test['L2_dis'])),
        y=df_test['L2_dis'], mode='markers', name='All Points (L2_dis)', 
        marker=dict(color=colors[18], size=5, opacity=0.5)), row=1, col=6)

    # Add box plots for suspected outliers using different colors
    fig.add_trace(go.Box(y=df_test['0_dis'], name='Box (0_dis)', boxpoints='suspectedoutliers',
                         marker=dict(color=colors[0]), boxmean=True), row=1, col=1)
    fig.add_trace(go.Box(y=df_test['1_dis'], name='Box (1_dis)', boxpoints='suspectedoutliers',
                         marker=dict(color=colors[4]), boxmean=True), row=1, col=2)
    fig.add_trace(go.Box(y=df_test['2_dis'], name='Box (2_dis)', boxpoints='suspectedoutliers',
                         marker=dict(color=colors[8]), boxmean=True), row=1, col=3)
    fig.add_trace(go.Box(y=df_test['Euc_dis'], name='Box (Euc_dis)', boxpoints='suspectedoutliers',
                         marker=dict(color=colors[12]), boxmean=True), row=1, col=4)
    fig.add_trace(go.Box(y=df_test['L1_dis'], name='Box (L1_dis)', boxpoints='suspectedoutliers',
                         marker=dict(color=colors[16]), boxmean=True), row=1, col=5)
    fig.add_trace(go.Box(y=df_test['L2_dis'], name='Box (L2_dis)', boxpoints='suspectedoutliers',
                         marker=dict(color=colors[18]), boxmean=True), row=1, col=6)

    # Update layout for aesthetics
    fig.update_layout(height=600, width=1600,
                      showlegend=False)

    max_abs_value = max(abs(df_test[['0_dis', '1_dis', '2_dis', 'Euc_dis', 'L1_dis']].max().max()),
                    abs(df_test[['0_dis', '1_dis', '2_dis', 'Euc_dis', 'L1_dis']].min().min()))


    y_axis_range = [-max_abs_value * 1.0, max_abs_value * 1.3]
    # Set the extended y-axis range for all subplots
    fig.update_yaxes(range=y_axis_range, row=1, col=1)
    fig.update_yaxes(range=y_axis_range, row=1, col=2)
    fig.update_yaxes(range=y_axis_range, row=1, col=3)
    fig.update_yaxes(range=y_axis_range, row=1, col=4)
    fig.update_yaxes(range=y_axis_range, row=1, col=5)

    fig.update_xaxes(showticklabels=False)
    return fig


def plot_single(df_test, column_name, title):
    fig = make_subplots(rows=1, cols=1, subplot_titles=[title])
    jitter_amount = 0.1
    fig.add_trace(go.Scatter(
        x=np.random.normal(1, jitter_amount, size=len(df_test[column_name])),
        y=df_test[column_name], mode='markers', 
        marker=dict(color='blue', size=5, opacity=0.5)
    ))

    fig.add_trace(go.Box(y=df_test[column_name], boxpoints='suspectedoutliers', boxmean=True))

    fig.update_layout(height=400, width=600, showlegend=False)
    return fig


def plot_side_by_side(df1, df2, column_name, title1, title2):
    fig = make_subplots(rows=1, cols=2, subplot_titles=[title1, title2])

    # Add the first box plot from df1
    fig.add_trace(go.Box(y=df1[column_name], name=title1, boxpoints='suspectedoutliers', boxmean=True), row=1, col=1)

    # Add the second box plot from df2
    fig.add_trace(go.Box(y=df2[column_name], name=title2, boxpoints='suspectedoutliers', boxmean=True), row=1, col=2)
    fig.update_yaxes(matches='y')
    fig.update_layout(height=400, width=600, showlegend=False)
    return fig

    
def transform_moving_ct(moving_CT_image, fixed_CT_image, coordination, pixdim):
    if len(coordination) != 3:
        raise ValueError(f"Expected coordination of length 3, but got {len(coordination)}")

    # Set the spacing for the images
    fixed_CT_image.SetSpacing(pixdim)
    moving_CT_image.SetSpacing(pixdim)

    # Define the translation transformation
    translation_updated = itk.TranslationTransform[itk.D, 3].New()
    translation_updated.SetOffset(np.array(
                        [coordination[2], coordination[1], coordination[0]],dtype=np.float64))

    # Resample the moving image with the transformation
    resampler_updated = itk.ResampleImageFilter.New(
                        Input=moving_CT_image, Transform=translation_updated,
                        UseReferenceImage=True, ReferenceImage=fixed_CT_image)
    resampler_updated.SetInterpolator(itk.LinearInterpolateImageFunction.New(fixed_CT_image))

    resampler_updated.Update()
    transformed_moving_CT_image_updated = resampler_updated.GetOutput()
    transformed_moving_CT_array_updated = itk.array_view_from_image(transformed_moving_CT_image_updated).astype(int)

    return transformed_moving_CT_array_updated


def making_array(outlier):

    fixed_CT_image = itk.imread(outlier['fixed']) 
 # Load the fixed image

    moving_CT_image = itk.imread(outlier['moving'])  # Load the moving image

    # True and predicted coordinates from the outlier data
    true_coords = [outlier['true_0'], outlier['true_1'], outlier['true_2']]
    pred_coords = [outlier['pred_0'], outlier['pred_1'], outlier['pred_2']]
    pixdim = fixed_CT_image.GetSpacing()  # Assuming the pixel dimensions are the same for both images

    # Transform the moving image based on true coordinates
    transformed_moving_CT_array_true = transform_moving_ct(
                    moving_CT_image, fixed_CT_image, true_coords, pixdim)
    # Transform the moving image based on predicted coordinates
    transformed_moving_CT_array_pred = transform_moving_ct(
                    moving_CT_image, fixed_CT_image, pred_coords, pixdim)

    # Calculate the difference between the fixed image and transformed moving images (true and predicted coordinates)
    fixed_CT_array = itk.array_view_from_image(fixed_CT_image).astype(int)
    moving_CT_array = itk.array_view_from_image(moving_CT_image).astype(int)
    difference_true = transformed_moving_CT_array_true - fixed_CT_array
    difference_pred = transformed_moving_CT_array_pred - fixed_CT_array

    return fixed_CT_array, moving_CT_array, transformed_moving_CT_array_true, difference_true, transformed_moving_CT_array_pred, difference_pred


def plot_img(data, slice_index=50, colormap='jet', scaling_factor=1000.0):
    fixed_CT_array, moving_CT_array, transformed_moving_CT_array_true, difference_true, transformed_moving_CT_array_pred, difference_pred = data
    plt.figure(figsize=(6, 6))

    def add_subplot_with_colorbar(index, data, title):
        plt.subplot(3, 2, index)
        scaled_data = data[slice_index] / scaling_factor  # Scale the data
        img = plt.imshow(scaled_data, cmap=colormap)
        cbar = plt.colorbar(img, fraction=0.03, pad=0.03)
        cbar.ax.set_title(f'     x10^{int(np.log10(scaling_factor))}', fontsize=4)
        cbar.ax.tick_params(labelsize=6)

        plt.title(title, fontsize=6)
        
        # Set the tick label size for both x and y axes
        plt.tick_params(axis='both', which='major', labelsize=4)
        plt.tick_params(axis='both', which='minor', labelsize=4)

    # Add subplots with smaller titles and axis tick labels
    add_subplot_with_colorbar(1, fixed_CT_array, 'Fixed Image')
    add_subplot_with_colorbar(2, moving_CT_array, 'Moving Image')
    add_subplot_with_colorbar(3, transformed_moving_CT_array_true, 'Transformed Moving (True)')
    add_subplot_with_colorbar(4, difference_true, 'Difference for (True coordination)')
    add_subplot_with_colorbar(5, transformed_moving_CT_array_pred, 'Transformed Moving (Pred)')
    add_subplot_with_colorbar(6, difference_pred, 'Difference for (Pred coordination)')

    plt.tight_layout()
    return plt


def display_coordination_info(outlier):
    """
    Create an HTML string to display coordination information in a styled table.

    Parameters:
    - outlier: dict containing 'true_0', 'true_1', 'true_2', 'pred_0', 'pred_1', 'pred_2' coordinates.

    Returns:
    - A string containing HTML representation of the coordination table.
    """
    # Calculate differences
    _0_dis = outlier['pred_0'] - outlier['true_0']
    _1_dis = outlier['pred_1'] - outlier['true_1']
    _2_dis = outlier['pred_2'] - outlier['true_2']

    # Create a DataFrame for the coordinates
    coord_data = {
        'Coordination': ['True', 'Predicted', 'Difference'],
        'X': [outlier['true_0'], outlier['pred_0'], _0_dis],
        'Y': [outlier['true_1'], outlier['pred_1'], _1_dis],
        'Z': [outlier['true_2'], outlier['pred_2'], _2_dis]
    }
    coord_df = pd.DataFrame(coord_data)
    coord_df = coord_df.round(2)

    # Convert the DataFrame to HTML without the index
    coord_html = coord_df.to_html(index=False, justify='center')
    
    return coord_html


def load_css(file_path):
    with open(file_path, 'r') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def plot_colored_wordcloud(data_dict, title):
    # Normalize the values to a range between 0 and 1
    min_val = min(data_dict.values())
    max_val = max(data_dict.values())
    norm_values = {k: (v - min_val) / (max_val - min_val) for k, v in data_dict.items()}
    
    # Define a colormap
    colormap = plt.cm.viridis
    
    # Generate word cloud
    wordcloud = WordCloud(width=300, height=150, background_color='white').generate_from_frequencies(data_dict)
    
    # Recolor the word cloud based on normalized values
    def color_func(word, *args, **kwargs):
        return mcolors.to_hex(colormap(norm_values[word]))

    wordcloud = wordcloud.recolor(color_func=color_func)
    
    # Display the word cloud
    plt.figure(figsize=(6, 3))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    
    # Adjust the colorbar to be smaller
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=min_val, vmax=max_val)),
        ax=plt.gca(),
        orientation='vertical',
        fraction=0.02,  # Smaller fraction makes the colorbar smaller
        pad=0.04         # Padding between the plot and colorbar
    )
    cbar.set_label('(mm)', rotation=0, labelpad=-25, y=1.06, fontsize=8)  # Add label on top of the colorbar
    cbar.ax.tick_params(labelsize=8)  # Adjust colorbar label size
    
    plt.tight_layout()  # Ensure everything fits without overlap
    plt.show()
    st.pyplot(plt)


def calculate_ensemble_average(dataset_option):
    df_ensemble = pd.read_pickle('df_best_esms.pkl')

    df_list = []  # List to hold DataFrames for each model
        
        # Loop through each model/tag in available_roots
    for index, row in df_ensemble[:-1].iterrows():
        try:
            # Depending on the dataset option, load the appropriate dataset
            if dataset_option == 'Test':
                df = read_from_pickle(row['test_output'])
            elif dataset_option == 'Val':
                df = read_from_pickle(row['val_output'])
            else:
                df = read_from_pickle(row['train_output'])
            
            # Append the DataFrame to the list if valid
            if df is not None:
                df_list.append(df)
            else:
                st.write(f"No data found for model {row['tag']} in {dataset_option} mode.")
        
        except Exception as e:
            st.write(f"Error loading data for model {row['tag']}: {e}")
            continue  # Skip this model if loading fails
    

    # Check if any DataFrames were loaded
    if not df_list:
        return None
        
        # Concatenate all DataFrames and compute the mean
        combined_df = pd.concat(df_list)
        averaged_df = combined_df.mean(numeric_only=True)
        
        return averaged_df


def plot_ensemble_average_boxplots(averaged_data):
    N = 6  # Number of subplots (for the 6 deltas)
    colors = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]

    # Create subplots
    fig = make_subplots(rows=1, cols=6, subplot_titles=['ΔX', 'ΔY', 'ΔZ', 'ΔR', 'L1', 'L2'])

    # Add scatter and box plots for averaged data
    deltas = ['0_dis', '1_dis', '2_dis', 'Euc_dis', 'L1_dis', 'L2_dis']
    for i, delta in enumerate(deltas, start=1):
        fig.add_trace(go.Scatter(
            x=[i], y=[averaged_data[delta]], mode='markers',
            marker=dict(color=colors[i-1], size=10), name=f"Avg {delta}"
        ), row=1, col=i)

    # Update layout
    fig.update_layout(height=600, width=1600, showlegend=False)
    return fig



def highlight_min_max(df, loss_columns):
    # Apply styling with conditional formatting for min and max values
    styles = df.style.apply(lambda x: ['background-color: cornflowerblue' if v == x.min() 
                                       else 'background-color: orange' if v == x.max() 
                                       else '' for v in x], subset=loss_columns)
    return styles


import plotly.express as px

def plot_overview(df_overview, mode, selected_approach):

    styled_df = pd.DataFrame()
    if selected_approach == "Ensemble":

        fig = px.scatter(
            df_overview, 
            x='esm', 
            y=f'avg_{mode}_loss', 
            title=f"{mode.capitalize()} losses in all Ensembles.",
            labels={'tr': 'Trial', f'{mode}_loss': f'{mode} Loss', 'esm': 'Ensembles'}
        )
        st.plotly_chart(fig)
        df_overview_clean = df_overview[['esm', 'tr', 'avg_train_loss', 'avg_val_loss', 'avg_test_loss']].dropna()
        df_overview_clean['esm'] = pd.to_numeric(df_overview_clean['esm'], errors='coerce')
        df_overview_sorted = df_overview_clean.sort_values(by='esm', ascending=True)
        df_overview_sorted[['avg_train_loss', 'avg_val_loss', 'avg_test_loss']] = df_overview_sorted[['avg_train_loss', 'avg_val_loss', 'avg_test_loss']].apply(lambda x: x.round(3).astype(str))

        styled_df = highlight_min_max(df_overview_sorted, ['avg_train_loss', 'avg_val_loss', 'avg_test_loss'])
    

    elif selected_approach == "Single":

        fig = px.scatter(
            df_overview, 
            x='esm', 
            y=f'{mode}_loss', 
            color='tr',
            title=f"{mode.capitalize()} losses in Trials.",
            labels={'tr': 'Trial', f'{mode}_loss': f'{mode} Loss', 'esm': 'Ensembles'}
        )
        st.plotly_chart(fig)
        df_overview_clean = df_overview[['esm', 'tr', 'train_loss', 'val_loss', 'test_loss']].dropna()
        df_overview_clean['esm'] = pd.to_numeric(df_overview_clean['esm'], errors='coerce')
        df_overview_sorted = df_overview_clean.sort_values(by=['esm', 'tr'] , ascending=True)
        df_overview_sorted[['train_loss', 'val_loss', 'test_loss']] = df_overview_sorted[['train_loss', 'val_loss', 'test_loss']].apply(lambda x: x.round(3).astype(str))

        styled_df = highlight_min_max(df_overview_sorted, ['train_loss', 'val_loss', 'test_loss'])


    

    return styled_df