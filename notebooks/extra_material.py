        
##########################################################################
# This section is used to extract the features in features_dl list from the csv file
# I can have a part like this to extract the position label and maybe in the future other parameters.

        # Load features
        df_features = pd.read_csv(os.path.join(data_dir, filename_stratified_sampling_test_csv), sep=';') #, decimal=',')
        df_features[patient_id_col] = ['%0.{}d'.format(patient_id_length) % int(x) for x in df_features[patient_id_col]]

        # Create list of images and labels
        images_list, labels_list, features_list, patient_ids_list = list(), list(), list(), list()
        labels_unique = [x for x in os.listdir(data_dir) if not x.endswith('.csv')]
        # sort() to make sure that different platforms use the same order --> required for random.shuffle() later
        labels_unique.sort()

        # Get PatientIDs
        patient_ids_list = df_features[patient_id_col].values.tolist()
        # sort() to make sure that different platforms use the same order --> required for random.shuffle() later
        patient_ids_list.sort()
        
        for patient_id in patient_ids_list:
            # Patient's features data
            df_features_i = df_features[df_features[patient_id_col] == patient_id]
            # Features
            features_list += [y for y in
                            df_features_i[features].values.tolist()]

            # features_list += [[float(str(x).replace(',', '.')) for x in y] for y in
            #                   df_features_i[features].values.tolist()]
            # Endpoint
            labels_list.append(int(df_features_i[data_preproc_config.endpoint])) # check this on Daniel bran to silent the warning.

        assert len(patient_ids_list) == len(features_list) == len(labels_list)

##########################################################################

##########################################################################
# This part is quite important since it will be used to make the data_dict for the following steps.
 
        # Note: '0' in front of string is okay: int('0123') will become 123
        data_dicts = [ # Some changes here chane segmentation map to weeklyCTs (ALSO, THIS PART SHOULD BE CHANGED FOR SUBTRACTIONCT)
            {'ct': os.path.join(data_dir, str(label_name), patient_id, data_preproc_config.filename_ct_npy),
            #  'subtractionct': os.path.join(data_dir, str(label_name), patient_id, data_preproc_config.filename_subtractionct),   
            'rtdose': os.path.join(data_dir, str(label_name), patient_id, data_preproc_config.filename_rtdose_npy),

            ######## Here I made some changes ########### 
            'segmentation_map': os.path.join(data_dir, str(label_name), patient_id,
                                            data_preproc_config.filename_segmentation_map_npy),

            #  'weeklyct': os.path.join(data_dir, str(label_name), patient_id,
            #                                   data_preproc_config.filename_weeklyct_npy),

            'features': feature_name,
            'label': label_name,
            'patient_id': patient_id}
            for patient_id, feature_name, label_name in zip(patient_ids_list, features_list, labels_list)
            if int(patient_id) not in exclude_patients
        ]
##########################################################################

##########################################################################
# It can be quite nice to add stratified sampling for my study, but the problem is that we do NOT need stratified sampling for this task,
# However, in one way we may need to use stratified sampling and that is when we want to have the same percent of each patients in a dataset.
# NOTE: when we do NOT want to divide the dataset based on patient number.

        # Whether to perform random split or to perform stratified sampling
        if sampling_type == 'random':
            # Select random subset for testing
            if perform_test_run:
                data_dicts = data_dicts[:config.n_samples]

            # Training-internal_validation-test data split
            patients_test = df_features[df_features[split_col] == 'test'][patient_id_col].tolist()
            train_val_dict = [x for x in data_dicts if x['patient_id'] not in patients_test]
            n_train = round(len(train_val_dict) * train_frac)

            train_dict = train_val_dict[:n_train]
            val_dict = train_val_dict[n_train:]
            test_dict = [x for x in data_dicts if x['patient_id'] in patients_test]

        elif sampling_type == 'stratified': # Most of the time, we are using this option.
            # Stratified sampling
            path_filename = os.path.join(data_dir, filename_stratified_sampling_full_csv)
            if not os.path.isfile(path_filename) or perform_stratified_sampling_full:
                # Create stratified_sampling.csv if file does not exists, or recreate if requested
                logger.my_print('Creating {}.'.format(path_filename))
                # Create stratified samples for train and validation set
                perform_stratified_sampling(df=df_features, frac=train_frac / (train_frac + val_frac),
                                            strata_groups=strata_groups, split_col=split_col,
                                            output_path_filename=path_filename, seed=seed, logger=logger)

            # Load list of patients with split info
            df_split = pd.read_csv(path_filename, sep=';')
            df_split[patient_id_col] = ['%0.{}d'.format(patient_id_length) % int(x) for x in df_split[patient_id_col]]

            # Exclude patients
            df_split = df_split[~df_split[patient_id_col].astype(np.int64).isin(exclude_patients)]

            # Make sure that files in the dataset folders comprehend with the label in filename_stratified_sampling_test_csv
            for l in labels_unique:
                # patient_ids_l = [x.replace('.npy', '') for x in os.listdir(os.path.join(data_dir, l))]
                patient_ids_l = os.listdir(os.path.join(data_dir, l))
                patient_ids_l = [x for x in patient_ids_l if
                                int(x) not in exclude_patients and x in df_split[patient_id_col].tolist()]
                for p in patient_ids_l:
                    df_i = df_split[df_split[patient_id_col] == p]
                    # print(df_i)
                    # print(labels_unique, l, int(df_i[data_preproc_config.endpoint].values[0]), int(l))
                    assert int(df_i[data_preproc_config.endpoint].values[0]) == int(l)

            # Split
            total_size = len(df_split)
            df_train = df_split[df_split[split_col] == 'train']
            df_val = df_split[df_split[split_col] == 'val']
            df_test = df_split[df_split[split_col] == 'test']

            # Print stats
            get_df_stats(df=df_split, groups=strata_groups, mode='full', frac=1, total_size=total_size, logger=logger)
            get_df_stats(df=df_train, groups=strata_groups, mode='train', frac=train_frac, total_size=total_size,
                        logger=logger)
            get_df_stats(df=df_val, groups=strata_groups, mode='val', frac=val_frac, total_size=total_size, logger=logger)
            get_df_stats(df=df_test, groups=strata_groups, mode='test', frac=1 - train_frac - val_frac,
                        total_size=total_size, logger=logger)

            train_dict, val_dict, test_dict = list(), list(), list()
            for d_dict in data_dicts:
                patient_id = d_dict['patient_id']
                df_split_i = df_split[df_split[patient_id_col] == patient_id]
                assert len(df_split_i) == 1
                split_i = df_split_i['Split'].values[0]
                if split_i == 'train':
                    train_dict.append(d_dict)
                elif split_i == 'val':
                    val_dict.append(d_dict)
                elif split_i == 'test':
                    test_dict.append(d_dict)
                else:
                    raise ValueError('Invalid split_i: {}.'.format(split_i))
##########################################################################
# Manual Loss Functions (Try to ad them to the next version)
import torch
import torch.nn.functional as F

def mse_loss(pred, target):
    return torch.mean((pred - target) ** 2)

def mae_loss(pred, target):
    return torch.mean(torch.abs(pred - target))

def chamfer_distance(pred, target):
    # pred and target are of shape (N, D) where N is the number of points and D is the dimensionality
    pred_expanded = pred.unsqueeze(1)  # (N, 1, D)
    target_expanded = target.unsqueeze(0)  # (1, M, D)
    
    dist_matrix = torch.cdist(pred_expanded, target_expanded, p=2)  # (N, M)
    
    min_dist_pred_to_target = torch.min(dist_matrix, dim=1)[0]
    min_dist_target_to_pred = torch.min(dist_matrix, dim=0)[0]
    
    return torch.mean(min_dist_pred_to_target) + torch.mean(min_dist_target_to_pred)

def hausdorff_distance(pred, target):
    pred_expanded = pred.unsqueeze(1)
    target_expanded = target.unsqueeze(0)
    
    dist_matrix = torch.cdist(pred_expanded, target_expanded, p=2)
    
    max_dist_pred_to_target = torch.max(torch.min(dist_matrix, dim=1)[0])
    max_dist_target_to_pred = torch.max(torch.min(dist_matrix, dim=0)[0])
    
    return torch.max(max_dist_pred_to_target, max_dist_target_to_pred)



def mutual_information(pred, target, num_bins=256):
    hist_2d = torch.histc(pred * num_bins + target, bins=num_bins**2, min=0, max=num_bins**2-1)
    hist_2d = hist_2d.view(num_bins, num_bins)
    
    pxy = hist_2d / torch.sum(hist_2d)  # joint distribution
    px = torch.sum(pxy, dim=1)
    py = torch.sum(pxy, dim=0)
    
    px_py = px.unsqueeze(1) * py.unsqueeze(0)
    nzs = pxy > 0
    
    return torch.sum(pxy[nzs] * torch.log(pxy[nzs] / px_py[nzs]))


def normalized_cross_correlation(pred, target):
    pred_mean = torch.mean(pred)
    target_mean = torch.mean(target)
    
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    
    numerator = torch.sum(pred_centered * target_centered)
    denominator = torch.sqrt(torch.sum(pred_centered ** 2) * torch.sum(target_centered ** 2))
    
    return numerator / denominator


def ssim(pred, target, C1=0.01**2, C2=0.03**2, kernel_size=11, sigma=1.5):
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([torch.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    window = gaussian(kernel_size, sigma).unsqueeze(1).mm(gaussian(kernel_size, sigma).unsqueeze(1).t()).unsqueeze(0).unsqueeze(0)
    window = window.expand(pred.size(1), 1, kernel_size, kernel_size)
    window = window.cuda() if pred.is_cuda else window

    mu1 = F.conv2d(pred, window, padding=kernel_size//2, groups=pred.size(1))
    mu2 = F.conv2d(target, window, padding=kernel_size//2, groups=target.size(1))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=kernel_size//2, groups=pred.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=kernel_size//2, groups=target.size(1)) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=kernel_size//2, groups=pred.size(1)) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def point_to_plane_error(pred, target, normals):
    diff = pred - target
    return torch.mean((diff * normals).sum(dim=1) ** 2)