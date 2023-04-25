import os
import pandas as pd
import sys
import numpy as np
import pickle
from LSTM import get_model_parameter, LSTM_Classification_Analysis_With_Feature_Subsetting



model_id = int(sys.argv[1])         # ID of neural network architecture used for modeling, e.g. 7
variable = str(sys.argv[2])         # A string of either 'Limited' or 'All'. 'All' means using all variables for modeling. 'Limited' means using essential variables only.
foldID_start = int(sys.argv[3])     # The ID of the first cross-validation trial to be performed.
foldID_end = int(sys.argv[4])       # The ID of the last cross-validation trial to be performed.
fea_num_start = int(sys.argv[5])    # The number of variables to start the sequential forward variable selection process.
fea_num_end = int(sys.argv[6])      # The number of variables to end the sequential forward variable selection process.
fit = eval(sys.argv[7])             # A Boolean variable indicating whether the numbers of units/neurons in layers should commensurate with the number of input features.
if len(sys.argv) > 8:               # If cutoff2 is None, use data including patient stage IIIb information for modeling and prediction; otherwise, use data without stage IIIb information.
    cutoff2 = int(sys.argv[8])
else:
    cutoff2 = None



# These variables are used to identify the data to perform sequential forward variable selection.
ii = 7
min_len = 5
max_len = 100
gap = 7
period = 365
ratio = 4
impute = 2



pd.set_option('display.max_columns', None)

save_folder = '../Processed_Data/Time_Course_Data_' + variable + '_Variables_New_Status_Call/Matched_Time_Sequence_Data/Interval_' + \
              str(ii) + '_MinLength_' + str(min_len) + '_MaxLength_' + str(max_len) + '_Gap_' + str(gap) + \
              '_Period_' + str(period) + '_Ratio_' + str(ratio) + '_Cutoff2_' + str(cutoff2) + '/'
file = open(save_folder + 'Matched_Case_And_Control_Data.pickle', 'rb')
matched_data1, matched_data2 = pickle.load(file)
file.close()

if impute == 1:
    matched_data = matched_data1
if impute == 2:
    matched_data = matched_data2
matched_data1 = None
matched_data2 = None

stat_1 = pd.read_csv('../Processed_Data/Variable_Histogram/lab/Data_Statistics.txt', sep='\t', engine='c',
                     na_values=['na', '-', ''], header=0, index_col=0, low_memory=False)
stat_1.index = np.array(['lab:' + str(i) for i in stat_1.index])
stat_2 = pd.read_csv('../Processed_Data/Variable_Histogram/flowsheet/Data_Statistics.txt', sep='\t', engine='c',
                     na_values=['na', '-', ''], header=0, index_col=0, low_memory=False)
stat_2.index = np.array(['flowsheet:' + str(i) for i in stat_2.index])
stat = pd.concat((stat_1, stat_2), axis=0)

all_mrn = list(matched_data.keys())
all_age = list(matched_data[all_mrn[0]].index)
for i in range(1, len(all_mrn)):
    all_age = all_age + list(matched_data[all_mrn[i]].index)
age_mean = np.mean(all_age)
age_std = np.std(all_age)

for mrn in all_mrn:
    matched_data[mrn].index = (matched_data[mrn].index - age_mean) / age_std
if impute == 2:
    all_var = matched_data[all_mrn[0]].columns
    id_1 = np.where([True if 'flowsheet:' in v else False for v in all_var])[0]
    id_2 = np.where([True if 'lab:' in v else False for v in all_var])[0]
    id_3 = np.where([True if 'naLabel' in v else False for v in all_var])[0]
    numeric_var = all_var[np.sort(np.setdiff1d(np.union1d(id_1, id_2), id_3))]
    for mrn in all_mrn:
        matched_data[mrn].loc[:, numeric_var] = matched_data[mrn].loc[:, numeric_var] - stat.loc[numeric_var, 'mean'][np.newaxis, :]
        matched_data[mrn].loc[:, numeric_var] = matched_data[mrn].loc[:, numeric_var] / stat.loc[numeric_var, 'std'][np.newaxis, :]



sum_case = pd.read_csv(save_folder + 'Case_Sample_Summary.txt', sep='\t', engine='c',
                       na_values=['na', '-', ''], header=0, index_col=0, low_memory=False)

result_folder = save_folder + 'SFS_Result_Impute_' + str(impute) + '_Model_' + str(model_id) + '_Fit_' + str(fit) + '/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

para = get_model_parameter('../Processed_Data/Prediction_Models/LSTM_Classifier_' + str(model_id) + '.txt')
para['num_fold'] = 100

if cutoff2 is None:
    if period == 365:
        para['batch_size'] = 5
    if period == 90:
        para['batch_size'] = 3
else:
    para['batch_size'] = 2

all_feature = matched_data[list(matched_data.keys())[0]].columns
all_feature = [i.split('---')[0] for i in all_feature]
all_feature = [i.split('_naLabel')[0] for i in all_feature]
all_feature_dup = np.array([i.split(':')[1] for i in all_feature])
all_feature = np.unique(all_feature_dup)

all_feature = np.setdiff1d(all_feature, ['CIGARETTES_YN', 'CIGARS_YN', 'SMOKING_TOB_USE',
                                         'INJECTION_YN', 'IV_DRUG_USER_YN', 'MARIJUANA_YN',
                                         'PILL_YN'])

result = {}
for foldID in range(foldID_start, foldID_end):
    cvID = 'cv_' + str(foldID)
    cv_folder = result_folder + cvID + '/'
    if not os.path.exists(cv_folder):
        os.makedirs(cv_folder)

    if os.path.exists(cv_folder + 'All_Data.pickle'):
        file = open(cv_folder + 'All_Data.pickle', 'rb')
        result[cvID] = pickle.load(file)
        file.close()
    else:
        result[cvID] = {}

    partition_folder = save_folder + 'Partition_Two_Tiers/' + cvID + '/'
    sampleID = {}
    sampleID['trainID'] = pd.read_csv(partition_folder + '/TrainList.txt', sep='\t', engine='c',
                                      na_values=['na', '-', ''], header=None, index_col=None).values[:, 0]
    sampleID['valID'] = pd.read_csv(partition_folder + '/ValList.txt', sep='\t', engine='c',
                                    na_values=['na', '-', ''], header=None, index_col=None).values[:, 0]
    sampleID['testID'] = pd.read_csv(partition_folder + '/TestList.txt', sep='\t', engine='c',
                                     na_values=['na', '-', ''], header=None, index_col=None).values[:, 0]
    red_sampleID = {}
    red_sampleID['trainID'] = pd.read_csv(partition_folder + '/TrainList_Inner.txt', sep='\t', engine='c',
                                          na_values=['na', '-', ''], header=None, index_col=None).values[:, 0]
    red_sampleID['valID'] = pd.read_csv(partition_folder + '/ValList.txt', sep='\t', engine='c',
                                        na_values=['na', '-', ''], header=None, index_col=None).values[:, 0]
    red_sampleID['testID'] = pd.read_csv(partition_folder + '/TestList_Inner.txt', sep='\t', engine='c',
                                         na_values=['na', '-', ''], header=None, index_col=None).values[:, 0]

    for num_feature in range(fea_num_start, fea_num_end, 1):
        num_feature_str = str(num_feature) + '_Features'
        num_feature_folder = cv_folder + num_feature_str + '/'
        if not os.path.exists(num_feature_folder):
            os.makedirs(num_feature_folder)
        if num_feature_str in result[cvID].keys():
            if 'Out_Tier_Test' in result[cvID][num_feature_str].keys():
                continue
        if num_feature == 1:
            cur_feature = np.array([])
        else:
            cur_feature = result[cvID][str(num_feature - 1) + '_Features']['Out_Tier_Test']['feature']
        pot_feature = np.setdiff1d(all_feature, cur_feature)
        if num_feature_str not in result[cvID].keys():
            result[cvID][num_feature_str] = {}
            temp_r = np.empty((len(pot_feature), 3))
            temp_r.fill(np.nan)
            result[cvID][num_feature_str]['Performance'] = pd.DataFrame(temp_r, index=['Add_' + l for l in pot_feature],
                columns=['train_AUROC', 'val_AUROC', 'test_AUROC'])

        for k in range(len(pot_feature)):
            add_feature_str = 'Add_' + pot_feature[k]
            if not pd.isna(result[cvID][num_feature_str]['Performance'].loc[add_feature_str, 'test_AUROC']):
                continue

            red_feature = np.sort(np.union1d(cur_feature, pot_feature[k]))
            red_resultFolder = num_feature_folder + add_feature_str + '/'
            if not os.path.exists(red_resultFolder):
                os.makedirs(red_resultFolder)
            if fit is True:
                num_f = len(np.where(np.isin(all_feature_dup, red_feature))[0])
                para['lstm_layers'] = [num_f + 1]
                para['dense_layers'] = [num_f + 1, int(num_f + 1 - (num_f + 1 - 2) / 3), int(num_f + 1 - (num_f + 1 - 2) * 2 / 3)]
            result_r_k = LSTM_Classification_Analysis_With_Feature_Subsetting(matched_data, sum_case, red_sampleID,
                red_resultFolder, para, red_feature)

            result[cvID][num_feature_str][add_feature_str] = result_r_k
            result[cvID][num_feature_str]['Performance'].loc[add_feature_str, 'train_AUROC'] = \
                result_r_k['perf'].loc['train', 'AUROC']
            result[cvID][num_feature_str]['Performance'].loc[add_feature_str, 'val_AUROC'] = \
                result_r_k['perf'].loc['val', 'AUROC']
            result[cvID][num_feature_str]['Performance'].loc[add_feature_str, 'test_AUROC'] = \
                result_r_k['perf'].loc['test', 'AUROC']
            file = open(cv_folder + 'All_Data.pickle', 'wb')
            pickle.dump(result[cvID], file)
            file.close()
        id_max = np.argmax(result[cvID][num_feature_str]['Performance'].test_AUROC.values)
        result[cvID][num_feature_str]['Best_Model'] = \
            result[cvID][num_feature_str][result[cvID][num_feature_str]['Performance'].index[id_max]]

        test_resultFolder = cv_folder + num_feature_str + '/Out_Tier_Test/'
        if not os.path.exists(test_resultFolder):
            os.makedirs(test_resultFolder)
        cur_feature = result[cvID][num_feature_str]['Best_Model']['feature']
        if fit is True:
            num_f = len(np.where(np.isin(all_feature_dup, cur_feature))[0])
            para['lstm_layers'] = [num_f + 1]
            para['dense_layers'] = [num_f + 1, int(num_f + 1 - (num_f + 1 - 2) / 3), int(num_f + 1 - (num_f + 1 - 2) * 2 / 3)]
        result[cvID][num_feature_str]['Out_Tier_Test'] = LSTM_Classification_Analysis_With_Feature_Subsetting(matched_data,
            sum_case, sampleID, test_resultFolder, para, cur_feature)

        file = open(cv_folder + 'All_Data.pickle', 'wb')
        pickle.dump(result[cvID], file)
        file.close()