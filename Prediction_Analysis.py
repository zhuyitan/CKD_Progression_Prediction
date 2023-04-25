from collections import Counter
import os
import pandas as pd
import sys
import tensorflow as tf
import numpy as np
import argparse
import pickle
import shutil
from LSTM import get_model_parameter, LSTM_Classification_Analysis


ii = int(sys.argv[1])           # Time interval for generating time course, e.g. 7
min_len = int(sys.argv[2])      # Minimum length of time course, e.g. 5
max_len = int(sys.argv[3])      # Maximum length of time course, e.g. 100
gap = int(sys.argv[4])          # Gap between the end of time series and the start of prediction period in days, e.g. 7. It is used to prevent information leakage.
period = int(sys.argv[5])       # Period for predicting whether the disease will progress to late stage in days, e.g. 365.
ratio = int(sys.argv[6])        # The maximum number of control patients that each case patient will be matched with, e.g. 4
model_id = int(sys.argv[7])     # ID of neural network architecture used for modeling, e.g. 7
variable = str(sys.argv[8])     # A string of either 'Limited' or 'All'. 'All' means using all variables for modeling. 'Limited' means using essential variables only.
if len(sys.argv) > 9:           # If cutoff2 is None, use data including patient stage IIIb information for modeling and prediction; otherwise, use data without stage IIIb information.
    cutoff2 = int(sys.argv[9])
else:
    cutoff2 = None



impute = 2      # For a numeric variable, this impute method uses its mean to replace missing values and add a binary indicator to label imputed missing values.

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

result_folder = save_folder + 'Prediction_Result_Impute_' + str(impute) + '_Model_' + str(model_id) + '/'
if os.path.exists(result_folder):
    shutil.rmtree(result_folder)
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

result = {}
sampleID = {}
summary = pd.DataFrame(np.empty((para['num_fold'], 9)).fill(np.nan), index=['cv_' + str(i) for i in range(para['num_fold'])],
                       columns=['train_AUROC', 'train_ACC', 'train_MCC', 'val_AUROC', 'val_ACC', 'val_MCC',
                                'test_AUROC', 'test_ACC', 'test_MCC'])
for foldID in range(para['num_fold']):
    cvID = 'cv_' + str(foldID)
    cv_folder = result_folder + cvID + '/'
    os.makedirs(cv_folder)

    partition_folder = save_folder + 'Partition/' + cvID + '/'
    sampleID[cvID] = {}
    sampleID[cvID]['trainID'] = pd.read_csv(partition_folder + '/TrainList.txt', sep='\t', engine='c',
                                            na_values=['na', '-', ''], header=None, index_col=None).values[:, 0]
    sampleID[cvID]['valID'] = pd.read_csv(partition_folder + '/ValList.txt', sep='\t', engine='c',
                                          na_values=['na', '-', ''], header=None, index_col=None).values[:, 0]
    sampleID[cvID]['testID'] = pd.read_csv(partition_folder + '/TestList.txt', sep='\t', engine='c',
                                           na_values=['na', '-', ''], header=None, index_col=None).values[:, 0]

    predResult, perM, perf, dropout = LSTM_Classification_Analysis(data=matched_data, summary=sum_case,
        sampleID=sampleID[cvID], resultFolder=cv_folder, para=para)

    result[cvID] = {}
    result[cvID]['predResult'] = predResult
    result[cvID]['perM'] = perM
    result[cvID]['perf'] = perf
    result[cvID]['dropout'] = dropout

    for i in ['train', 'val', 'test']:
        summary.loc[cvID, i + '_AUROC'] = result[cvID]['perf'].loc[i, 'AUROC']
        summary.loc[cvID, i + '_ACC'] = result[cvID]['perf'].loc[i, 'ACC']
        summary.loc[cvID, i + '_MCC'] = result[cvID]['perf'].loc[i, 'MCC']
    summary.to_csv(result_folder + '/Performance_Summary.txt', header=True, index=True, sep='\t', line_terminator='\r\n')

    file = open(result_folder + 'All_Data.pickle', 'wb')
    pickle.dump((result, sampleID, summary, para), file)
    file.close()
