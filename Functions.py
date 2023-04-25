import os
import pandas as pd
import keras
from keras.models import load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
import numpy as np
import configparser
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef
from keras import backend
from keras import optimizers
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping



def get_model_parameter(model_file):
    # path to a model parameter file
    config = configparser.ConfigParser()
    config.read(model_file)
    section = config.sections()
    params = {}
    for sec in section:
        for k, v in config.items(sec):
            if k not in params:
                params[k] = eval(v)
    return params



def get_DNN_optimizer(opt_name):
    if opt_name == 'SGD':
        optimizer = optimizers.GSD()
    elif opt_name == 'SGD_momentum':
        optimizer = optimizers.GSD(momentum=0.9)
    elif opt_name == 'SGD_momentum_nesterov':
        optimizer = optimizers.GSD(momentum=0.9, nesterov=True)
    elif opt_name == 'RMSprop':
        optimizer = optimizers.RMSprop()
    elif opt_name == 'Adagrad':
        optimizer = optimizers.Adagrad()
    elif opt_name == 'Adadelta':
        optimizer = optimizers.Adadelta()
    elif opt_name == 'Adam':
        optimizer = optimizers.Adam()
    elif opt_name == 'Adam_amsgrad':
        optimizer = optimizers.Adam(amsgrad=True)
    else:
        optimizer = optimizers.Adam()

    return optimizer



class DataGenerator(keras.utils.Sequence):
    def __init__(self, data, summary, dgLabel, batch_size, shuffle=True):
        self.data = data
        self.summary = summary
        self.num_var = data[str(summary.index[0])].shape[1] + 1
        self.numSample = summary.shape[0]
        self.dgLabel = dgLabel
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.uni_len = np.sort(np.unique(summary.Length))
        self.numBatch = 0
        self.ids = []
        for l in self.uni_len:
            idl = np.sort(np.where(self.summary.Length == l)[0])
            for i in range(int(np.ceil(len(idl) / self.batch_size))):
                self.ids.append(idl[int(i * self.batch_size):int(min((i + 1) * self.batch_size, len(idl)))])
                self.numBatch += 1
        sample_index = []
        sample_trans = []
        for l in range(len(self.ids)):
            for i in self.ids[l]:
                sample_index.append(str(self.summary.index[i]))
                sample_trans.append(1)
                sample_index = sample_index + [j + '---' + str(self.summary.index[i]) for j in summary.iloc[i, :].MatchedTo.split(',')]
                sample_trans = sample_trans + [0 for j in range(len(summary.iloc[i, :].MatchedTo.split(',')))]
        self.truth = pd.DataFrame(sample_trans, index=sample_index, columns=['Truth'])

    def __len__(self):
        # Denotes the number of batches per epoch
        return self.numBatch

    def __getitem__(self, index):
        # Generate one batch of data. Input data dimensions should be (batch size, sequence length, input vector
        # dimension at a time point)

        seq_len = self.summary.iloc[self.ids[index][0], :].Length
        for i in range(len(self.ids[index])):
            matched_sample = self.summary.iloc[self.ids[index][i], :].MatchedTo.split(',')
            temp_x = np.empty((len(matched_sample) + 1, seq_len, self.num_var))
            temp_x[0, :, 0] = self.data[str(self.summary.index[self.ids[index][i]])].index
            temp_x[0, :, 1:] = self.data[str(self.summary.index[self.ids[index][i]])].values
            temp_y = [1]
            for j in range(len(matched_sample)):
                temp_x[1 + j, :, 0] = self.data[matched_sample[j] + '---' + str(self.summary.index[self.ids[index][i]])].index
                temp_x[1 + j, :, 1:] = self.data[matched_sample[j] + '---' + str(self.summary.index[self.ids[index][i]])].values
                temp_y.append(0)
            if i == 0:
                x = temp_x.copy()
                y = np.array(temp_y)
            else:
                x = np.concatenate((x, temp_x), axis=0)
                y = np.concatenate((y, np.array(temp_y)))
#        print(self.dgLabel + '---' + str(index))
        return x, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        if self.shuffle:
            self.ids = []
            temp_ids = []
            for l in self.uni_len:
                idl = np.where(self.summary.Length == l)[0]
                np.random.shuffle(idl)
                for i in range(int(np.ceil(len(idl) / self.batch_size))):
                    temp_ids.append(idl[int(i * self.batch_size):int(min((i + 1) * self.batch_size, len(idl)))])
            id_temp_ids = np.array(range(len(temp_ids)))
            np.random.shuffle(id_temp_ids)
            for i in id_temp_ids:
                self.ids.append(temp_ids[i])



class LSTM_Classifier():
    def __init__(self, params, num_class, input_data_dim, dropout):
        self.params = params
        self.input_data_dim = input_data_dim
        self.dropout = dropout
        self.num_class = num_class

        input = Input(shape=(None, self.input_data_dim), name='Input')
        # For an input with variable length, the shape should be (None, input vector dimension at a time point).
        # None is used when the input sequence length may change; otherwise an positive integer is used.
        # Batch size is not included in the input shape, because the batch size is not fixed.
        # If using the batch_shape parameter instead of shape, it should be (batch size, None, input vector dimension at
        # a time point), and batch size is fixed for each batch.

        if len(self.params['lstm_layers']) > 1:
            d = LSTM(units=self.params['lstm_layers'][0], return_sequences=True, name='LSTM_' + str(0))(input)
            # units is the dimension of hidden state in LSTM. It is usually smaller than the input vector dimension.
            # The numbers of parameters for the input, input gate, forget gate, and output gate are all
            # units + input vector dimension at a time point * units + units * units
            # So the total number of parameters of a LSTM layer is four times of the number.
            for i in range(1, len(self.params['lstm_layers']) - 1):
                d = LSTM(units=self.params['lstm_layers'][i], return_sequences=True, name='LSTM_' + str(i))(d)
            d = LSTM(units=self.params['lstm_layers'][-1], return_sequences=False,
                     name='LSTM_' + str(len(self.params['lstm_layers']) - 1))(d)
        else:
            d = LSTM(units=self.params['lstm_layers'][0], return_sequences=False, name='LSTM_' + str(0))(input)

        for i in range(len(self.params['dense_layers'])):
            d = Dense(self.params['dense_layers'][i], activation=self.params['activation'], name='Dense_' + str(i))(d)
            d = Dropout(self.dropout, name='Dropout_Dense_' + str(i))(d)
        output = Dense(self.num_class, activation='softmax', name='output')(d)
        model = Model(inputs=input, outputs=output)
        model.compile(optimizer=get_DNN_optimizer(self.params['optimizer']), loss=self.params['loss'])
        # model.compile(optimizer=Adam(lr=0.01), loss=self.params['loss'])
        print(model.summary())
        self.model = model



def LSTM_Classification_Analysis(data, summary, sampleID, resultFolder, para):

    data_dim = data[list(data.keys())[0]].shape[1] + 1

    trainID = sampleID['trainID']
    valID = sampleID['valID']
    testID = sampleID['testID']
    train_dg = DataGenerator(data=data, summary=summary.iloc[trainID, :], dgLabel='trainDG', batch_size=para['batch_size'], shuffle=True)
    val_dg = DataGenerator(data=data, summary=summary.iloc[valID, :], dgLabel='valDG', batch_size=para['batch_size'], shuffle=False)
    class_weight = {}
    class_weight[0] = 1
    class_weight[1] = np.sum(train_dg.truth.values == 0) / np.sum(train_dg.truth.values == 1)

    perM = {}
    for i in ['train', 'val']:
        perM[i] = np.empty((len(para['drop']), para['epochs']))
        perM[i].fill(np.inf)
        perM[i] = pd.DataFrame(perM[i], index=['dropout_' + str(i) for i in para['drop']],
                               columns=['epoch_' + str(i) for i in range(para['epochs'])])

    for dpID in range(len(para['drop'])):
        label = 'dropout_' + str(para['drop'][dpID])
        print(label)

        monitor = 'val_loss'
        train_logger = CSVLogger(resultFolder + '/log_dropout_' + str(para['drop'][dpID]) + '.csv')
        model_saver = ModelCheckpoint(resultFolder + '/model_dropout_' + str(para['drop'][dpID]) + '.h5',
                                      monitor=monitor, save_best_only=True, save_weights_only=False)
        reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=para['rlr_factor'], patience=para['rlr_patience'],
                                      verbose=1, mode='auto', min_delta=para['rlr_min_delta'],
                                      cooldown=para['rlr_cooldown'], min_lr=para['rlr_min_lr'])
        early_stop = EarlyStopping(monitor=monitor, patience=para['es_patience'], min_delta=para['es_min_delta'],
                                   verbose=1)
        callbacks = [model_saver, train_logger, reduce_lr, early_stop]

        temp = LSTM_Classifier(params=para, num_class=2, input_data_dim=data_dim, dropout=para['drop'][dpID])
        history = temp.model.fit_generator(generator=train_dg, epochs=para['epochs'], verbose=para['verbose'],
            steps_per_epoch=train_dg.numBatch, class_weight=class_weight, max_queue_size=1, workers=1,
            use_multiprocessing=False, callbacks=callbacks, validation_data=val_dg)

        numEpoch = len(history.history['loss'])
        i = np.where(perM['train'].index == label)[0]
        perM['train'].iloc[i, :numEpoch] = history.history['loss']
        numEpoch = len(history.history['val_loss'])
        i = np.where(perM['val'].index == label)[0]
        perM['val'].iloc[i, :numEpoch] = history.history['val_loss']

        backend.clear_session()

    dpID, epID = np.unravel_index(np.argmin(perM['val'].values, axis=None), perM['val'].shape)
    model = load_model(resultFolder + '/model_dropout_' + str(para['drop'][dpID]) + '.h5')

    for i in range(len(para['drop'])):
        if i == dpID:
            continue
        os.remove(resultFolder + '/model_dropout_' + str(para['drop'][i]) + '.h5')
        os.remove(resultFolder + '/log_dropout_' + str(para['drop'][i]) + '.csv')

    predResult = {}

    train_dg = DataGenerator(data=data, summary=summary.iloc[trainID, :], dgLabel='trainDG_eval', batch_size=para['batch_size'], shuffle=False)
    predResult['train'] = pd.DataFrame(model.predict_generator(train_dg)[:, 1], index=train_dg.truth.index, columns=['Prediction'])
    predResult['train'] = pd.concat(objs=(train_dg.truth, predResult['train']), axis=1)

    val_dg = DataGenerator(data=data, summary=summary.iloc[valID, :], dgLabel='valDG_eval', batch_size=para['batch_size'], shuffle=False)
    predResult['val'] = pd.DataFrame(model.predict_generator(val_dg)[:, 1], index=val_dg.truth.index, columns=['Prediction'])
    predResult['val'] = pd.concat(objs=(val_dg.truth, predResult['val']), axis=1)

    test_dg = DataGenerator(data=data, summary=summary.iloc[testID, :], dgLabel='testDG_eval', batch_size=para['batch_size'], shuffle=False)
    predResult['test'] = pd.DataFrame(model.predict_generator(test_dg)[:, 1], index=test_dg.truth.index, columns=['Prediction'])
    predResult['test'] = pd.concat(objs=(test_dg.truth, predResult['test']), axis=1)

    backend.clear_session()

    perf = np.empty((3, 3))
    perf.fill(np.nan)
    perf = pd.DataFrame(perf, index=['train', 'val', 'test'], columns=['AUROC', 'ACC', 'MCC'])
    for k in ['train', 'val', 'test']:
        perf.loc[k, 'AUROC'] = roc_auc_score(predResult[k].Truth, predResult[k].Prediction)
        perf.loc[k, 'ACC'] = accuracy_score(predResult[k].Truth, np.round(predResult[k].Prediction).astype(np.int))
        perf.loc[k, 'MCC'] = matthews_corrcoef(predResult[k].Truth, np.round(predResult[k].Prediction).astype(np.int))
    perf.to_csv(resultFolder + '/Performance_dropout_' + str(para['drop'][dpID]) + '.txt', header=True, index=True,
                sep='\t', line_terminator='\r\n')

    print(perf)

    return predResult, perM, perf, 'dropout_' + str(para['drop'][dpID])



def LSTM_Classification_Analysis_With_Feature_Subsetting(matched_data, sum_case, sampleID, resultFolder, para, feature):

    all_feature = matched_data[list(matched_data.keys())[0]].columns
    ori_all_feature = all_feature
    all_feature = [i.split('---')[0] for i in all_feature]
    all_feature = [i.split('_naLabel')[0] for i in all_feature]
    all_feature = np.array([i.split(':')[1] for i in all_feature])
    id_k = np.sort(np.where(np.isin(all_feature, feature))[0])
    selected_feature = ori_all_feature[id_k]
    data = {}
    for k in matched_data.keys():
        data[k] = matched_data[k].loc[:, selected_feature]

    predResult, perM, perf, dropout = LSTM_Classification_Analysis(data=data, summary=sum_case, sampleID=sampleID,
                                                                   resultFolder=resultFolder, para=para)

    result = {}
    result['predResult'] = predResult
    result['perM'] = perM
    result['perf'] = perf
    result['dropout'] = dropout
    result['feature'] = feature

    return result


