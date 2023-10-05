import numpy as np
import xarray as xr
import pandas as pd
import copy
from datetime import datetime, timedelta
from keras.utils import to_categorical
# import visualkeras
import tensorflow as tf
from model_builders import *
from sklearn.metrics import balanced_accuracy_score
import optuna
from optuna.samplers import TPESampler
import keras
from keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import sys
import os
import joblib

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

## GLOBAL SEED ##    
np.random.seed(42)
tf.random.set_seed(42)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def create_tf_datasets(input_data, output_data):
    # Convert xarray dataset to numpy array for TensorFlow Dataset
    input_images = input_data.transpose('time', 'lat', 'lon','channel').values
    output_one_hot = output_data.values

    # Create TensorFlow Datasets
    input_dataset = tf.data.Dataset.from_tensor_slices(input_images)
    output_dataset = tf.data.Dataset.from_tensor_slices(output_one_hot)

    # Combine input and output datasets into a joint dataset
    joint_dataset = tf.data.Dataset.zip((input_dataset, output_dataset))

    return joint_dataset

def create_datasets(input_anoms, var_name, df_shifts, week_out):
# Assuming you have the xarray.Dataset 'input_data' and the pandas.Series 'output_data'
    input_data = copy.deepcopy(input_anoms[var_name])

    array_temp = input_data.data
    array_temp[np.isfinite(array_temp)==False]=0
    input_data.data = array_temp

    input_data = (input_data - input_data.mean('time')) / (input_data.std('time'))
    # Reshape the data to add a new dimension
    values_reshaped = input_data.values.reshape(input_data.shape[0], input_data.shape[1], input_data.shape[2], 1)

    # Create a new xarray.DataArray with the reshaped data and the original coordinates
    input_data = xr.DataArray(values_reshaped, coords=input_data.coords, dims=('time', 'lat', 'lon', 'channel'))
    output_data = copy.deepcopy(df_shifts[f'week{week_out}']).dropna()

    # Step 1: Create a common date index that includes all dates in both the input and output data
    common_dates = np.intersect1d(input_data['time'].values, output_data.index)

    # Step 2: Reindex the input xarray dataset and the output DataFrame to the common date index
    input_data = input_data.sel(time=common_dates)
    output_data = output_data.loc[common_dates]

    # Step 3: One-hot encode the output DataFrame using to_categorical
    num_classes = len(output_data.unique())  # Number of classes (number of weeks in this case)
    output_data_encoded = to_categorical(output_data, num_classes=num_classes)
    output_data_encoded = pd.DataFrame(output_data_encoded,index=output_data.index)

    # Step 4: Create masks for training, validation, and testing periods
    train_mask = (output_data.index >= '1980-01-01') & (output_data.index <= '2010-12-31')
    val_mask = (output_data.index >= '2011-01-01') & (output_data.index <= '2015-12-31')
    test_mask = (output_data.index >= '2016-01-01') & (output_data.index <= '2020-12-31')

    # Step 5: Split the input xarray dataset and the output DataFrame into subsets
    input_train = input_data.sel(time=train_mask)
    input_val = input_data.sel(time=val_mask)
    input_test = input_data.sel(time=test_mask)

    output_train = output_data_encoded.loc[train_mask]
    output_val = output_data_encoded.loc[val_mask]
    output_test = output_data_encoded.loc[test_mask]

    train_joint_dataset = create_tf_datasets(input_train, output_train)
    val_joint_dataset = create_tf_datasets(input_val, output_val)
    test_joint_dataset = create_tf_datasets(input_test, output_test)

    buffer_size = train_joint_dataset.cardinality()
    train_joint_dataset = train_joint_dataset.shuffle(buffer_size)
    return train_joint_dataset, val_joint_dataset, test_joint_dataset

def get_output_from_dataset(dataset):
    output_array = []
    for input_data, output_data in dataset.as_numpy_iterator():
        output_array.append(output_data)

    # Convert the list of NumPy arrays into a single NumPy array
    output_array = np.array(output_array)
    return output_array

def balanced_accuracy(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)
    return tf.py_function(balanced_accuracy_score, (y_true, y_pred), tf.float32)

def logging_callback(study, frozen_trial):
    previous_best_value = study.user_attrs.get("previous_best_value", None)
    if previous_best_value != study.best_value:
        study.set_user_attr("previous_best_value", study.best_value)
        print(
            "Trial {} finished with best value: {} and parameters: {}. ".format(
            frozen_trial.number,
            frozen_trial.value,
            frozen_trial.params,
            )
        )
        
        
class Objective(object):
    def __init__(self, train_joint_dataset, val_joint_dataset, test_joint_dataset,
                 path_models, variable, week):
        self.train_joint_dataset = train_joint_dataset
        self.val_joint_dataset = val_joint_dataset
        self.test_joint_dataset = test_joint_dataset
        self.path_models = path_models
        self.variable = variable
        self.week = week
 
    def __call__(self, trial):    
        keras.backend.clear_session()
        
        model_base = trial.suggest_categorical('model_base',['vanilla','resnet50','resnet101',\
                                                             'inception','xception','densenet'])
        ks = trial.suggest_categorical('ks',[3,5,7,9,11])
        ps = trial.suggest_categorical('ps',[2,4,6,8])
        type_pooling = trial.suggest_categorical('type_pooling',[None, 'avg','max'])
        stc = trial.suggest_categorical('stc',[1,2,3,4])
        stp = trial.suggest_categorical('stp',[1,2,3,4])
        do = trial.suggest_categorical('do',[0.3,0.4,0.5])
        md = trial.suggest_categorical('md',[2,4,8,16])
        nfilters = trial.suggest_categorical('nfilters',[4,8,16,32])
        activation = trial.suggest_categorical('activation',['LeakyReLU','ReLU'])
        weighted_loss = trial.suggest_categorical('weighted_loss',[True,False])
        
        dict_params = {'model_base':model_base,
                       'ks':ks,
                       'ps':ps,
                       'type_pooling':type_pooling,
                       'stc':stc,
                       'stp':stp,
                       'do':do,
                       'md':md,
                       'nfilters':nfilters,
                       'activation':activation,
                       'weighted_loss':weighted_loss}
        print(dict_params)                                      
        # instantiate and compile model
        if dict_params['model_base']=='vanilla':
            model = build_vanilla_cnn(dict_params['ks'],
                                      dict_params['ps'],
                                      dict_params['type_pooling'],
                                      dict_params['stc'],
                                      dict_params['stp'],
                                      dict_params['do'],
                                      dict_params['md'],
                                      dict_params['nfilters'],
                                      dict_params['activation'])
        elif dict_params['model_base']=='resnet50':
            model = build_resnet50_model(dict_params['type_pooling'],
                                         dict_params['do'],
                                         dict_params['md'],
                                         dict_params['activation'])
        elif dict_params['model_base']=='resnet101':
            model = build_resnet101_model(dict_params['type_pooling'],
                                         dict_params['do'],
                                         dict_params['md'],
                                         dict_params['activation'])
        elif dict_params['model_base']=='inception':
            model = build_inception_model(dict_params['type_pooling'],
                                         dict_params['do'],
                                         dict_params['md'],
                                         dict_params['activation'])
        elif dict_params['model_base']=='xception':
            model = build_xception_model(dict_params['type_pooling'],
                                         dict_params['do'],
                                         dict_params['md'],
                                         dict_params['activation'])
        elif dict_params['model_base']=='densenet':
            model = build_densenet_model(dict_params['type_pooling'],
                                         dict_params['do'],
                                         dict_params['md'],
                                         dict_params['activation'])
            
        model.compile(loss=keras.losses.categorical_crossentropy, 
                optimizer=keras.optimizers.Adam(lr=0.0001),metrics=[balanced_accuracy,'accuracy'])
        
        epochs = 100
        early_stopping_patience = 5

        # Create the EarlyStopping callback
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_balanced_accuracy',  # Metric to monitor
            patience=early_stopping_patience,  # Number of epochs with no improvement
            restore_best_weights=True  # Restore the weights of the best model
        )

        # Train the model with early stopping
        try:
            os.mkdir(f'{self.path_models}{self.variable}')
        except: pass
    
        filepath = f'{self.path_models}{self.variable}/model_{self.week}_v0.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, 
                                     mode='auto',save_weights_only=False)
        
        if dict_params['weighted_loss']==True:
            
            y_train = get_output_from_dataset(self.train_joint_dataset)
            y_train_integers = np.argmax(y_train, axis=1)
            class_weights = compute_class_weight(class_weight='balanced',classes=np.unique(y_train_integers),
                                                 y = y_train_integers)
            d_class_weights = dict(enumerate(class_weights))
            
            history = model.fit(
                self.train_joint_dataset.batch(32),
                validation_data=self.val_joint_dataset.batch(32),
                class_weight = d_class_weights,
                epochs=epochs,
                callbacks=[checkpoint,early_stopping_callback],
                verbose=0
            )
        else:
            history = model.fit(
                self.train_joint_dataset.batch(32),
                validation_data=self.val_joint_dataset.batch(32),
                epochs=epochs,
                callbacks=[checkpoint,early_stopping_callback],
                verbose=0
            )
        
        test_loss, test_balanced_accuracy, test_accuracy = model.evaluate(self.test_joint_dataset.batch(32))
        val_balanced_accuracy = np.max(history.history['val_balanced_accuracy'])
        val_accuracy = np.max(history.history['val_accuracy'])
        
        trial.set_user_attr('test_balanced_accuracy',test_balanced_accuracy)
        trial.set_user_attr('test_accuracy',test_accuracy)
        trial.set_user_attr('val_balanced_accuracy',val_balanced_accuracy)
        trial.set_user_attr('val_accuracy',val_accuracy)
        
        return val_balanced_accuracy
    
    
# Get variable and origin from command-line arguments
name_var = sys.argv[1]
week_out = sys.argv[2]
path_weekly_anoms = '/glade/scratch/jhayron/Data4Predictability/WeeklyAnoms/'
input_anoms = xr.open_dataset(f'{path_weekly_anoms}{name_var}.nc')
var_name = list(input_anoms.data_vars.keys())[0]
week_out_str = f'week{week_out}'

wr_series = pd.read_csv('/glade/work/jhayron/Data4Predictability/WR_Series.csv',\
                index_col=0,names=['week0'],skiprows=1,parse_dates=True)
for wk in range(2,10):
    series_temp = copy.deepcopy(wr_series["week0"])
    series_temp.index = series_temp.index - timedelta(weeks = wk-1)
    series_temp.name = f'week{wk-1}'
    if wk==2:
        df_shifts = pd.concat([pd.DataFrame(wr_series["week0"]),pd.DataFrame(series_temp)],axis=1)  
    else:
        df_shifts = pd.concat([df_shifts,pd.DataFrame(series_temp)],axis=1)
        
train_joint_dataset, val_joint_dataset, test_joint_dataset = \
    create_datasets(input_anoms, var_name, df_shifts, week_out)
path_models = '/glade/work/jhayron/Data4Predictability/models/CNN/v0/'
optimizer_direction = 'maximize'
number_of_random_points = 30  # random searches to start opt process
maximum_time = 0.12*60*60  # seconds
objective = Objective(train_joint_dataset,val_joint_dataset,test_joint_dataset,
                      path_models,name_var,week_out_str)
    
results_directory = f'/glade/work/jhayron/Data4Predictability/models/CNN/results_optuna/{week_out_str}/'
try:
    os.mkdir(results_directory)
except:
    pass

study_name = f'study_{name_var}_{week_out_str}_v0'
storage_name = f'sqlite:///{study_name}.db'


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction=optimizer_direction,
        sampler=TPESampler(n_startup_trials=number_of_random_points),
        study_name=study_name, storage=storage_name,load_if_exists=True)

study.optimize(objective, timeout=maximum_time, gc_after_trial=True,callbacks=[logging_callback],)

# save results
df_results = study.trials_dataframe()
df_results.to_pickle(results_directory + f'df_optuna_results_{name_var}_v0.pkl')
df_results.to_csv(results_directory + f'df_optuna_results_{name_var}_v0.csv')
#save study
joblib.dump(study, results_directory + f'optuna_study_{name_var}_v0.pkl')