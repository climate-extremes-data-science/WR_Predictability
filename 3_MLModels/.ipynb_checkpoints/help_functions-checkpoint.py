import tensorflow as tf
import copy
import xarray as xr
import numpy as np
import pandas as pd
from keras.utils import to_categorical

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

#     input_data = (input_data - input_data.mean('time')) / (input_data.std('time'))
    
#     input_data[np.isfinite(array_temp)==False] = 0
    
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