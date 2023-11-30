import pandas as pd
import tensorflow as tf


def load_csv(path: str):
    data_scv = pd.read_csv(filepath_or_buffer=path)
    print(data_scv)
    data_scv.head()
    data_scv.tail()
    print(data_scv.dtypes)
    print(data_scv.deep_val)
    X_train = data_scv[['deep_val', 'm5_20_land_rate', 'm5_60_land_rate']]
    print(X_train)
    print('---------')
    print(data_scv.columns.values)
    x_train_df = data_scv[data_scv.columns.values[0:-1]]
    X_train = x_train_df.to_numpy()
    y_train = data_scv.dependentVal.to_numpy()
    print(X_train)
    return X_train, y_train, x_train_df


(X_train, y_train, x_train_df) = load_csv('./data/1701276691523.csv')
(X_test, y_test, x_test_df) = load_csv('./data/1701334932633_test.csv')

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=9, activation='relu', input_shape=(len(x_train_df.columns.values),)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units=64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units=32, activation='relu'))
model.add(tf.keras.layers.Dense(1))
optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
model.fit(X_train, y_train, epochs=15)
evaluate_info = model.evaluate(X_test, y_test)
print("evaluate_info>>>>>>>")
print(evaluate_info)
