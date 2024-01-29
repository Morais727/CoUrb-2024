import pickle
import flwr as fl
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

arquivos =  [
                'datasets_brutos/Balanceado_cifar_cnn.csv',
                'datasets_brutos/Balanceado_mnist_dnn.csv'
            ]

erros = []

for arquivo in arquivos:
    try:
        # Carrega o arquivo CSV
        data = pd.read_csv(arquivo)
        labels = data.iloc[:, -1].values
        selected_feature = data.iloc[:, :-1].values  # Seleciona todas as colunas exceto a última

        # Normalizando
        minmax = MinMaxScaler()
        minmax.fit(selected_feature)
        selected_feature_norm = minmax.transform(selected_feature)

        arquivo = arquivo.split('.')
        base = arquivo[0].split('_')

        nome = (f'MODELOS/MINMAX_XGB_{base[2]}_{base[3]}.pkl')

        with open(nome, 'wb') as arquivo:
            pickle.dump(minmax, arquivo)

        X_train, X_test, y_train, y_test = train_test_split(selected_feature_norm, labels, test_size=0.2)

        modelo = tf.keras.models.Sequential()
        modelo.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
        modelo.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        modelo.add(tf.keras.layers.Flatten())
        modelo.add(tf.keras.layers.Dense(128, activation='relu'))
        modelo.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        modelo.fit(X_train, y_train, epochs=2, batch_size=32, validation_split=0.2)

        nome_arquivo = (f'MODELOS/CLASSIFICADOR_XGB_{base[2]}_{base[3]}.h5')
        modelo.save(nome_arquivo)

        y_pred = (modelo.predict(X_test) > 0.5).astype('int32')
        accuracy = accuracy_score(y_test, y_pred)

        print(f'Acurácia: {accuracy}')

    except Exception as e:
        erros.append(f'ERRO ao processar {arquivo}: {str(e)}')
print(erros)
