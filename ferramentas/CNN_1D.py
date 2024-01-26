import pickle
import flwr as fl
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

arquivos = [
                'NIID/Balanceado_NIID_CIFAR_CNN.csv',
                'NIID/Balanceado_NIID_MNIST_DNN.csv',
                'IID/Balanceado_IID_CIFAR_CNN.csv',
                'IID/Balanceado_IID_MNIST_DNN.csv',
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

        caminho = arquivo.split('/')
        base = arquivo.split('_')
        extensao = base[3].split('.')

        nome = (f'{caminho[0]}/MINMAX_{base[1]}_{base[2]}_{extensao[0]}_1D')

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

        nome_modelo = (f'{caminho[0]}/MODELO_{base[1]}_{base[2]}_{extensao[0]}_1D.h5')
        modelo.save(nome_modelo)

        y_pred = (modelo.predict(X_test) > 0.5).astype('int32')
        accuracy = accuracy_score(y_test, y_pred)

        print(f'Acurácia: {accuracy}')

    except Exception as e:
        erros.append(f'ERRO ao processar {arquivo}: {str(e)}')
print(erros)
