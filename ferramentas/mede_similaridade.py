import os
import glob
import numpy as np
from itertools import product
from sklearn.metrics.pairwise import cosine_similarity

caminhos = ['IID','NIID']
modelos = ['DNN','CNN']
ataques = ['ALTERNA_INICIO', 'ATACANTES', 'EMBARALHA', 'INVERTE_TREINANDO', 'INVERTE_SEM_TREINAR', 'INVERTE_CONVEGENCIA', 'ZEROS', 'RUIDO_GAUSSIANO', 'NORMAL']

for caminho, modelo, ataque in product(caminhos, modelos, ataques):
    gradientes_files = glob.glob(f"TESTES/{caminho}/GRADIENTES/{modelo}/{ataque}/gradiente_*.npy")

    conjunto_gradientes = []

    # Carregar o gradiente de referência TESTES\NIID\GRADIENTES\DNN\ALTERNA_INICIO\gradiente_19_20.npy
    gradiente_referencia = np.load(f"TESTES/{caminho}/GRADIENTES/{modelo}/{ataque}/gradiente_19_18.npy", allow_pickle=True)
    if gradiente_referencia is not None:
        print("Gradiente de referência carregado corretamente!")
    else:
        print("Erro ao carregar o gradiente de referência.")

    # Carregar os gradientes de cada conjunto
    for gradiente_file in gradientes_files:
        gradiente_file = gradiente_file.replace('\\', '/')
        gradientes_array = np.load(gradiente_file, allow_pickle=True)
        conjunto_gradientes.append(gradientes_array)

        # Verificar se o arquivo foi carregado corretamente
        if gradientes_array is not None:
            print(f"Arquivo '{gradiente_file}' foi carregado corretamente!")
        else:
            print(f"Erro ao carregar o arquivo '{gradiente_file}'.")

    # Calcular a similaridade entre cada gradiente e o gradiente de referência
    for i, gradientes_array in enumerate(conjunto_gradientes):
        similaridades = []
        for gradiente in gradientes_array:
            # Ajustar a forma do gradiente atual
            gradiente = gradiente.reshape(-1, 1)
            gradiente_referencia_reshaped = gradiente_referencia.reshape(-1, 1)
            
            # Calcular a similaridade do cosseno
            similarity = cosine_similarity(gradiente, gradiente_referencia_reshaped)
            similaridades.append(similarity[0][0])

        # Calcular a média da similaridade entre os gradientes do conjunto atual e o gradiente de referência
        media_similaridades = sum(similaridades) / len(similaridades)
        arquivo_similaridade = f"TESTES/{caminho}/GRADIENTES/{modelo}/{ataque}/similaridade_entre_gradiente.csv"
        os.makedirs(os.path.dirname(arquivo_similaridade), exist_ok=True) 
        with open(arquivo_similaridade, 'a') as file:
            file.write(str(media_similaridades) + '\n')
