import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import glob

# Encontrar todos os caminhos de arquivos de gradientes disponíveis
gradientes_files = glob.glob("TESTES/IID/GRADIENTES/DNN/gradiente_*.npy")

# Lista para armazenar os gradientes de referência e os gradientes de cada conjunto
conjunto_gradientes = []

# Carregar o gradiente de referência
gradiente_referencia = np.load("TESTES/IID/GRADIENTES/DNN/gradiente_19_20.npy", allow_pickle=True)
if gradiente_referencia is not None:
    print("Gradiente de referência carregado corretamente!")
else:
    print("Erro ao carregar o gradiente de referência.")

# Carregar os gradientes de cada conjunto
for gradiente_file in gradientes_files:
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
        similarity = cosine_similarity(gradiente.reshape(1, -1), gradiente_referencia.reshape(1, -1))
        similaridades.append(similarity[0][0])

    # Calcular a média da similaridade entre os gradientes do conjunto atual e o gradiente de referência
    media_similaridades = sum(similaridades) / len(similaridades)
    print(f"Média da similaridade para o conjunto {i+1}: {media_similaridades}")
