import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Carregar os gradientes de cada conjunto
conjunto_gradientes = []
for i in range(1, 11):
    filename = f"TESTES/IID/GRADIENTES/DNN/gradiente_{i}.npy"
    gradientes_array = np.load(filename, allow_pickle=True)  # Permitir pickle ao carregar o arquivo numpy
    conjunto_gradientes.append(gradientes_array)

    # Verificar se o arquivo foi carregado corretamente
    if gradientes_array is not None:
        print(f"Arquivo 'gradiente_{i}.npy' foi carregado corretamente!")
    else:
        print(f"Erro ao carregar o arquivo 'gradiente_{i}.npy'.")

# Calcular a similaridade entre os gradientes de cada conjunto
for i, gradientes_array in enumerate(conjunto_gradientes):
    similaridades = []
    for j in range(len(gradientes_array)):
        for k in range(j + 1, len(gradientes_array)):
            # Calcular a similaridade de cosseno entre os gradientes j e k
            gradiente_j = gradientes_array[j]
            gradiente_k = gradientes_array[k]
            similarity = cosine_similarity(gradiente_j.reshape(1, -1), gradiente_k.reshape(1, -1))
            similaridades.append(similarity[0][0])

    # Calcular a média da similaridade entre os gradientes do conjunto atual
    media_similaridades = sum(similaridades) / len(similaridades)
    print(f"Média da similaridade para o conjunto {i+1}: {media_similaridades}")
