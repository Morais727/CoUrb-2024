import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Carregar os gradientes de cada conjunto
conjunto_gradientes = []
for i in range(1, 11):
    filename = f"TESTES/IID/GRADIENTES/gradiente_{i}.csv"
    gradientes_df = np.load(filename)  # Supondo que não haja cabeçalho no arquivo CSV
    conjunto_gradientes.append(gradientes_df)

# Calcular a similaridade entre os gradientes de cada conjunto
for i, gradientes_df in enumerate(conjunto_gradientes):
    similaridades = []
    for j in range(len(gradientes_df)):
        for k in range(j + 1, len(gradientes_df)):
            # Calcular a similaridade de cosseno entre os gradientes j e k
            gradiente_j = np.array(gradientes_df.iloc[j])
            gradiente_k = np.array(gradientes_df.iloc[k])
            similarity = cosine_similarity(gradiente_j.reshape(1, -1), gradiente_k.reshape(1, -1))
            similaridades.append(similarity[0][0])

    # Calcular a média da similaridade entre os gradientes do conjunto atual
    media_similaridades = sum(similaridades) / len(similaridades)
    print(f"Média da similaridade para o conjunto {i+1}: {media_similaridades}")
