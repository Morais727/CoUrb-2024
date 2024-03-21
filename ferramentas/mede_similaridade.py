import pandas as pd
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

# Carregar os gradientes de cada conjunto
conjunto_gradientes = []
for i in range(1, 11):  # Supondo que os gradientes são salvos em arquivos gradiente_1.csv, gradiente_2.csv, ..., gradiente_10.csv
    filename = f"TESTES/IID/GRADIENTES/gradiente_{i}.csv"
    gradientes_df = pd.read_csv(filename)
    gradientes_tensor = tf.convert_to_tensor(gradientes_df["gradiente"].values, dtype=tf.float32)
    conjunto_gradientes.append(gradientes_tensor)

# Calcular a similaridade entre os gradientes de cada conjunto
for i, gradientes_tensor in enumerate(conjunto_gradientes):
    similaridades = []
    for j in range(len(gradientes_tensor)):
        for k in range(j + 1, len(gradientes_tensor)):
            similarity = cosine_similarity(tf.reshape(gradientes_tensor[j], (1, -1)), tf.reshape(gradientes_tensor[k], (1, -1)))
            similaridades.append(similarity[0][0])

    # Calcular a média da similaridade entre os gradientes do conjunto atual
    media_similaridades = sum(similaridades) / len(similaridades)
    print(f"Média da similaridade para o conjunto {i+1}: {media_similaridades}")
