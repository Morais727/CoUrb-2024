import numpy as np
import tensorflow as tf

# Carregar o conjunto de dados MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar os pixels para ter valores entre 0 e 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Definir o número de clientes
num_clients = 10

# Inicializar listas para armazenar os dados de cada cliente
client_data_images = [[] for _ in range(num_clients)]
client_data_labels = [[] for _ in range(num_clients)]

# Inicializar um conjunto vazio para armazenar as classes atribuídas a cada cliente
client_assigned_classes = [set() for _ in range(num_clients)]

# Inicializar um dicionário para contar as ocorrências de cada classe em cada cliente
class_counts = [{} for _ in range(num_clients)]

# Atribuir amostras do MNIST aos clientes de forma aleatória
for i in range(len(x_train)):
    label = y_train[i]
    client_index = np.random.randint(num_clients)  # Selecionar aleatoriamente o cliente para atribuir a amostra
    client_data_images[client_index].append(x_train[i])
    client_data_labels[client_index].append(label)
    client_assigned_classes[client_index].add(label)
    if label not in class_counts[client_index]:
        class_counts[client_index][label] = 0
    class_counts[client_index][label] += 1

# Converter as listas de dados de cada cliente em arrays numpy
client_data_images = [np.array(images) for images in client_data_images]
client_data_labels = [np.array(labels) for labels in client_data_labels]

# Calcular e exibir a quantidade de dados, classes e amostras por classe em cada cliente
for i in range(num_clients):
    num_data = len(client_data_images[i])
    num_classes = len(client_assigned_classes[i])
    print(f"Cliente {i}: Quantidade de Dados: {num_data}, Quantidade de Classes: {num_classes}")
    print(f"Quantidade de Amostras por Classe:")
    for class_label, count in class_counts[i].items():
        print(f"Classe {class_label}: {count} amostras")

# Exemplo de como acessar os dados de um cliente específico (por exemplo, cliente 0)
client_0_x, client_0_y = client_data_images[0], client_data_labels[0]
print("\nExemplo de dados para o Cliente 0:")
print("Formato dos dados de treinamento:", client_0_x.shape)
print("Formato dos rótulos de treinamento:", client_0_y.shape)
