import flwr as fl
import tensorflow as tf

class ClienteFlower(fl.client.NumPyClient):

	def __init__(self,cid):
		self.x_treino, self.y_treino, self.x_teste, self.y_teste = self.load_data()
		self.modelo = self.cria_modelo()
		self.cid = cid

	def cria_modelo(self):
		modelo = tf.keras.models.Sequential()
		modelo.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
		modelo.add(tf.keras.layers.Dense(128, activation='relu'))
		modelo.add(tf.keras.layers.Dense(64, activation='relu'))
		modelo.add(tf.keras.layers.Dense(10, activation='softmax'))

		modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
			
		return modelo

	def load_data(self):
		mnist = tf.keras.datasets.mnist
		(x_treino, y_treino), (x_teste, y_teste) = mnist.load_data()
		x_treino, x_teste = x_treino/255.0, x_teste/255.0
		return x_treino, y_treino, x_teste, y_teste

	def get_parameters(self, config):
		
		return self.modelo.get_weights()


	def fit(self, parameters, config):
		server_round = int(config["server_round"])
		print(config["server_round"])
		#atualiza modelo
		self.modelo.set_weights(parameters)
		#treina novo modelo
		history = self.modelo.fit(self.x_treino, self.y_treino, epochs=1, verbose=1)
		#retorna novos pesos
		accuracy = history.history["accuracy"][0]  
		loss = history.history["loss"][0]              
		
		return self.modelo.get_weights(),len(self.x_treino),{"accuracy": accuracy, "loss": loss}                



	def evaluate(self, parameters, config):
		#atualiza modelo
		self.modelo.set_weights(parameters)
		#avalia modelo
		loss, accuracy = self.modelo.evaluate(self.x_teste, self.y_teste)
		#retorna modelo
		return loss, len(self.x_teste), {"accuracy": accuracy}