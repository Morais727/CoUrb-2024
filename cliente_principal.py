import os
import sys
import time
import random
import pickle
import flwr as fl
import numpy as np
import tensorflow as tf


class ClienteFlower(fl.client.NumPyClient):
    def __init__(self,cid, modelo_definido, iid_niid, modo_ataque, dataset, 
                 total_clients, alpha_dirichlet,noise_gaussiano, round_inicio, 
                 per_cents_atacantes):  
        self.modelo_definido = str(modelo_definido)
        self.iid_niid = str(iid_niid)
        self.modo_ataque = str(modo_ataque)
        self.dataset = str(dataset)
        self.total_clients = int(total_clients)
        self.alpha_dirichlet = alpha_dirichlet
        self.noise_gaussiano = noise_gaussiano
        self.round_inicio = round_inicio
        self.per_cents_atacantes = int((int(total_clients) * per_cents_atacantes)/100)
        self.atacantes = per_cents_atacantes

        self.cid = int(cid)
        self.modelo = self.cria_modelo()
        self.x_treino, self.y_treino, self.x_teste, self.y_teste = self.load_data()

    def cria_modelo(self):
        modelo = tf.keras.models.Sequential()

        if self.modelo_definido =='DNN':
            modelo.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
            modelo.add(tf.keras.layers.Dense(128, activation='relu'))
            modelo.add(tf.keras.layers.Dense(64, activation='relu'))
            modelo.add(tf.keras.layers.Dense(10, activation='softmax'))

        else:
            modelo.add(tf.keras.layers.Conv2D(32,(3,3), activation= 'relu', input_shape=(32, 32, 3))) 
            modelo.add(tf.keras.layers.MaxPooling2D(2,2))
            modelo.add(tf.keras.layers.Conv2D(64,(3,3), activation='relu'))
            modelo.add(tf.keras.layers.MaxPooling2D(2,2))
            modelo.add(tf.keras.layers.Conv2D(64,(3,3), activation='relu'))
            modelo.add(tf.keras.layers.Flatten())
            modelo.add(tf.keras.layers.Dense(64, activation='relu'))
            modelo.add(tf.keras.layers.Dense(10, activation='softmax'))
        
        modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return modelo
    
    def load_data(self):
        n_clients = self.total_clients
        alpha_dirichlet = self.alpha_dirichlet
        
        if self.dataset == 'MNIST':  
            (x_treino, y_treino), (x_teste, y_teste) = tf.keras.datasets.mnist.load_data()
        else:
            (x_treino, y_treino), (x_teste, y_teste) = tf.keras.datasets.cifar10.load_data()
            
        x_treino, x_teste = x_treino/255.0, x_teste/255.0                                            
        
        if self.iid_niid== 'IID':        
            x_treino,y_treino,x_teste,y_teste = self.split_dataset(x_treino,y_treino,x_teste,y_teste, n_clients) 

        elif self.iid_niid== 'NIID':             
           x_treino,y_treino,x_teste,y_teste = self.split_dataset_dirichlet(x_treino,y_treino,x_teste,y_teste, n_clients, alpha_dirichlet)

        return x_treino, y_treino, x_teste, y_teste

    def split_dataset_dirichlet(self,x_train, y_train, x_test, y_test, n_clients, alpha_dirichlet):
        n_train_samples = len(x_train)
        n_test_samples = len(x_test)
        
        class_proportions = np.random.dirichlet(alpha_dirichlet, n_clients)
        
        train_samples_per_client = (class_proportions * n_train_samples).astype(int)
        test_samples_per_client = (class_proportions * n_test_samples).astype(int)
        
        x_train_clients = []
        y_train_clients = []
        x_test_clients = []
        y_test_clients = []
        
        start_train_idx = 0
        start_test_idx = 0
        
        for i in range(n_clients):
            end_train_idx = start_train_idx + train_samples_per_client[i]
            end_test_idx = start_test_idx + test_samples_per_client[i]
            
            x_train_clients.append(x_train[start_train_idx:end_train_idx])
            y_train_clients.append(y_train[start_train_idx:end_train_idx])
            
            x_test_clients.append(x_test[start_test_idx:end_test_idx])
            y_test_clients.append(y_test[start_test_idx:end_test_idx])
            
            start_train_idx = end_train_idx
            start_test_idx = end_test_idx
        
        return x_train_clients, y_train_clients, x_test_clients, y_test_clients
    
    def split_dataset(self, x_train, y_train, x_test, y_test, n_clients):
        p_train = int(len(x_train)/n_clients)
        p_test  = int(len(x_test)/n_clients)

        random.seed(self.cid)
        selected_train = random.sample(range(len(x_train)), p_train)

        random.seed(self.cid)
        selected_test  = random.sample(range(len(x_test)), p_test)
        
        x_train  = x_train[selected_train]
        y_train  = y_train[selected_train]

        x_test   = x_test[selected_test]
        y_test   = y_test[selected_test]

        return x_train, y_train, x_test, y_test

    def get_parameters(self, config):

        return self.modelo.get_weights()
    
    def fit(self, parameters, config): 
        server_round = int(config['server_round'])
        modo= self.modo_ataque

        if self.modelo_definido == 'CNN':
            camada_alvo = 8
        elif self.modelo_definido == 'DNN':
            camada_alvo = 5

        if modo=='ALTERNA_INICIO' and server_round >= self.round_inicio and self.cid <= self.per_cents_atacantes: 
            situacao = 1
            self.modelo.set_weights(parameters)
            history = self.modelo.fit(self.x_treino, self.y_treino, epochs=1, verbose=2)
            
            accuracy = history.history["accuracy"][0]  
            loss = history.history["loss"][0] 
        
            a = self.modelo.get_weights()
            camada = random.randint(0,camada_alvo)
            shape_list = np.shape(a[camada]) 
            min_value = random.random()
            max_value = random.random()
            
            a[camada] = min_value + (max_value- min_value) * np.random.rand(*shape_list)    
                            
            return a, len(self.x_treino),{"accuracy": accuracy, "loss": loss, "situacao":situacao,"variavel":self.modelo_definido,"camada":camada_alvo,"ataque":modo}
            
        elif modo=='ATACANTES' and server_round >= self.round_inicio and self.cid <= self.per_cents_atacantes: 
            situacao = 1
            self.modelo.set_weights(parameters)
            history = self.modelo.fit(self.x_treino, self.y_treino, epochs=1, verbose=2)
            accuracy = history.history["accuracy"][0]  
            loss = history.history["loss"][0] 
            a = self.modelo.get_weights()
            
            camada = random.randint(0,camada_alvo)

            shape_list = np.shape(a[camada])
        
            min_value = random.random()
            max_value = random.random()
            
            a[camada] = min_value + (max_value- min_value) * np.random.rand(*shape_list)
            
            return a, len(self.x_treino),{"accuracy": accuracy, "loss": loss, "situacao":situacao,'variavel':self.modelo_definido,'camada':camada_alvo}               
        
        elif modo=='EMBARALHA' and server_round >= self.round_inicio and self.cid <= self.per_cents_atacantes:
            situacao = 1 
            self.modelo.set_weights(parameters)

            history = self.modelo.fit(self.x_treino, self.y_treino, epochs=1, verbose=0)
            
            accuracy = history.history["accuracy"][0]  
            loss = history.history["loss"][0] 
            
            a = self.modelo.get_weights()
            for i in range(0,camada_alvo):
                camada = i

                shape_list = np.shape(a[camada])
            
                min_value = np.min(a[camada])
                max_value = np.max(a[camada])
                
                
                a[camada] = min_value + (max_value- min_value) * np.random.rand(*shape_list)
            
            return a, len(self.x_treino),{"accuracy": accuracy, "loss": loss, "situacao":situacao,'variavel':self.modelo_definido,'camada':camada_alvo}

        elif modo=='INVERTE_TREINANDO' and server_round >= self.round_inicio and self.cid <= self.per_cents_atacantes: 
            situacao = 1

            a = parameters                
            pesos_invertidos = [np.flipud(peso) for peso in a]                
            self.modelo.set_weights(pesos_invertidos)
            history = self.modelo.fit(self.x_treino, self.y_treino, epochs=1, verbose=0)
            accuracy = history.history["accuracy"][0]  
            loss = history.history["loss"][0] 
            
            return self.modelo.get_weights(), len(self.x_treino),{"accuracy": accuracy, "loss": loss, "situacao":situacao,'variavel':self.modelo_definido,'camada':camada_alvo}
        
        elif modo=='INVERTE_SEM_TREINAR' and server_round >= self.round_inicio and self.cid <= self.per_cents_atacantes:       
            situacao = 1
            a = self.modelo.get_weights()                
            pesos_invertidos = [np.flipud(peso) for peso in a]
            accuracy = 99.999
            loss = 0.001 
            
            return pesos_invertidos, len(self.x_treino),{"accuracy": accuracy, "loss": loss, "situacao":situacao,'variavel':self.modelo_definido,'camada':camada_alvo}

        elif modo=='INVERTE_CONVEGENCIA' and server_round >= self.round_inicio and self.cid <= self.per_cents_atacantes:
            situacao = 1
            pesos_locais = self.modelo.get_weights()
            self.modelo.set_weights(parameters)
            pesos_globais = self.modelo.get_weights()
            modelo_corrompido = []
            
            for i in range(camada_alvo):
                camada_corrompida = pesos_globais[i] + (pesos_globais[i] - pesos_locais[i])
                modelo_corrompido.append(camada_corrompida)
            
            self.modelo.set_weights(modelo_corrompido)            
            
            accuracy = 99.999
            loss = 0.001 
            return self.modelo.get_weights(), len(self.x_treino), {"accuracy": accuracy, "loss": loss, "situacao": situacao,"ataque":modo}

        elif modo== 'ZEROS' and server_round >= self.round_inicio and self.cid <= self.per_cents_atacantes:
            situacao = 1       		           
            a = parameters
            
            for i in range(camada_alvo):
                camada = i
                a[camada] = np.zeros_like(a[camada])
            self.modelo.set_weights(a)
    
            history = self.modelo.fit(self.x_treino, self.y_treino, epochs=1, verbose=2)
            accuracy = history.history["accuracy"][0]  
            loss = history.history["loss"][0]                           
      
            return a, len(self.x_treino),{"accuracy": accuracy, "loss": loss, "situacao":situacao,'variavel':self.modelo_definido,'camada':camada_alvo}
        
        elif modo=='RUIDO_GAUSSIANO':
            situacao = 1
            self.modelo.set_weights(parameters)
            history = self.modelo.fit(self.x_treino, self.y_treino, epochs=1, verbose=2)
            
            accuracy = history.history["accuracy"][0]  
            loss = history.history["loss"][0] 
        
            a = self.modelo.get_weights()
            camada = random.randint(0,camada_alvo)
            shape_list = np.shape(a[camada_alvo])

            noise = self.noise_gaussiano
            loc = float(self.cid) * np.random.uniform(1.5,2)
            a[camada_alvo] += np.random.normal(loc, noise, shape_list)

            return a, len(self.x_treino),{"accuracy": accuracy, "loss": loss, "situacao":situacao,'variavel':self.modelo_definido,'camada':camada_alvo}
            
        else:
            situacao = 0       
            self.modelo.set_weights(parameters)
            history = self.modelo.fit(self.x_treino, self.y_treino, epochs=1, verbose=0)                
            accuracy = history.history["accuracy"][0]  
            loss = history.history["loss"][0]              
            
            return self.modelo.get_weights(),len(self.x_treino),{"accuracy": accuracy, "loss": loss, "situacao":situacao,'variavel':self.modelo_definido,'camada':camada_alvo}                

    def evaluate(self, parameters, config):
        self.modelo.set_weights(parameters)
        loss, accuracy = self.modelo.evaluate(self.x_teste, self.y_teste, verbose=2)
        
        return loss, len(self.x_teste), {
                                            "accuracy": accuracy, 'porcentagem_ataque': int(self.atacantes),'modelo':self.modelo_definido,"ataque":self.modo_ataque,'iid_niid':self.iid_niid, 
                                            'dataset':self.dataset,'alpha_dirichlet':self.alpha_dirichlet,'noise_gaussiano':self.noise_gaussiano, 'round_inicio':self.round_inicio
                                         }
    

