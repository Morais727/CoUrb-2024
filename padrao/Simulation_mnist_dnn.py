import flwr as fl
import tensorflow as tf
import servidor_padrão
from cliente_padrão import ClienteFlower

def Cliente(cid):
	return ClienteFlower(cid)
history= fl.simulation.start_simulation(client_fn=Cliente,
										num_clients=25,	
										strategy = servidor_padrão.Timming(fraction_fit = 1.0),									
										config=fl.server.ServerConfig(num_rounds=20))
