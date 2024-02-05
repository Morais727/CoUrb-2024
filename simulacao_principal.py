import argparse
import flwr as fl
from functools import partial
from cliente_principal import ClienteFlower
import servidor_principal

def parse_args():
    parser = argparse.ArgumentParser(description="Flower Simulation")
    parser.add_argument("--total_clients", type=int, default=25, help="Total de clients") # SE DADOS NON-IID ==> 10,25,50,100
    parser.add_argument("--num_rounds", type=int, default=20, help="Quantidade de rounds") 
    parser.add_argument("--fraction_fit", type=float, default=1.0, help="Fração de clientes que serao trienados durante o round")
    parser.add_argument("--modelo_definido", type=str, default='DNN', help="Modelo que sera treinado durante o round") # DNN OU CNN
    parser.add_argument("--iid_niid", type=str, default='IID', help="Define se dados IID OU NON-IID")
    parser.add_argument("--modo_ataque", type=str, default='ZEROS', help="Define o modelo de ataque que sera utilizado") # ALTERNA_INICIO, ATACANTES, EMBARALHA, INVERTE_TREINANDO, INVERTE_SEM_TREINAR, INVERTE_CONVEGENCIA, ZEROS, RUIDO_GAUSSIANO, NORMAL
    parser.add_argument("--dataset", type=str, default='MNIST', help="Define o dataset que sera utilizado") # MNIST OU CIFAR10
    parser.add_argument("--variavel", type=int, default=2, help="Define o alvo do ataque") # Quantidade de atacantes, início do ataque, ruído aplicado (Seu referencial de ruído será dividido por 100)

    return parser.parse_args()

def main():
    args = parse_args()
    
    modelo_definido = args.modelo_definido 
    iid_niid = args.iid_niid  
    modo_ataque = args.modo_ataque   
    dataset = args.dataset 
    total_clients = args.total_clients 
    num_rounds = args.num_rounds
    fraction_fit = args.fraction_fit
    variavel= args.variavel

    parametros = [0] * total_clients
    parametros[:variavel] = [1] * variavel
    tamanho = parametros.count(1)
    
    def Cliente(cid, parametros, modelo_definido, iid_niid, modo_ataque, dataset, total_clients, tamanho):
        return ClienteFlower(cid, parametros, modelo_definido, iid_niid, modo_ataque, dataset, total_clients, tamanho)

    client_fn = partial(Cliente, parametros=parametros, modelo_definido=modelo_definido, iid_niid=iid_niid,
                        modo_ataque=modo_ataque, dataset=dataset, total_clients=total_clients, tamanho=tamanho)

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=total_clients,
        strategy=servidor_principal.Timming(fraction_fit=fraction_fit),
        config=fl.server.ServerConfig(num_rounds=num_rounds)
    )

if __name__ == "__main__":
    main()
