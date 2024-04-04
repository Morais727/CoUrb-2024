import argparse
import flwr as fl
from functools import partial
from cliente_principal import ClienteFlower
import servidor_principal_copy
def parse_args():
    parser = argparse.ArgumentParser(description="Flower Simulation")
    parser.add_argument("--total_clients", type=int, default=20, help="Total de clients") 
    parser.add_argument("--num_rounds", type=int, default=20, help="Quantidade de rounds") 
    parser.add_argument("--fraction_fit", type=float, default=1.0, help="Fração de clientes que serao trienados durante o round")
    parser.add_argument("--modelo_definido", type=str, default='DNN', help="Modelo que sera treinado durante o round") # DNN OU CNN
    parser.add_argument("--iid_niid", type=str, default='IID', help="Define se dados IID OU NON-IID")
    parser.add_argument("--modo_ataque", type=str, default='ZEROS', help="Define o modelo de ataque que sera utilizado") # ALTERNA_INICIO, ATACANTES, EMBARALHA, INVERTE_TREINANDO, INVERTE_SEM_TREINAR, INVERTE_CONVEGENCIA, ZEROS, RUIDO_GAUSSIANO, NORMAL
    parser.add_argument("--dataset", type=str, default='MNIST', help="Define o dataset que sera utilizado") # MNIST OU CIFAR10
    parser.add_argument("--alpha_dirichlet", type=float, nargs="+", default=[1], help="Define o alpha para NON-IID")
    parser.add_argument("--noise_gaussiano", type=float, default=0, help="Define o alpha para ruído gaussiano")#Usar números entre 0 e 1
    parser.add_argument("--round_inicio", type=int, default=0, help="Define o round de inicio do ataque")#Usar números inteiros
    parser.add_argument("--per_cents_atacantes", type=int, default=0, help="Define o percentual de atacantes")#Usar números inteiros
    

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
    alpha_dirichlet = args.alpha_dirichlet
    noise_gaussiano = args.noise_gaussiano
    round_inicio = args.round_inicio
    per_cents_atacantes = args.per_cents_atacantes
   

  
    try:
        def Cliente(cid, modelo_definido, iid_niid, modo_ataque, dataset, total_clients, alpha_dirichlet, noise_gaussiano,round_inicio, per_cents_atacantes):
            return ClienteFlower(cid, modelo_definido, iid_niid, modo_ataque, dataset, total_clients, alpha_dirichlet, noise_gaussiano,round_inicio, per_cents_atacantes)
        
        strategy=servidor_principal_copy.Timming(fraction_fit=fraction_fit)

        client_fn = partial(Cliente, modelo_definido=modelo_definido, iid_niid=iid_niid,
                            modo_ataque=modo_ataque, dataset=dataset, total_clients=total_clients,
                            alpha_dirichlet=alpha_dirichlet, noise_gaussiano=noise_gaussiano, round_inicio=round_inicio,per_cents_atacantes=per_cents_atacantes)

        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=total_clients,
            strategy=strategy,
            config=fl.server.ServerConfig(num_rounds=num_rounds)
        )
    except ValueError as e:
            print(f"Erro na inicialização do cliente federado: {e}")
if __name__ == "__main__":
    main()
