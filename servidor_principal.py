import pickle
import flwr as fl
import numpy as np
import os
import csv
import xgboost as xgb
from xgboost import Booster
import tensorflow as tf
from logging import WARNING
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple, Union
from collections import Counter
from flwr.common import (DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy import Strategy

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

# flake8: noqa: E501
class Timming(fl.server.strategy.FedAvg):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,

        modo_execucao
    ) -> None:
       
        super().__init__()
        self.percents = None
        self.resultados    = []  
        self.classificacao = {} 
        self.modo_execucao = modo_execucao           
    
        minmax_mnist_dnn_path = 'MODELOS/MINMAX_XGB_mnist_dnn.pkl'
        modelo_mnist_dnn_path = 'MODELOS/CLASSIFICADOR_XGB_DNN.h5'

        minmax_cifar10_cnn_path = 'MODELOS/MINMAX_XGB_cifar_cnn.pkl'
        modelo_cifar10_cnn_path = 'MODELOS/CLASSIFICADOR_XGB_CNN.h5'

        # Verifica se os arquivos existem antes de tentar carregá-los
        if os.path.exists(minmax_mnist_dnn_path) and os.path.exists(modelo_mnist_dnn_path):
            with open(minmax_mnist_dnn_path, 'rb') as file:
                self.minmax_dnn = pickle.load(file)
            self.loaded_model_dnn = Booster(model_file=modelo_mnist_dnn_path)
            # self.loaded_model_dnn = tf.keras.saving.load_model(modelo_mnist_dnn_path)
        else:
            print(f"Erro: Arquivos não encontrados - {minmax_mnist_dnn_path}, {modelo_mnist_dnn_path}")

        if os.path.exists(minmax_cifar10_cnn_path) and os.path.exists(modelo_cifar10_cnn_path):
            with open(minmax_cifar10_cnn_path, 'rb') as file:
                self.minmax_cnn = pickle.load(file)
            self.loaded_model_cnn = Booster(model_file=modelo_cifar10_cnn_path)
            # self.loaded_model_cnn = tf.keras.saving.load_model(modelo_cifar10_cnn_path)
        else:
            print(f"Erro: Arquivos não encontrados - {minmax_cifar10_cnn_path}, {modelo_cifar10_cnn_path}")


        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters        
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn        
   
    def __repr__(self) -> str:
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available
        clients."""
        self.num_clients = int(num_available_clients * self.fraction_fit)
        return max(self.num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None
        return initial_parameters
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        """Configure the next round of training."""         
        config = {
                    "server_round":server_round
                }
        fit_ins = FitIns(parameters, config)
         # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
       
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

       # Parameters and config
        config = {
                    "server_round":server_round
                }
        
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results:   List[Tuple[ClientProxy, FitRes]],
        failures:  List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        atual = [] 
        self.classificacao = {} 
        percents_atual = 0.0  
        self.verifica_acertos = [] 
        if server_round > 1:
            for client, fit_res in results:
                result   = parameters_to_ndarrays(fit_res.parameters)
                situacao = fit_res.metrics['situacao']
                modelo = fit_res.metrics['modelo']
                camadas = fit_res.metrics['camada']
                iid      = client.cid
                data     = []               
                normas   = []                                

                for i in range(camadas+1):
                    ultimo_modelo = self.modelo_anterior[i]
                    
                    result[i] = result[i].flatten()
                    ultimo_modelo = ultimo_modelo.flatten()

                    norm1 = np.linalg.norm (result[i], ord=1)
                    norm2 = np.linalg.norm (result[i], ord=2)
                    norm3 = np.power(np.sum(np.abs(result[i]) ** 3), 1/3)
                    
                    delta1 = norm1 - (np.linalg.norm (ultimo_modelo, ord=1))
                    delta2 = norm2 - (np.linalg.norm (ultimo_modelo, ord=2))
                    delta3 = norm3  - (np.power(np.sum(np.abs(ultimo_modelo) ** 3), 1/3))
                    
                    normas.extend([norm1,delta1,norm2,delta2,norm3,delta3])

                if self.modo_execucao == 1:
                    normas.extend([f'{situacao}\n'])
                    nome_arquivo = f"DADOS_BRUTOS/{modelo}/data.csv"
                    os.makedirs(os.path.dirname(nome_arquivo), exist_ok=True)
                    with open(nome_arquivo,'a') as file:
                        file.write(",".join(map(str, normas)))

                elif self.modo_execucao == 0:
                    data.append(normas)
                    
                    selected_feature = np.array(data)
                    
                    if modelo == "CNN":
                        minmax_selecionado = self.minmax_cnn
                        modelo_selecionado = self.loaded_model_cnn
                    elif modelo == "DNN":
                        minmax_selecionado = self.minmax_dnn
                        modelo_selecionado = self.loaded_model_dnn
                    else:
                        raise ValueError(f"Complemento do nome inválido: {modelo}")

                    # Acessar o minmax usando o objeto selecionado
                    normalizado = minmax_selecionado.transform(selected_feature)

                    predict = modelo_selecionado.predict(xgb.DMatrix(normalizado))
                    prev = (predict > 0.5).astype('int32')

                    # predict = modelo_selecionado.predict(normalizado)
                    # prev = (predict > 0.5).astype('int32')
                    
                    chaves = {
                                (0): 'n_atak',
                                (1): 'atak',
                            }
                    
                    chave = chaves.get(( int(prev[0])), 'atak')            
                    self.classificacao.setdefault(chave,[]).append(iid) 
            
                    if situacao == prev[0]:
                        self.resultados.append('Acertos')
                        atual.append('Acertos')
                    else:
                        self.resultados.append('Erros')
                        atual.append('Erros')
                
                    self.verifica_acertos.append((server_round,iid,situacao,prev[0]))
        if self.modo_execucao == 0:
            cont_atual = Counter(atual)
            tot = sum(cont_atual.values())
            contagem = Counter(self.resultados)
            total = sum(contagem.values())
            if cont_atual['Acertos'] > 0:
                percents_atual = (cont_atual['Acertos'] / tot) * 100
            if contagem['Acertos'] > 0:
                self.percents = (contagem['Acertos'] / total) * 100
            else:
                self.percents = 0
                percents_atual = 0
            
            print(f'\n\nround >>>>> {server_round}')
            print(f'Percentual de acertos atual: {percents_atual:.2f}%')
            print(f'Percentual de acertos geral: {self.percents:.2f}%  {contagem}')        
            print(f'{self.classificacao}\n\n')   
            
            if not results:
                return None, {}
            # Do not aggregate if there are failures and failures are not accepted
            if not self.accept_failures and failures:
                return None, {}

            # Converte resultados
            weights_results = []
            malicious = []
            for client, fit_res in results:
                if 'atak' in self.classificacao.keys() and server_round:
                    if client.cid in self.classificacao['atak']: 
                        malicious.append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)) 
                        
                    else:       
                        weights_results.append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))                                  
                else:
                    weights_results.append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))

            if weights_results == []:
                parameters_aggregated = ndarrays_to_parameters(self.modelo_anterior) 
            
        elif self.modo_execucao == 1:
            if not results:
                return None, {}
            # Do not aggregate if there are failures and failures are not accepted
            if not self.accept_failures and failures:
                return None, {}

            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]

        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results)) 
        self.modelo_anterior = parameters_to_ndarrays(parameters_aggregated)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
        
        return parameters_aggregated, metrics_aggregated 

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")
         
        if self.modo_execucao == 0:
            for client,eval_res in results:            
                nome_arquivo = f"TESTES/{eval_res.metrics['iid_niid']}/LOG_EVALUATE/{eval_res.metrics['ataque']}_{eval_res.metrics['dataset']}_{eval_res.metrics['modelo']}_{eval_res.metrics['porcentagem_ataque']}_{eval_res.metrics['alpha_dirichlet']}_{eval_res.metrics['noise_gaussiano']}_{eval_res.metrics['round_inicio']}.csv"
                os.makedirs(os.path.dirname(nome_arquivo), exist_ok=True)   
                with open(nome_arquivo,'a') as file:          
                    file.write(f"\n{server_round},{client.cid},{eval_res.metrics['accuracy']},{eval_res.loss}")
            
            arquivo_verifica_acertos = f"TESTES/{eval_res.metrics['iid_niid']}/LOG_ACERTOS/{eval_res.metrics['ataque']}_{eval_res.metrics['dataset']}_{eval_res.metrics['modelo']}_{eval_res.metrics['porcentagem_ataque']}_{eval_res.metrics['alpha_dirichlet']}_{eval_res.metrics['noise_gaussiano']}_{eval_res.metrics['round_inicio']}.csv"        
            os.makedirs(os.path.dirname(arquivo_verifica_acertos), exist_ok=True)
            with open(arquivo_verifica_acertos, 'a', newline='') as arquivo_csv:
                escritor_csv = csv.writer(arquivo_csv)
                for linha in self.verifica_acertos:
                    escritor_csv.writerow(linha)

        return loss_aggregated, metrics_aggregated            
    