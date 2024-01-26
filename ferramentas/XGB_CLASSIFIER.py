import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

arquivos =  [
                'datasets_brutos/Balanceado_cifar_cnn.csv',
                'datasets_brutos/Balanceado_mnist_dnn.csv'
            ]
erros = []
for arquivo in arquivos:
    try:
        # Carrega o arquivo CSV                                             
        data = pd.read_csv(arquivo)
        data = data.sample(frac=0.5)

        labels = data.iloc[:, -1]
        selected_feature = data.iloc[:, :-1].values
        

        # Normalizando
        minmax                = MinMaxScaler()
        minmax.fit(selected_feature)
        selected_feature_norm = minmax.transform(selected_feature)

        arquivo = arquivo.split('.')
        base = arquivo[0].split('_')
        
        
        nome = (f'MODELOS/MINMAX_XGB_{base[2]}_{base[3]}.pkl')

        with open(nome, 'wb') as min:
            pickle.dump(minmax, min)

        X_train, X_test, y_train, y_test = train_test_split(selected_feature_norm, labels, test_size=0.2)

        # Cria e treina um classificador XGBoost
        xgb_classifier = XGBClassifier(max_depth=25)
        xgb_classifier.fit(X_train, y_train)

        nome_arquivo = (f'MODELOS/CLASSIFICADOR_XGB_{base[2]}_{base[3]}.model')
        xgb_classifier.save_model(nome_arquivo)

        # Faz previsões no conjunto de teste
        y_pred = xgb_classifier.predict(X_test)

        # Calcula a acurácia das previsões
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Acurácia: {accuracy}')


        
    except Exception as e:
        # Captura e lida com a exceção
        erros.append(f'ERRO ao processar o arquivo {arquivo}: {str(e)}')

print(erros)