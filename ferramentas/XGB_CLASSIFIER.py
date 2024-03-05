import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

arquivos =  [
                'DADOS_BRUTOS/Balanceado_CNN.csv',
                'DADOS_BRUTOS/Balanceado_DNN.csv'
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
        
        
        nome = (f'MODELOS/MINMAX_XGB_{base[2]}.pkl')

        with open(nome, 'wb') as min:
            pickle.dump(minmax, min)

        X_train, X_test, y_train, y_test = train_test_split(selected_feature_norm, labels, test_size=0.2)

        # Cria e treina um classificador XGBoost
        xgb_classifier = XGBClassifier(max_depth=25)
        xgb_classifier.fit(X_train, y_train)

        nome_arquivo = (f'MODELOS/CLASSIFICADOR_XGB_{base[2]}.h5')
        xgb_classifier.get_booster().save_model(nome_arquivo)

        # Faz previsões no conjunto de teste
        y_pred = xgb_classifier.predict(X_test)

        # Calcula a acurácia das previsões
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Imprime as métricas
        print(f'Acurácia: {accuracy}')
        print(f'Precisão: {precision}')
        print(f'Recall: {recall}')
        print(f'F1-Score: {f1}')
        
    except Exception as e:
        # Captura e lida com a exceção
        erros.append(f'ERRO ao processar o arquivo {arquivo}: {str(e)}')

print(erros)