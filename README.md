# Python_IA
Repositório com as atividades e projetos da mentoria de Python em Inteligência Artificial na Dell Lead 

# Atividade 1:

Revisamos métodos de listas e funções em Python

# Atividade 2:

Aplicamos métodos de análise de dados utilizando numpy e pands(leitura e manipulação de dataset)

# Atividade 3:

- Parte 1: Aplicando Regressão com Scikit-Learn e Numpy no dataset 'tips.csv'. Fazendo a predição, calculando os pesos, o erro quadrático médio e criando as visualizações(Sickit-Learn e Numpy) que mostra no mesmo plot o scatterplot entre a entrada e a saída e a linha de regressão aprendida pelo modelo.

- Parte 2: Considere o conjunto de dados 'california_housing_train.csv' que está no diretório Atividade 3. Os dados referem-se às casas encontradas em um determinado distrito da Califórnia e a algumas estatísticas resumidas sobre elas com base nos dados do censo de 1990. Considere que haja um problema de regressão no qual desejamos criar um modelo que faça a predição do valor mediano das casas de um distrito ('median_house_value') com base em outras informações. Crie pelo menos 3 modelos que façam essa predição utilizando mais de uma variável de entrada e compare a diferença de MSE. Qual o melhor modelo encontrado? Justifique a escolha das variáveis.

- Parte 3: Considerando o mesmo problema tratado na Parte 2, treine modelos de regressão utilizando transformações não-lineares dos atributos (x², x³, etc...). Pode-se utilizar transformações em um ou mais atributos. Treine pelo menos 3 modelos diferentes e faça o plot das curvas de regressão comparada com o scatterplot (análogo ao que foi feito na questão 5 da Parte 1). Calcule o MSE para cada um dos modelos. Qual modelo se ajustou melhor aos dados em termos de MSE?

- Parte 4: Considere o conjunto de dados utilizado nas partes 2 e 3 de preços de casas na Califórnia. Utilizando um método da validação cruzada (conjuntos de treino, validação e teste) experimente diferentes métodos de:

1. engenharia de atributos (seleção de atributos, atributos polinomiais, normalização)
2. escolha de hiperparâmetros (passo de aprendizagem, coeficiente de regularização, número de iterações)

      Treine no conjunto de treino e avalie no conjunto de validação. Avalie pelo menos 5 abordagens candidatas.

      Ao final, avalie como o seu melhor modelo performa no conjunto de teste e apresente o resultado.

      Dica¹: utilize as classes SGDRegressor ou Ridge do scikitlearn.

      Dica²: normalize os dados de entrada e saída

      Dica³: tente utilizar grid search e/ou k-fold

# Atividade 4:
Considere um conjunto de dados obtidos analisando imagens de câncer de mama, onde há o diagnóstico se o tumor é maligno ou benigno. Seu objetivo é desenvolver e avaliar modelos para realizar a classificação de câncer de mama em Maligno (1) ou Benigno (0). Esse modelo deve ter uma saída probabilística que indique a probabilidade de um tumor ser maligno. Você deverá cumprir os seguintes requerimentos:

1. Carregue o conjunto de dados Breast Cancer Wisconsin (Diagnostic) Data Set. Ele pode ser obtido através do sklearn.datasets, da UCI Machine Learning Repository ou do Kaggle.
2. Realize uma breve análise exploratória dos dados, criando ao menos 3 gráficos. Há desbalanceamento entre as classes? Extra: quais atributos são mais importantes para classificação?
3. Separe os dados em conjuntos de treino e teste, usando random_state = 42. Os dados de treino podem ser subdivididos em treino e validação de forma livre para ajustar hiperparâmetros.
4. Treine e avalie no conjunto de teste um modelo de regressão logística usando hiperparâmetros default e todos os dados de entrada, como baseline de desempenho. Lembre-se de normalizar os dados de entrada. As métricas de avaliação serão Acurácia, Precision, Recall, F1 score e AUC.
5. Treine pelo menos mais 3 modelos de regressão logística diferentes, treinando e avaliando nos conjuntos de treino e validação. Modelos diferentes incluem usar atributos diferentes, transformações não-lineares nos atributos (regressão logística polinomial) ou diferentes hiperparâmetros. Após isso, escolha o melhor desses modelos para retreinar com todos os dados de treino e avalie no conjunto de teste. Compare com o resultado com a baseline. Dica: pode-se utilizar o grid search.
6. Extra: utilize um modelo ainda não estudado na mentoria para realizar a classificação, avaliando no conjunto de teste. Compare os resultados.
