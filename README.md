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
