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

# Atividade 5:
Nesta atividade iremos trabalhar com o problema de classificação de dígitos em imagens, utilizando o conhecido conjunto de dados MNIST.

O conjunto de dados é formado por imagens preto e branco de 28x28 pixels que representam dígitos escritos à mão, de 0 à 9. Cada pixel possui um valor entre 0 (totalmente preto) e 255 (totalmente branco). O desafio é treinar um modelo para reconhecer corretamente cada dígito representado. Ou seja, é um problema de classificação multi-classe de 10 classes (0, 1, 2, 3, 4, 5, 6, 7, 8, 9). O conjunto já é previamente dividido em treino e teste.

Dado esse problema, faça o que se pede:

1. Treine e avalie (no conjunto de teste) um classificador baseline com uma MLP de uma camada. Defina a quantidade de neurônios na camada oculta e argumente o motivo da sua escolha. Utilize métricas de desempenho para classificação multiclasse de sua escolha.

2. Crie uma arquitetura de MLP realizando a otimização de hiperparâmetros, usando uma técnica da sua escolha (grid search, random search). Os hiperparâmetros que serão otimizados são de sua escolha, mas devem conter o número de camadas e o número de neurônios em cada camada. Teste o melhor modelo encontrado no conjunto de teste e compare com o baseline.

OBS¹: A forma mais simples de extração de atributos de imagens é considerar cada pixel como um atributo. Para isso, basta transformar a matriz de cada imagem em um vetor unidimensional. Outras técnicas de extração de atributos baseadas em PDI (processamento digital de imagens) podem ser utilizadas.

OBS²: A MLP pode ser utilizada com a implementação do scikit-learn ou do keras.

# Atividade 6:
Criação de um modelo SVM para apresentação sobre o método.

# Atividade 7:
### Parte 1: Clusterização
Considere o conjunto de dados sobre clientes de um shopping disponibilizado em: 

https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv 

Esse conjunto de dados é composto pelas variáveis CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100). Suponha que o seu trabalho como analista de dados seja encontrar perfis de consumidores. 
Considerando as variáveis numéricas, faça a clusterização dos dados, encontrando a quantidade ótima de clusters. 
Mostre os valores dos atributos dos centróides de cada cluster. 
Que observações podem ser feitas sobre cada cluster? Ou seja, que tipo de cliente cada cluster representa. Discuta.

Extra: crie scatterplots com os dados clusterizados. Considere fazer gráficos dois-a-dois (use a cor para representar gênero): 
Annual Income (k$) x Spending Score (1-100)
Annual Income (k$) x Age
Age x Spending Score (1-100)

### Parte 2: Redução de dimensionalidade
Considere o conjunto de dados MNIST utilizado na Atividade 6. Realize a redução de dimensionalidade dos dados com o algoritmo PCA. Escolha um algoritmo de classificação e aplique ele, comparando os resultados com e sem a redução de dimensionalidade. A quantidade de dimensões utilizadas é de sua escolha. Discuta o resultado.

Extra: faça o plot de imagens do dataset após a redução de dimensionalidade.
