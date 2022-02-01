# Python_IA
Repositório com as atividades e projetos da bolsa de Python em Inteligência Artificial na Dell Lead 

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

##### Extra: crie scatterplots com os dados clusterizados. Considere fazer gráficos dois-a-dois (use a cor para representar gênero): 

Annual Income (k$) x Spending Score (1-100)

Annual Income (k$) x Age

Age x Spending Score (1-100)

### Parte 2: Redução de dimensionalidade
Considere o conjunto de dados MNIST utilizado na Atividade 6. Realize a redução de dimensionalidade dos dados com o algoritmo PCA. Escolha um algoritmo de classificação e aplique ele, comparando os resultados com e sem a redução de dimensionalidade. A quantidade de dimensões utilizadas é de sua escolha. Discuta o resultado.

##### Extra: faça o plot de imagens do dataset após a redução de dimensionalidade.

# Atividade 8:
Nessa atividade vocês irão trabalhar em um problema de classificação de texto multiclasse. Considere o conjunto de dados sobre fetch_20newsgroups

"The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. To the best of our knowledge, it was originally collected by Ken Lang, probably for his paper “Newsweeder: Learning to filter netnews,” though he does not explicitly mention this collection. The 20 newsgroups collection has become a popular data set for experiments in text applications of machine learning techniques, such as text classification and text clustering."

Esse conjunto de dados pode ser carregado através:
scikit-learn from sklearn.datasets import fetch_20newsgroups 

twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)

Dado esse contexto, escolha um único classificador, sem otimizar hiperparametros, treine e teste modelos considerando

1. Bag of Words (contagem), sem pré-processamento
2. TF-IDF, sem pré-processamento
3. Bag of Words, com pré-processamento
4. TF-IDF, com pré-processamento
5. Considere a métrica da acurácia e compare os resultados em uma tabela.

As etapas de pré-processamento devem conter pelo menos:
1. lowercase
2. remoção de pontuação
3. remoção de números
4. remoção de stopwords (dica: utilize a biblioteca NLTK)
5. lematização ou stemming (apenas um dos dois)

Outras etapas que você julgar necessárias podem ser utilizadas. Crie uma função para cada etapa e uma função chamada preprocess() que chame todas as etapas.

# Trabalho Final:
Neste trabalho final você irá pôr em prática todas as etapas exercitadas ao longo da mentoria sobre projetos de aprendizado de máquina em uma tarefa de classificação de NLP: detecção de sarcasmo. Segundo Yaghoobian et al:

“Sarcasm detection is the task of identifying irony containing utterances in sentiment-bearing text. However, the figurative and creative nature of sarcasm poses a great challenge for affective computing systems performing sentiment analysis.”

Detecção de sarcasmo é uma tarefa com muitas aplicações práticas interessantes,  mas também é extremamente desafiadora. Padrões linguísticos e sociais que categorizam o sarcasmo podem não estar contido unicamente no texto ou no uso de palavras específicas, dificultando que algoritmos de aprendizado de máquina aprendam a generalizar. Nesse contexto, foi proposto por Khodak et al o dataset SARC, coletado da plataforma Reddit:


“We introduce the Self-Annotated Reddit Corpus (SARC), a large corpus for sarcasm research and for training and evaluating systems for sarcasm detection. The corpus has 1.3 million sarcastic statements -- 10 times more than any previous dataset -- and many times more instances of non-sarcastic statements, allowing for learning in both balanced and unbalanced label regimes. Each statement is furthermore self-annotated -- sarcasm is labeled by the author, not an independent annotator -- and provided with user, topic, and conversation context.”


Em anexo, encontra-se uma amostra do SARC. Dado esse contexto, você deverá criar e validar um modelo de detecção de sarcasmo utilizando esse dataset. O trabalho deverá conter:


1. Análise exploratória

    a. Mostre exemplos de cada classe.

    b. Crie pelo menos 5 gráficos, contendo um que mostre o balanceamento entre as classes. Sugestões de outros gráficos:    termos mais frequentes em cada classe, distribuição da quantidade de palavras em cada classe.

    c. Discuta seus achados.


2. Aprendizado não-supervisionado

    a. Qualquer técnica pode ser utilizada, clusterização ou redução de dimensionalidade. Pode-se usar parte da análise exploratória ou como auxiliar na classificação.


3. Limpeza e pré-processamento dos dados

    a.Pelo menos 3 técnicas de pré-processamento de texto (que já não sejam utilizadas por padrão na vetorização)

    b.Utilize alguma técnica de balanceamento de dados


4. Engenharia de atributos

    a. Selecione dentre os dados disponíveis quais devem ser utilizados como atributos de entrada e o método para representação vetorial.

    b. Utilize pelo menos dois conjuntos de atributos de entrada, comparando as performances. Justifique suas escolhas.


5. Estabelecimento de um baseline

    a.Utilize um modelo simples, não faça otimização de hiperparâmetros. Justifique sua escolha.


6. Seleção e avaliação de modelos

    a. Escolha uma ou mais métricas de desempenho apropriadas para esta tarefa.

    b. Utilize pelo menos 4 algoritmos, realizando a otimização de hiperparâmetros. Destes modelos, ao menos um deve ser um algoritmos deve ser um que não foi visto durante a mentoria. Estude o funcionamento dele. Sugestão: modelo de Deep Learning.

    c. Avalie os algoritmos com hiperparâmetros otimizados no conjunto de teste.


7. Análise de resultados

    a. Mostre exemplos dos erros (falso positivos e falso negativos) para o melhor dos algoritmos experimentados.

    b. Discuta os resultados e aponte quais seriam possíveis melhorias


8. (OPCIONAL) Interpretabilidade

    a. Utilize algum método que permita uma interpretação das predições de algum dos seus modelos.


9. (OPCIONAL) Deploy

    a. Implemente uma API para que o seu melhor modelo possa ser utilizado via requisições HTTP.
