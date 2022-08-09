# Big Data Analytics com R e Microsoft Azure Machine Learning
# Projeto 2
# Construir um modelo de Machine Learning capaz de prever, 
# com base em novos dados, se a chama será extinta ou não ao usar um extintor de incêndio.


#### Carregando pacotes ####
library(readxl)
library(caTools)
library(dplyr)
library(corrplot)
# install.packages("faraway")
library(faraway)
# install.packages("ROCR")
library(ROCR)
library(ggplot2)
# install.packages('pROC')
library(pROC)
library(caret)

pairs(dataset)

#### Carregando o dataset pelo googlesheets ####
url <- 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSsz_NXkgvLT5LspnBLt62LVvneIDerWABd7-SNAUieDo5486Ek2CcoDOb6YRP9yEbuDsalMLoXVGQf/pub?output=csv'
dataset <- read.csv(url, dec = ",")
View(dataset)

#### Analisando o dataset ####
class(dataset)
summary(dataset)
dim(dataset)
str(dataset)
colnames(dataset)

#### Tratamento dos dados ####

# Verificando dados faltantes
any(is.na(dataset))
sum(is.na(dataset)) 

# Coluna Airflow possui 1632 (9,3%) valores zerados
dim(dataset[dataset$AIRFLOW == 0,])[1]
dim(dataset[dataset$AIRFLOW == 0,])[1] / dim(dataset)[1] *100

# Optei por remover esses valores pois não sabemos se é um erro na coleta dos dados
dataset <- dataset[dataset$AIRFLOW != 0,]
View(dataset)

# Criando uma coluna para identificar FUEL como Liquid Fuels ou LPG.
?transform
unique(dataset$FUEL)
dataset <- transform(dataset, TYPE = ifelse(FUEL=="lpg", "LPG", "LF"))

?relocate
dataset <- dataset %>% relocate(TYPE, .after = FUEL)
View(dataset)

# Convertendo a coluna FUEL, SIZE, STATUS e TYPE para fator
dataset$FUEL <- as.factor(as.numeric(as.factor(dataset$FUEL)))
dataset$TYPE <- as.factor(as.numeric(as.factor(dataset$TYPE)))
dataset$SIZE <- as.factor(dataset$SIZE)
dataset$STATUS <- as.factor(dataset$STATUS)
View(dataset)
str(dataset)

# Verificando a distribuição da variável dependente
# Dataset está bem equilibrado em relação a distribuição da variável dependente.
prop.table(table(dataset$STATUS))

#### Criando dataset para teste e treino ####
ind <- sample.split(Y = dataset$STATUS, SplitRatio = 0.7)

train <- dataset[ind,]
test <- dataset[!ind,]

# Verificando tamanho e distribuição da variável dependente em test e train
dim(train)
dim(test)

prop.table(table(train$STATUS))
prop.table(table(test$STATUS))
  
#### Criando o modelo de regressão logística ####
modelo <- glm(STATUS ~ ., data = train, family = binomial(link = "logit"))
summary(modelo)


#### Predição na base test ####
?predict
pred <- predict(modelo, newdata = test, type = "response")

#### Criação do gráfico ROC e AUC #####

ROCRpred <- prediction(pred, test$STATUS)

ROCRperf <- performance(ROCRpred, 'tpr','fpr')

plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0,1, by=0.1))

auc(test$STATUS, pred)

#### Confusion Matrix e Acurácia ####
# Acurácia de 88%, podemos considerar que o modelo teve um bom desempenho

tb <- table(round(pred), test$STATUS)

accuracy <- sum(diag(tb))/sum(tb)
accuracy

#### Feature Selection com Stepwise ####
# Ao utilizarmos o stepwise para selecionar as variáveis mais
# significativas ele não nos mostrou uma opção melhor

step(modelo, direction = 'both')


#### Boosted Logistic Regression ####
# Testando outro algoritmo que teve melhor desempenho
# Com acurácia de 90%

trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

fit <- train(STATUS ~ .,
             data = train,
             method = c("LogitBoost", "regLogistic", "plr")[1],
             trControl = trctrl)

fit

fit$finalModel

pred_boost <- predict(fit, newdata = test)

?confusionMatrix
confusionMatrix(pred_boost, test$STATUS, positive = "1")

#### Resultado Final ####
# Recomendamos a utilização do modelo preditivo Boosted Logistic Regression 
# teve uma acurácia de 90% superior ao modelo de Regressão Logística que foi de 88%
