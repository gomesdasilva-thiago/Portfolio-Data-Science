# Projeto 1
# construir um modelo de Machine Learning capaz de prever o consumo de energia de carros 
# elétricos com base em diversos fatores, tais como o tipo e número de motores elétricos do veículo, 
# o peso do veículo, a capacidade de carga, entre outros atributos.



#### Configurando o repositório ####

setwd('E:/DSAFormacaoCientistaDados/BigDataAnalyticsRAzureML/Projetos-1-2/Projeto01')
getwd()

#### Carregando pacotes ####
# install.packages('mlbench')
# install.packages('caret')
# install.packages('janitor')
# install.packages('car')
# install.packages('lmtest')


library(dplyr)
library(ggplot2)
library(mlbench)
library(caret)
library(randomForest)
library(janitor)
library(e1071)
library(car)
library(lmtest)
library(readxl)


#### Carregando dataset ####

dataset <- read_xlsx('FEV-data-Excel.xlsx')


#### Analisando o dataset ####
View(dataset)
class(dataset)
summary(dataset)     
str(dataset)
dim(dataset)
colnames(dataset)

#### Adicionando coluna de index e realocando em primeiro ####
dataset$index <- 1:nrow(dataset)

?relocate
dataset <- dataset %>% relocate(index, .before = 1 )

#### Limpando os nomes das colunas para evitar problemas ####
?clean_names
dataset <- clean_names(dataset, "lower_camel")


#### Identificando os dados faltantes ####
any(is.na(dataset))
sum(is.na(dataset)) 
# Como nossa base de dados é pequena e temos 30 valores NA's
# os valores serão atualizados com base nas informações encontradas no site ev-database.org 

# Adicionando coluna com a contagem de NA's por registro
dataset$countNA <- rowSums(is.na(dataset))
View(dataset)

# 11 carros possuem dados NA's
View(dataset %>% filter(countNA > 0))

# 9 carros com variável target mean - Energy consumption [kWh/100 km] faltando
View(dataset %>% filter(is.na(meanEnergyConsumptionKWh100Km)))

#### Atualizando dados NA's com base no site ev-database.org ####

# (index 10) Citroen e-C4 *** Vehicle Consumption 170 Wh/km *** (https://ev-database.org/car/1587/Citroen-e-C4)
dataset[10,"meanEnergyConsumptionKWh100Km"] = (170/1000) * 100 # conversão Wh/km > kWh e para 100km

# (index 30) Peugeot e-2008 *** Vehicle Consumption	184 Wh/km *** (https://ev-database.org/car/1206/Peugeot-e-2008-SUV)
dataset[30,"meanEnergyConsumptionKWh100Km"] = (184/1000) * 100 # conversão Wh/km > kWh e para 100km
dataset[30,24] = 8.5 # Acceleration 0-100 kph [s]
dataset[30,18] = 482 # Maximum load capacity [kg]
dataset[30,"permissableGrossWeightKg"] = 2030 # GVWH

# (index 40) Tesla Model 3 Standard Range Plus *** Vehicle Consumption	146 Wh/km *** (https://ev-database.org/car/1485/Tesla-Model-3-Standard-Range-Plus)
dataset[40,"meanEnergyConsumptionKWh100Km"] = (146/1000) * 100 # conversão Wh/km > kWh e para 100km
dataset[40,18] = 389 # Maximum load capacity [kg]
dataset[40,"permissableGrossWeightKg"] = 2014 # GVWH

# (index 41) Tesla Model 3 Long Range *** Vehicle Consumption	154 Wh/km *** (https://ev-database.org/car/1321/Tesla-Model-3-Long-Range-Dual-Motor)
dataset[41,"meanEnergyConsumptionKWh100Km"] = (154/1000) * 100 # conversão Wh/km > kWh e para 100km
dataset[41,18] = 388 # Maximum load capacity [kg]
dataset[41,"permissableGrossWeightKg"] = 2232 # GVWH

# (index 42) Tesla Model 3 Performance *** Vehicle Consumption	162 Wh/km *** (https://ev-database.org/car/1322/Tesla-Model-3-Performance)
dataset[42,"meanEnergyConsumptionKWh100Km"] = (162/1000) * 100 # conversão Wh/km > kWh e para 100km
dataset[42,18] = 388 # Maximum load capacity [kg]
dataset[42,"permissableGrossWeightKg"] = 2232 # GVWH

# (index 43) Tesla Model S Long Range Plus *** Vehicle Consumption 178 Wh/km *** (https://ev-database.org/car/1323/Tesla-Model-S-Long-Range-Plus)
dataset[43,"meanEnergyConsumptionKWh100Km"] = (178/1000) * 100 # conversão Wh/km > kWh e para 100km
dataset[43,"maximumLoadCapacityKg"] = 744 # https://www.evspecifications.com/en/model/3 # Trunk volume
dataset[43,"permissableGrossWeightKg"] = 2694 # https://www.evspecifications.com/en/model/3 # GVWH

# (index 44) Tesla Model S Performance *** Vehicle Consumption *	183 Wh/km *** (https://ev-database.org/car/1324/Tesla-Model-S-Performance)
dataset[44,"meanEnergyConsumptionKWh100Km"] = (183/1000) * 100 # conversão Wh/km > kWh e para 100km
dataset[44,"maximumLoadCapacityKg"] = 744 # https://www.evspecifications.com/en/model/3 # Trunk volume
dataset[44,"permissableGrossWeightKg"] = 2720 # https://www.evspecifications.com/en/model/3 # GVWH

# (index 45) Tesla Model X Long Range Plus *** Vehicle Consumption 204 Wh/km *** (https://ev-database.org/car/1325/Tesla-Model-X-Long-Range-Plus)
dataset[45,"meanEnergyConsumptionKWh100Km"] = (204/1000) * 100 # conversão Wh/km > kWh e para 100km
dataset[45,"maximumLoadCapacityKg"] = 357 # https://www.evspecifications.com/en/ # Trunk volume
dataset[45,"permissableGrossWeightKg"] = 3079 # https://www.evspecifications.com/en/model/3 # GVWH

# (index 46) Tesla Model X Performance *** Vehicle Consumption	221 Wh/km *** (https://ev-database.org/car/1208/Tesla-Model-X-Performance)
dataset[46,"meanEnergyConsumptionKWh100Km"] = (221/1000) * 100 # conversão Wh/km > kWh e para 100km
dataset[46,"maximumLoadCapacityKg"] = 357 # https://www.evspecifications.com/en/ # Trunk volume
dataset[46,"permissableGrossWeightKg"] = 3120 # https://www.evspecifications.com/en/model/3 # GVWH

# (index 52) Mercedes-Benz EQV (https://ev-database.org/car/1240/Mercedes-EQV-300-Long)
dataset[52,24] = 12.1 #Acceleration 0-100 kph [s]
dataset[52,"typeOfBrakes"] = "disc (front + rear)" # https://www.evspecifications.com/en/model/0a98172
dataset[52,"bootCapacityVdaL"] = mean(dataset$`bootCapacityVdaL`, na.rm = T) # Informação não encontrada, utilizado a média da coluna

# (index 53) Nissan e-NV200 evalia (https://ev-database.org/car/1117/Nissan-e-NV200-Evalia)
dataset[53,24] = 14 #Acceleration 0-100 kph [s]

# Verificando novamente valores nulos
any(is.na(dataset))
sum(is.na(dataset)) 

#### Convertendo as colunas chr em numérico ####
dataset$`typeOfBrakes` <- as.numeric(as.factor(dataset$`typeOfBrakes`))
dataset$`driveType` <- as.numeric(as.factor(dataset$`driveType`))
View(dataset)

# Analisando o dataset após tratamento dos valores nulos
class(dataset)
summary(dataset)     
str(dataset)
dim(dataset)
colnames(dataset)
View(dataset)

# Salvando o dataset tratado como csv para uso no Azure
write.csv(dataset, "dataset_final.csv", row.names = FALSE)

#### Analisando os dados graficamente ####
colNames <- colnames(dataset[,5:26])
colNames

# Histograma

histVar <- function(df, variables) {
  for (variable in variables) {
    hist(df[[variable]], main = paste("Histograma de" , variable))
  }
}


histVar(dataset, colNames)

# Outliers com boxplot

boxPlotGraf <- function(df, variables) {
  for (variable in variables) {
    boxplot(df[[variable]], main = paste("Boxplot de" , variable))
  }
}

boxPlotGraf(dataset, colNames)


#### Criando o modelo de regressão linear múltipla ####
# Como o dataset possui apenas 53 registros não optei por separar entre treino e teste
# mas sim treinar o modelo com o dataset todo e avaliar as medidas do modelo
datasetReg <- dataset[,5:26] # Criando o dataset somente com as variáveis necessárias
str(datasetReg)
class(datasetReg)
dim(datasetReg)
View(datasetReg)

modeloLM <- lm(meanEnergyConsumptionKWh100Km ~ ., data = datasetReg) # Criando o modelo
summary(modeloLM)

#### Analisando correlação entre as variáveis independentes ####
cor(datasetReg[,1:21])

# Podemos notar que existem alta correlações entre algumas variáveis explicativas
# por isso faremos o feature selection utilizando o algoritmo stepwise
# para escolhermos as variáveis mais significativas


#### Utilizando o algoritmo stepwise para melhor ajuste e seleção das variáveis ####
step(modeloLM, direction = "both", scale = 0.962^2)


modeloLM2 <- lm(formula = meanEnergyConsumptionKWh100Km ~ enginePowerKm + 
     driveType + batteryCapacityKWh + rangeWltpKm + lengthCm + 
     heightCm + minimalEmptyWeightKg + maximumLoadCapacityKg + 
     numberOfDoors + tireSizeIn + maximumSpeedKph + maximumDcChargingPowerKW, 
   data = datasetReg)

summary(modeloLM2)

modeloLM2$coefficients

#### Avaliando o modelo ####
par(mfrow = c(2,2))
plot(modeloLM2, which = c(1:4), pch = 20)

#### Outliers ####
# Identificado registro 22 como um possível outlier,
# porém analisando o registro não foi evidenciado fatores
# tão discrepantes então decidi mantê-lo

?outlierTest
outlierTest(modeloLM2)


#### Avaliando normalidade dos resíduos ####
# Teste de Shapiro-Wilk
# H0: a distribuição Normal modela adequadamente os resíduos do modelo.
# H1: a distribuição Normal não modela adequadamente os resíduos do modelo.
# p-value = 0.8696 o que indica que a distribuição dos resíduos é normal.

shapiro.test(modeloLM2$residuals)


#### Avaliando a independência ####

# Teste de Durbin-Watson
# H0: os valores dos resíduos do modelo são independentes.
# H1: os resíduos são autocorrelacionados.
# p-value de 0,54 então não rejeitamos a H0, e a suposição de independência foi atendida.

durbinWatsonTest(modeloLM2)

#### Avaliando a homogeneidade de variâncias (Homocedasticidade) ####
# Score Test for Non-Constant Error Variance
# H0: os resíduos são homoscedásticos.
# H1: os resíduos não são homoscedásticos.
# p-value = 0.4997 então não rejeitamos a H0, e a suposição de homocedasticidade foi atendida.

?bptest
?ncvTest
ncvTest(lm(meanEnergyConsumptionKWh100Km ~ enginePowerKm + 
             driveType + batteryCapacityKWh + rangeWltpKm + lengthCm + 
             heightCm + minimalEmptyWeightKg + maximumLoadCapacityKg + 
             numberOfDoors + tireSizeIn + maximumSpeedKph + maximumDcChargingPowerKW, data = datasetReg))

#### Avaliação final ####

summary(modeloLM)
summary(modeloLM2)

# modeloLM2 escolhido pois tem o maior Adjusted R-squared,
# embora a diferença seja baixa (0,0062) este modelo é mais simples
# devido a eliminação das variáveis que não tem impacto no modelo

# Modelo final:
summary(modeloLM2)
modeloLM2$coefficients

# R2 0,9523, ou seja, 95,23% da variação dos scores médios do consumo de energia
# podem ser explicados pelas variáveis preditoras


#### Azure Machine Learning ####


# Link do Projeto feito no Azure ML https://gallery.azure.ai/Experiment/Projeto01-DSA

# Foram avaliados os modelos: 
# - Regressão Linear (sem e com filter based feature selection)
# - Network Regression
# - Boosted Decision Tree Regression

# Boosted Decision Tree Regression teve o melhor desempenho:

# Mean Absolute Error	0.1887
# Root Mean Squared Error	0.285366
# Relative Absolute Error	0.053768
# Relative Squared Error	0.004822
# Coefficient of Determination	0.995178