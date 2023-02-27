# Plantilla para el Pre Procesado de Datos
# Importar el dataset
dataset = read.csv('Data.csv')
#dataset = dataset[, 2:3]
dataset$Age= ifelse(is.na(dataset$Age),median(dataset$Age,na.rm=T),dataset$Age)
dataset$Salary= ifelse(is.na(dataset$Salary),median(dataset$Salary,na.rm=T),dataset$Salary)
############ Creación de factores
dataset$Country <- factor(dataset$Country,levels=c('France','Spain','Germany',labels=c(1,2,3)))
dataset$Country <- factor(dataset$Purchased,levels=c('No','Yes',labels=c(0,1)))
# Dividir los datos en conjunto de entrenamiento y conjunto de test
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Escalado de valores
training_set[,2:3] = scale(training_set[,2:3])
testing_set[,2:3] = scale(testing_set[,2:3])
