# Plantilla para el Pre Procesado de Datos - Datos faltantes
# Importar el dataset
dataset = read.csv('Data.csv')


# Tratamiento de los valores NA
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),dataset$Salary)
dataset$Country[1]<- NA
View(dataset)

dataset$Country<-ifelse(is.na(dataset$Country),
       sample(dataset$Country[-is.na(dataset$Country)],1),dataset$Country)
class(dataset[,1])
