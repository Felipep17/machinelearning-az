# Regresión Lineal Simple

# Importar el dataset
dataset = read.csv('Salary_Data.csv')
#dataset = dataset[, 2:3]

# Dividir los datos en conjunto de entrenamiento y conjunto de test
# install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Escalado de valores
# training_set[,2:3] = scale(training_set[,2:3])
# testing_set[,2:3] = scale(testing_set[,2:3])

# Ajustar el modelo de regresión lineal simple con el conjunto de entrenamiento
regressor = lm(Salary ~ YearsExperience,
               data = training_set)

# Predecir resultados con el conjunto de test
y_pred = predict(regressor, newdata = testing_set)

# Visualización de los resultados en el conjunto de entrenamiento

library(ggplot2)
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = "red") +
  geom_line(aes(x = training_set$YearsExperience, 
                y = predict(regressor, newdata = training_set)),
            colour = "blue") +
  ggtitle("Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)") +
  xlab("Años de Experiencia") +
  ylab("Sueldo (en $)")

# Visualización de los resultados en el conjunto de testing
ggplot() + 
  geom_point(aes(x = testing_set$YearsExperience, y = testing_set$Salary),
             colour = "aquamarine4",cex=2) +
  geom_line(aes(x = training_set$YearsExperience, 
                y = predict(regressor, newdata = training_set)),
            colour = "purple4",linewidth=1.5,lty=2) +
  ggtitle("Sueldo vs Años de Experiencia (Conjunto de Testing)") +
  xlab("Años de Experiencia") +
  ylab("Sueldo (en $)")
summary(regressor)
library(ggfortify)
autoplot(regressor)
model<- lm(Salary ~ YearsExperience,
           data = dataset)
summary(model)
autoplot(regressor)
qqPlot(regressor)
shapiro.test(residuals(regressor))
library(lmtest)
bptest(regressor)
library(MASS)
plot(fitted.values(regressor),studres(regressor),pch=19,xlab='Valores Ajustados',ylab=' Residuos Estudentizados')
lines(lowess(studres(regressor)~fitted.values(regressor)),lwd=2,col='red1')
abline(h=0,lwd=2,lty=2)
