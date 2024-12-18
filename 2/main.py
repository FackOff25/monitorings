import numpy as np
import data as data
from bayes import NaiveBayesG

Data = np.array([np.array(d[0]) for d in data.Data])
Classes = np.array([d[1] for d in data.Data])

train = [0,1,2,4,
         8,9,10,12]
test = list(range(len(Data)))
for i in train:
    test.remove(i)

X_train = np.array([np.array(Data[i]) for i in train])
y_train = np.array([Classes[i] for i in train])

X_test = np.array([np.array(Data[i]) for i in test])
y_test = np.array([Classes[i] for i in test])

bayes = NaiveBayesG()
bayes.fit(X_train, y_train)

man_y_pred = bayes.predict(X_test)
print("Ошибка в %d случев из %d" % ((y_test != man_y_pred).sum(), X_test.shape[0]))
