import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

 
 
df = pd.read_csv('train.csv')
#dt = pd.read_csv('test.csv')
 
#print(df.info())
 
df.drop(['bdate'], axis = 1, inplace = True)
df.drop(['langs'], axis = 1, inplace = True)
df.drop(['city'], axis = 1, inplace = True)
df.drop(['people_main'], axis = 1, inplace = True)
df.drop(['life_main'], axis = 1, inplace = True)
df.drop(['last_seen'], axis = 1, inplace = True)
df.drop(['occupation_type'], axis = 1, inplace = True)
df.drop(['occupation_name'], axis = 1, inplace = True)
df.drop(['career_end'], axis = 1, inplace = True)
df.drop(['career_start'], axis = 1, inplace = True)




 
print(df['education_form'].value_counts())
 
def form(index):
    if index == 'Full-time':
        return 3
    if index == 'Distance Learning':
        return 2
    if index == 'Part-time':
        return 1
 
    return 0
 
 
df['education_form'] = df['education_form'].apply(form)
 
df[list(pd.get_dummies(df['education_status']).columns)] = pd.get_dummies(df['education_status'])
print(df.info())
df.drop(['education_status'], axis = 1, inplace = True)
print(df.info())

#df[list(pd.get_dummies(df['education form']))]
x = df.drop('result', axis = 1) # Данные о пассажирах
y = df['result']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
percent = accuracy_score(y_test, y_pred) * 100
print(percent)
#print(df.info())