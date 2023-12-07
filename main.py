import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz

iris = pd.read_csv('Iris.csv')
iris.info() # melihat Informasi dataset
iris.head() # melihat informasi dataset 5 baris pertama
iris.drop('Id',axis=1,inplace=True) # Menghilangkan kolom yang tidak penting

# Memisahkan atribut dan label
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm' ]]
y = iris['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

# Membuat model Decision tree
tree_model = DecisionTreeClassifier()
# Melatih model menggunakan data latih
tree_model = tree_model.fit(X_train, y_train)

# Evaluasi Model
y_pred = tree_model.predict(X_test)
acc_score = round(accuracy_score(y_pred, y_test),3)
print('Accuracy: ' , acc_score)
print(tree_model.predict([[6.2, 3.4, 5.4, 2.3]])[0])

export_graphviz(
    tree_model,
    out_file = "iris_tree.dot",
    feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
    class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica' ],
    rounded= True,
    filled =True)
