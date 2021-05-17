from sklearn import tree
from sklearn import ensemble
from sklearn import svm
from sklearn.naive_bayes import GaussianNB,MultinomialNB

model = {
 "decision_tree_gini": tree.DecisionTreeClassifier(
 criterion="gini"
 ),
 "decision_tree_entropy": tree.DecisionTreeClassifier(
 criterion="entropy"
 ),
 "rf": ensemble.RandomForestClassifier(),
 "svm": svm.SVC(kernel = 'linear'),
 "naive_bayes_gaussian": GaussianNB(),
 "naive_bayes_multinominal": MultinomialNB()
}