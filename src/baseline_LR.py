from sklearn.linear_model import LogisticRegression
from datautil import load_eng_all_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from datautil import load_zho_tfidplus_features
# >>> X = [[0, 0], [1, 1]]
# >>> Y = [0, 1]
# >>> clf = RandomForestClassifier(n_estimators=10)
# >>> clf = clf.fit(X, Y)

def test_lr(X,y):
#    X, y = load_eng_all_features()
    train_data, test_data, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    clf = LogisticRegression()
    clf.fit(train_data, y_train)
    y_pred=clf.predict(test_data)
    print ('logistic regression, f1 scores:', f1_score(y_test, y_pred, average='macro'))

def test_random_forest(X,y):
 #   X, y = load_eng_all_features()
    train_data, test_data, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(train_data, y_train)
    y_pred=clf.predict(test_data)
    print ('random forest, f1 scores:', f1_score(y_test, y_pred, average='macro'))

def test_ada_boost(X,y):
 #   X, y = load_eng_all_features()
    train_data, test_data, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    clf = AdaBoostClassifier(n_estimators=10)
    clf.fit(train_data, y_train)
    y_pred=clf.predict(test_data)
    print ('adaboost, f1 scores:', f1_score(y_test, y_pred, average='macro'))

if __name__=='__main__':
    print ('test english')
    X, y = load_eng_all_features()

    test_lr(X,y)
    test_random_forest(X,y)
#    test_ada_boost(X,y)

    print ('test chinese')
    X, y = load_zho_tfidplus_features()

    test_lr(X,y)
    test_random_forest(X,y)
