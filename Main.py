import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SequentialFeatureSelector
import warnings

if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    df = pd.read_csv('secondary_data.csv', delimiter=';')

    ####################################################################################
    #                            GESTIONE VALORI NULLI E CATEGORICI                    #
    ####################################################################################

    print('\n\n\n------------------------      Prime righe      ------------------------')
    print(df.head())

    print('\n\n\n------------------------      Info generali      ------------------------')
    print(df.info())

    print('\n\n\n------------------------      Somma elementi nulli      ------------------------')
    print(df.isnull().sum())

    print('\n\n\n------------------------      Numero di categorie      ------------------------')
    colonneNulle = df.columns[df.isnull().any()].tolist()
    for col in df[colonneNulle]:
        print('{} = {} ---> {} categorie'.format(col, df[col].unique(), len(df[col].unique())-1))

    for col in df[colonneNulle]:
        df[col].fillna(df[col].mode()[0], inplace=True)
    print('\n\n\n------------------------      Elementi nulli dopo la modifica      ------------------------')
    print(df.isnull().sum())

    colonneCategoriche = df.select_dtypes(include='object').columns
    for col in df[colonneCategoriche]:
        df[col] = LabelEncoder().fit_transform(df[col])
    print('\n\n\n------------------------      Prime righe dopo la discretizzazione      ------------------------')
    print(df.head())

    ####################################################################################
    #                          STANDARDIZZAZINE E FEATURE SELECTION                    #
    ####################################################################################

    X = df.iloc[:, 1:].values
    y = df['class'].values

    scaler = MinMaxScaler()
    scaler.fit(X)
    X_Stand = scaler.transform(X)
    print('\n\n\n------------------------      Feature standardizzate      ------------------------')
    print(X_Stand)

    estimator = LogisticRegression()
    sfs = SequentialFeatureSelector(estimator, n_features_to_select='auto')
    sfs.fit(X_Stand, y)
    print('\n\n\n------------------------      Feature selezionate      ------------------------')
    feature = df.columns.drop('class')
    bestFeature = feature[sfs.get_support()]
    print(bestFeature)
    sfs.transform(X_Stand)

    X_train, X_test, y_train, y_test = train_test_split(X_Stand, y, test_size=0.3, shuffle=True)

    ####################################################################################
    #                     TRAINING DEI MODELLI E PREDIZIONI                            #
    ####################################################################################

    model = []
    previsioni = []

    model.append('Logistic Regression')
    logistRegrModel = LogisticRegression(class_weight='balanced', multi_class='multinomial', solver='saga')
    logistRegrModel.fit(X_train, y_train)
    previsioni.append(logistRegrModel.predict(X_test))

    model.append('Decision Tree')
    treeModel = tree.DecisionTreeClassifier(max_depth=10)
    treeModel.fit(X_train, y_train)
    previsioni.append(treeModel.predict(X_test))
    plt.figure()
    tree.plot_tree(treeModel)

    model.append('K-NN')
    knnModel = KNeighborsClassifier(n_neighbors=5)
    knnModel.fit(X_train, y_train)
    previsioni.append(knnModel.predict(X_test))

    ####################################################################################
    #                          RISULTATI SENZA GRID SEARCH                             #
    ####################################################################################

    for i in range(len(model)):
        print('\n\n\n------------------------      Risultati {}     ------------------------'.format(model[i]))
        print('Accuracy is {}%'.format(round(accuracy_score(y_test, previsioni[i])*100, 2)))
        print('Precision is {}%'.format(round(precision_score(y_test, previsioni[i], average='weighted')*100, 2)))
        print('Recall is {}%'.format(round(recall_score(y_test, previsioni[i], average='weighted')*100, 2)))
        print('F1-Score is {}%'.format(round(f1_score(y_test, previsioni[i], average='weighted')*100, 2)))

    ####################################################################################
    #                       CROSS VALIDATION E GRID SEARCH                             #
    ####################################################################################

    gridLR = {
        'solver': ['saga', 'newton-cg', 'lbfgs', 'sag']
    }
    logistRegrModelCV = GridSearchCV(estimator=LogisticRegression(),
                                     param_grid=gridLR,
                                     cv=5)
    logistRegrModelCV.fit(X_train, y_train)

    gridT = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [8, 10, 12],
    }
    treeModelCV = GridSearchCV(estimator=tree.DecisionTreeClassifier(),
                               param_grid=gridT,
                               cv=5)
    treeModelCV.fit(X_train, y_train)

    gridKNN = {
        'n_neighbors': [3, 5, 7, 11]
    }
    knnModelCV = GridSearchCV(estimator=KNeighborsClassifier(),
                              param_grid=gridKNN,
                              cv=5)
    knnModelCV.fit(X_train, y_train)

    print('\n\n\n------------------------      Parametri migliori     ------------------------')
    print('\nLogistic regression -> {}'.format(logistRegrModelCV.best_params_))
    print('\nDecision Tree -> {}'.format(treeModelCV.best_params_))
    print('\nK-NN -> {}'.format(knnModelCV.best_params_))

    ####################################################################################
    #                        RISULTATI DOPO LA CROSS VALIDATION                        #
    ####################################################################################

    previsioni.append(logistRegrModelCV.predict(X_test))
    previsioni.append(treeModelCV.predict(X_test))
    previsioni.append(knnModelCV.predict(X_test))

    print('\n\n\n------------------------      Modello migliore     ------------------------')
    best = 0
    index = -1
    for i in range(len(model)):
        if round(accuracy_score(y_test, previsioni[i+3]) * 100, 2) > best:
            best = round(accuracy_score(y_test, previsioni[i+3]) * 100, 2)
            index = i

    print(model[i])
    print('Accuracy is {}%'.format(round(accuracy_score(y_test, previsioni[i+3]) * 100, 2)))
    print('Precision is {}%'.format(round(precision_score(y_test, previsioni[i+3], average='weighted') * 100, 2)))
    print('Recall is {}%'.format(round(recall_score(y_test, previsioni[i+3], average='weighted') * 100, 2)))
    print('F1-Score is {}%'.format(round(f1_score(y_test, previsioni[i+3], average='weighted') * 100, 2)))

    ####################################################################################
    #                                   ENSAMBLE                                       #
    ####################################################################################



    #plt.show()

    # TODO fare grafici, ensamble se mi va
