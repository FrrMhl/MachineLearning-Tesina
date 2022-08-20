import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import StackingClassifier
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
    sfs = SequentialFeatureSelector(estimator, n_features_to_select='auto', tol=0.01)
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
    logistRegrModel = LogisticRegression(class_weight='balanced')
    logistRegrModel.fit(X_train, y_train)
    previsioni.append(logistRegrModel.predict(X_test))

    model.append('Decision Tree')
    treeModel = tree.DecisionTreeClassifier(max_depth=10)
    treeModel.fit(X_train, y_train)
    previsioni.append(treeModel.predict(X_test))

    model.append('K-NN')
    knnModel = KNeighborsClassifier(n_neighbors=5)
    knnModel.fit(X_train, y_train)
    previsioni.append(knnModel.predict(X_test))

    ####################################################################################
    #                          RISULTATI SENZA GRID SEARCH                             #
    ####################################################################################

    accuracy = []
    for i in range(len(model)):
        print('\n\n\n------------------------      Risultati {}     ------------------------'.format(model[i]))
        print('Accuracy is {}%'.format(round(accuracy_score(y_test, previsioni[i])*100, 2)))
        accuracy.append(round(accuracy_score(y_test, previsioni[i])*100, 2))
        print('Precision is {}%'.format(round(precision_score(y_test, previsioni[i], average='weighted')*100, 2)))
        print('Recall is {}%'.format(round(recall_score(y_test, previsioni[i], average='weighted')*100, 2)))
        print('F1-Score is {}%'.format(round(f1_score(y_test, previsioni[i], average='weighted')*100, 2)))

    ####################################################################################
    #                       CROSS VALIDATION E GRID SEARCH                             #
    ####################################################################################

    gridLR = {
        'solver': ['saga', 'newton-cg', 'lbfgs', 'sag', 'liblinear']
    }
    logistRegrModelCV = GridSearchCV(estimator=LogisticRegression(class_weight='balanced'),
                                     param_grid=gridLR,
                                     cv=5)
    logistRegrModelCV.fit(X_train, y_train)

    gridT = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [10, 15, 20],
    }
    treeModelCV = GridSearchCV(estimator=tree.DecisionTreeClassifier(),
                               param_grid=gridT,
                               cv=5)
    treeModelCV.fit(X_train, y_train)

    gridKNN = {
        'n_neighbors': [1, 3, 5]
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
        accuracy.append(round(accuracy_score(y_test, previsioni[i+3]) * 100, 2))
        if round(accuracy_score(y_test, previsioni[i+3]) * 100, 2) > best:
            best = round(accuracy_score(y_test, previsioni[i+3]) * 100, 2)
            index = i

    print(model[index])
    print('Accuracy is {}%'.format(round(accuracy_score(y_test, previsioni[index+3]) * 100, 2)))
    print('Precision is {}%'.format(round(precision_score(y_test, previsioni[index+3], average='weighted') * 100, 2)))
    print('Recall is {}%'.format(round(recall_score(y_test, previsioni[index+3], average='weighted') * 100, 2)))
    print('F1-Score is {}%'.format(round(f1_score(y_test, previsioni[index+3], average='weighted') * 100, 2)))

    ####################################################################################
    #                                   ENSAMBLE                                       #
    ####################################################################################

    print('\n\n\n------------------------      Valutazione con Ensamble     ------------------------')
    estimators = [
        ('lg', LogisticRegression(solver=logistRegrModelCV.best_params_.get('solver'),
                                  class_weight='balanced')),
        ('dt', tree.DecisionTreeClassifier(criterion=treeModelCV.best_params_.get('criterion'),
                                           max_depth=treeModelCV.best_params_.get('max_depth'))),
        ('knn', KNeighborsClassifier(n_neighbors=knnModelCV.best_params_.get('n_neighbors')))
    ]
    modelloUnico = StackingClassifier(estimators=estimators,
                                      final_estimator=LogisticRegression(),
                                      cv=5)
    modelloUnico.fit(X_train, y_train)
    accuracyEnsamble = round(modelloUnico.score(X_test, y_test) * 100, 2)
    print('\nAccuracy con ensamble -> {}%'.format(accuracyEnsamble))

    if accuracyEnsamble > best:
        print('Con lo Stacking Ensamble abbiamo una valutazione migliore')
    else:
        print('La soluzione migliore rimane {}'.format(model[index]))

    ####################################################################################
    #                                  GRAFICI                                         #
    ####################################################################################

    nRighe, nCol = df.shape
    classi = ['Velenoso', 'Commestibile']
    conteggio = df['class'].value_counts()
    percentuali = [conteggio.iloc[0] / nCol * 100, conteggio.iloc[1] / nCol * 100]
    plt.pie(x=percentuali, labels=classi, autopct='%1.1f%%', startangle=90)

    subset = bestFeature.tolist()
    subset.append('class')
    sns.pairplot(df[subset], hue='class')

    plt.figure()
    barWidth = 0.25
    lr = [accuracy[0], accuracy[3]]
    dt = [accuracy[1], accuracy[4]]
    knn = [accuracy[2], accuracy[5]]

    br1 = np.arange(2)
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    plt.bar(br1, lr, color='r', width=barWidth, edgecolor='grey', label='LR')
    plt.bar(br2, dt, color='g', width=barWidth, edgecolor='grey', label='DT')
    plt.bar(br3, knn, color='b', width=barWidth, edgecolor='grey', label='K-NN')

    for i in range(2):
        plt.text(br1[i], lr[i], lr[i], ha='center', bbox=dict(facecolor='pink', alpha=0.8))
        plt.text(br2[i], dt[i], dt[i], ha='center', bbox=dict(facecolor='pink', alpha=0.8))
        plt.text(br3[i], knn[i], knn[i], ha='center', bbox=dict(facecolor='pink', alpha=0.8))

    plt.xlabel('Time', fontweight='bold', fontsize=15)
    plt.ylabel('Accuracy', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(2)], ['Before', 'After'])
    plt.title('Accuracy prima e dopo grid search')

    plt.legend()

    plt.show()
