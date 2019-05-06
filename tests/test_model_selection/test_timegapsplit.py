import pandas as pd
import numpy as np
from datetime import timedelta
import datetime

from sklego.model_selection import TimeGapSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso


df = pd.DataFrame(np.random.randint(0, 30, size=(30, 4)), columns=list('ABCy'))
df['date'] = pd.date_range(start='1/1/2018', end='1/30/2018')[::-1]
df.index = df.index.astype(str) + "_"
df = df.sort_values('date')

train = df.head(25)

X_train = train[['A', 'B', 'C']]
y_train = train['y']


def test_timegapsplit():
    cv = TimeGapSplit(df=df, date_col='date',
                      train_duration=timedelta(days=5),
                      valid_duration=timedelta(days=3),
                      gap_duration=timedelta(days=0))

    for i, indices in enumerate(cv.split(X_train, y_train)):
        train_mindate = df.loc[X_train.iloc[indices[0]].index]['date'].min()
        train_maxdate = df.loc[X_train.iloc[indices[0]].index]['date'].max()
        valid_mindate = df.loc[X_train.iloc[indices[1]].index]['date'].min()
        valid_maxdate = df.loc[X_train.iloc[indices[1]].index]['date'].max()

        assert train_mindate <= train_maxdate <= valid_mindate <= valid_maxdate

    # regression testing, check if output changes of the last fold
    assert train_mindate == datetime.datetime.strptime('2018-01-16', "%Y-%m-%d")
    assert train_maxdate == datetime.datetime.strptime('2018-01-20', "%Y-%m-%d")
    assert valid_mindate == datetime.datetime.strptime('2018-01-21', "%Y-%m-%d")
    assert valid_maxdate == datetime.datetime.strptime('2018-01-23', "%Y-%m-%d")


def test_timegapsplit_too_big_gap():
    try:
        TimeGapSplit(df=df, date_col='date',
                     train_duration=timedelta(days=5),
                     valid_duration=timedelta(days=3),
                     gap_duration=timedelta(days=5))
    except AssertionError:
        print("Successfully failed")


def test_timegapsplit_with_a_gap():
    gap_duration = timedelta(days=2)
    cv_gap = TimeGapSplit(df=df, date_col='date',
                          train_duration=timedelta(days=5),
                          valid_duration=timedelta(days=3),
                          gap_duration=gap_duration)

    for i, indices in enumerate(cv_gap.split(X_train, y_train)):
        train_mindate = df.loc[X_train.iloc[indices[0]].index]['date'].min()
        train_maxdate = df.loc[X_train.iloc[indices[0]].index]['date'].max()
        valid_mindate = df.loc[X_train.iloc[indices[1]].index]['date'].min()
        valid_maxdate = df.loc[X_train.iloc[indices[1]].index]['date'].max()

        assert train_mindate <= train_maxdate <= valid_mindate <= valid_maxdate
        assert valid_mindate - train_maxdate >= gap_duration


def test_timegapsplit_with_gridsearch():

    cv = TimeGapSplit(df=df, date_col='date',
                      train_duration=timedelta(days=5),
                      valid_duration=timedelta(days=3),
                      gap_duration=timedelta(days=0))

    Lasso(random_state=0, tol=0.1, alpha=0.8).fit(X_train, y_train)

    pipe = Pipeline([
        ('reg', Lasso(random_state=0, tol=0.1)),
    ])
    alphas = [0.1, 0.5, 0.8]
    grid = GridSearchCV(pipe, {'reg__alpha': alphas}, cv=cv)
    grid.fit(X_train, y_train)
    best_C = grid.best_estimator_.get_params()['reg__alpha']

    assert best_C
