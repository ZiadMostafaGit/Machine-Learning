import argparse
import numpy as np
import matplotlib.pyplot as plt

from data_helper import  *
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import mean_squared_error


KFLODS = 4
RANDOM_STATE = 17
np.random.seed(RANDOM_STATE)



def do_regression(X_train, t_train, is_ridge=False, alpha=1):
    if not is_ridge:
        model = LinearRegression(fit_intercept = True)
    else:
        model = Ridge(fit_intercept=True, alpha=alpha)

    model.fit(X_train, t_train)
    avg_abs_weght = abs(model.coef_).mean()

    print(f'\tintercept: {model.intercept_} - abs avg weight: {avg_abs_weght}')
    return model


def eval_model(model, X, t, keymsg, squared = False):
    t_pred = model.predict(X)
    error = mean_squared_error(t, t_pred, squared=squared)  # rmse

    print(f'\t Error of {keymsg}: {error}')
    return error


def try_polynomial_degrees(X_train, t_train, X_val, t_val, degrees, preprocess_option, use_cross_features = True):
    train_errors_lst = []
    val_errors_lst = []

    for degree in degrees:
        print(f'Try polynoimal of degree {degree} - using cross features? {use_cross_features}')
        X_train_p, X_val_p = preprocess_data(X_train, X_val, poly_degree=degree,
                    preprocess_option=preprocess_option, use_cross_features=use_cross_features)

        model = do_regression(X_train_p, t_train)
        train_error = eval_model(model, X_train_p, t_train, 'train')
        train_errors_lst.append(train_error)

        val_error = eval_model(model, X_val_p, t_val, 'val')
        val_errors_lst.append(val_error)


    if len(degrees) > 1:
        plt.title(f'Degree vs train/val errors')
        plt.xlabel('Degree')
        plt.ylabel('RMSE')

        plt.xticks(degrees)
        plt.plot(degrees, train_errors_lst, label='train error')
        plt.plot(degrees, val_errors_lst, label='val error')
        plt.legend(loc='best')

        plt.grid()
        plt.show()

    return train_errors_lst, val_errors_lst


def try_single_features(X_train, t_train, X_val, t_val, features, poly_degree, use_cross_features=True, preprocess_option=1):
    train_errors_lst = []
    val_errors_lst = []

    for feat in features:
        print(f'Trying individual feature id: {feat}')
        X_train_p = X_train[:, feat:feat+1]
        X_val_p = X_val[:, feat: feat + 1]

        X_train_p, X_val_p = preprocess_data(X_train_p, X_val_p, poly_degree=poly_degree,
                                use_cross_features=use_cross_features, preprocess_option=preprocess_option)

        model = do_regression(X_train_p, t_train)
        train_error = eval_model(model, X_train_p, t_train, 'train')
        train_errors_lst.append(train_error)

        val_error = eval_model(model, X_val_p, t_val, 'val')
        val_errors_lst.append(val_error)

    features = np.array(features)

    plt.bar(features, train_errors_lst, color='b', width=0.25)
    plt.bar(features + 0.25, val_errors_lst, color='g', width=0.25)
    plt.xlabel('Feature IDs', fontweight='bold', fontsize=10)
    plt.ylabel('RMSE', fontweight='bold', fontsize=10)
    plt.xticks(features)
    plt.legend(labels=['Train', 'Val'])
    plt.title('Single Feature Performance - Bar Plot')
    plt.show()


def try_regualrized_ridge_CV(X, t, alphas, poly_degree=2, use_cross_features=False, preprocess_option=1):
    # We can't scale here
    # CV will make several splits, for each split will scale
    X, _ = preprocess_data(X, None, poly_degree=poly_degree,
                                   use_cross_features=use_cross_features, preprocess_option=0)

    preprocessor = get_preprocessor(preprocess_option)
    if preprocessor is None:
        pipeline = Pipeline(steps=[('model', Ridge())])
    else:
        pipeline = Pipeline(steps=[("scaler", preprocessor), ('model', Ridge())])

    grid = {'model__alpha': alphas}
    kf = KFold(n_splits=KFLODS, random_state=RANDOM_STATE, shuffle=True)
    search = GridSearchCV(pipeline, grid, scoring='neg_mean_squared_error', cv=kf)

    search.fit(X, t)
    rmses = np.sqrt(-search.cv_results_['mean_test_score'])
    for (alpha, rmse) in zip(alphas, rmses):
        print(f'alpha = {alpha} - rmse = {rmse}', end='')
        if alpha == search.best_params_['model__alpha']:
            print('\t\t**BEST PARAM**')
        else:
            print()

    plt.title(f'log10(Aphas) for degree {poly_degree} vs croos validation RMSE')
    plt.xlabel('log10(alpha)')
    plt.ylabel('RMSE')

    # visualizing with alpha is hard. Use lgo10
    alphas = np.log10(alphas)   # TRY commenting this line
    plt.xticks(alphas)
    plt.plot(alphas, rmses, label='train error')

    plt.grid()
    plt.show()

    print('*************************************\n')

    return search.best_params_, search.best_estimator_


def lasso_selection(X, t, X_train, t_train, X_val, t_val, preprocessor_option):
    X_train, X_val = preprocess_data(X_train, X_val, poly_degree=1, preprocess_option=preprocess_option)

    # Don't use grid search with lasso
    # Seems small alpha is better. let's explore deeper
    alphas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
              0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # try ridge with ALL features
    try_regualrized_ridge_CV(X, t, alphas, poly_degree=1, preprocess_option=preprocess_option)

    alpha_indices_dct = {}

    for alpha in alphas:
        model = Lasso(fit_intercept = True, alpha = alpha, max_iter = 10000)
        selector = SelectFromModel(estimator=model)
        selector.fit(X_train, t_train)
        #print(selector.threshold_)
        flags =selector.get_support()
        indices = np.flatnonzero(flags)
        alpha_indices_dct[alpha] = indices

        pred_t = selector.estimator_.predict(X_val)
        lass_val_err = mean_squared_error(t_val, pred_t)

        print(f'alpha={alpha}, selects {len(indices)} features and has {lass_val_err} val error')
    print('+++++++++++++++++++++++++++++++++++++++++++++\n')

    # try ridge with 9 selected features
    # alpha=0.3, selects 11 features and has 84.99862618663573 val error
    # alpha=0.4, selects 9 features and has 85.22667919766862 val error
    indices = alpha_indices_dct[0.4]    # these indices seems informative enough with good error
    try_regualrized_ridge_CV(X[:, indices], t, alphas, poly_degree=1, preprocess_option=preprocess_option)



def lasso_selection_(X_train, t_train, preprocessor_option):
    # Don't use grid search with lasso
    alphas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
              0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
              1, 10]

    alphas = [0.01, 1]

    preprocessor = get_preprocessor(preprocessor_option)
    avg_scores = []

    for alpha in alphas:
        model = Lasso(fit_intercept = True, alpha = alpha, max_iter = 10000)
        if preprocessor is not None:
            # makepipeline won't be problem as we create lasso with its lambda
            pipeline = make_pipeline(preprocessor, model)
        else:
            pipeline = model

        kf = KFold(n_splits=KFLODS, random_state=RANDOM_STATE, shuffle=True)
        scores = cross_val_score(pipeline, X_train, t_train, cv=kf,
                                 scoring='neg_mean_squared_error')
        scores *= -1  # change to mean_squared_error
        print(f'alpha: {alpha}', scores.mean(), scores.std())
        avg_scores.append(scores.mean())

    mn_idx = avg_scores.index(min(avg_scores))
    best_alpha = alphas[mn_idx]
    print(f'\nbest alpha is {best_alpha}')
    # best alpha is 0.01

    model = Lasso(fit_intercept=True, alpha=best_alpha)
    pipeline = make_pipeline(preprocessor, model)   # Do we need to reinitate object from preprocessor? ToDO
    pipeline.fit(X_train, t_train)    # scale and fit
    # discard all features with weights close to zero
    mask = np.logical_not(np.isclose(model.coef_, 0))
    print(np.count_nonzero(mask), 'Selected features:', mask)
    # 31 Selected features: [ True False  True  True  True  True  True  True  True False  True  True True  True  True  True  True  True False  True  True  True  True False True  True False  True False False  True  True  True  True  True False True  True  True]
    return mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Regressors Homework')

    parser.add_argument('--dataset', type=str, default='data2_200x30.csv')
    parser.add_argument('--preprocessing', type=int, default=1,
                        help='0 for no processing, 1 for min/max scaling and 2 for standrizing')

    parser.add_argument('--choice', type=int, default=6,
                        help='1 for simple linear regression, '
                             '2 for polynomial on all features with degrees [1, 2, 3, 4] with cross features'
                             '3 for polynomial on all features with degrees [1, 2, 3, 4] with monomial features'
                             '4 individual features test'
                             '5 find best lambda with grid search with ridge'
                             '6 Lasso selection'
                        )
    parser.add_argument('--extra_args', type=str, help='extra values passed', default='0,3,6')

    args = parser.parse_args()

    X, t, X_train, t_train, X_val, t_val = load_data(args.dataset)

    preprocess_option = args.preprocessing

    if args.choice == 1:
        try_polynomial_degrees(X_train, t_train, X_val, t_val, [1], preprocess_option)
    elif args.choice == 2:
        try_polynomial_degrees(X_train, t_train, X_val, t_val, [1, 2, 3, 4], preprocess_option, use_cross_features=True)
    elif args.choice == 3:
        try_polynomial_degrees(X_train, t_train, X_val, t_val, [1, 2, 3, 4], preprocess_option, use_cross_features=False)
    elif args.choice == 4:
        features = args.extra_args.split(',')
        features = [int(v) for v in features]

        if features[0] == -1:   # use all
            features = list(range(X_train.shape[1]))

        degrees = [1, 2, 3]

        for degree in degrees:
            try_single_features(X_train, t_train, X_val, t_val, features, poly_degree=degree)

    elif args.choice == 5:
        alphas = [0.01, 0.1, 1, 10, 100, 1000, 10000]
        try_regualrized_ridge_CV(X, t, alphas, poly_degree=2, use_cross_features=False, preprocess_option=preprocess_option)

    elif args.choice == 6:
        # Use lasso to select features for us
        lasso_selection(X, t, X_train, t_train, X_val, t_val, preprocess_option)  
