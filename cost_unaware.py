print('cost_unaware.py')

import numpy as np

import init_data
df = init_data.get_data()

# パラメータの設定
seed = 42

# データの分割
def split_data(df, seed):
    from sklearn.model_selection import train_test_split

    X = df.drop(['treatment','exposure','visit','conversion'], axis=1)
    X = (X - X.mean()) / X.std()
    T = df['treatment']
    y = df['visit']

    X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(X, T, y, test_size=0.3, random_state=seed, stratify=T)

    return X_train, X_test, T_train, T_test, y_train, y_test

X_train, X_test, T_train, T_test, y_train, y_test = split_data(df, seed)

def softmax(scores): 
    if scores.ndim == 2:
        scores = scores.T
        scores = scores - np.max(scores, axis=0)
        y = np.exp(scores) / np.sum(np.exp(scores), axis=0)
        return y.T

    scores = scores - np.max(scores) # オーバーフロー対策 
    return np.exp(scores) / np.sum(np.exp(scores))


def softmax_naive(scores):
    exps = np.exp(scores)
    return exps / np.sum(exps)

N1 = np.sum(T_train == 1)
N0 = np.sum(T_train == 0)

y_train_1 = y_train[T_train == 1]
y_train_0 = y_train[T_train == 0]

X_train_1 = X_train[T_train == 1]
X_train_0 = X_train[T_train == 0]

def get_tau_direct():
    import numpy as np
    from scipy.optimize import minimize


    # 初期重み
    w_initial = np.random.rand(X_train.shape[1])

    # 最適化
    result = minimize(loss_function, w_initial, method='BFGS')

    if result.success:
        print("Optimized weights:", result.x)
        print("Minimum loss:", result.fun)
    else:
        print("Optimization failed:", result.message)

    # sとqの最適解の導出
    w = result.x
    scores = np.dot(X_test, w)
    probabilities = softmax(scores)
    tau_direct = probabilities * np.sum(np.exp(scores))

    return tau_direct

# 損失関数 L(s)
def loss_function(w):
    scores_1 = np.dot(X_train_1, w)
    scores_0 = np.dot(X_train_0, w)
    probabilities_1 = softmax(scores_1)
    probabilities_0 = softmax(scores_0)
    return -np.sum(y_train_1 * np.log(probabilities_1)) / N1 + np.sum(y_train_0 * np.log(probabilities_0)) / N0

def get_tau_sl():
    # from sklearn.ensemble import RandomForestRegressor
    from lightgbm import LGBMRegressor
    from econml.metalearners import SLearner
    from sklearn.linear_model import LinearRegression

    # モデルの構築
    # models = RandomForestRegressor(max_depth=10, random_state=0)
    models = LGBMRegressor()
    S_learner = SLearner(overall_model = models)
    S_learner.fit(y_train, T_train, X = X_train)

    # 効果の推定
    tau_sl = S_learner.effect(X_test)

    return tau_sl

def get_tau_xl():
    # 必要なライブラリのインポート
    # from sklearn.ensemble import RandomForestRegressor
    from lightgbm import LGBMRegressor
    from sklearn.linear_model import LogisticRegression
    from econml.metalearners import XLearner

    # モデルの構築
    # models = RandomForestRegressor(max_depth=10, random_state=0)
    models = LGBMRegressor()
    propensity_model = LogisticRegression()
    X_learner = XLearner(models=models, propensity_model=propensity_model)
    X_learner.fit(y_train, T_train, X = X_train)

    # 効果の推定
    tau_xl = X_learner.effect(X_test)

    return tau_xl

def uplift(tau_sl, tau_xl, tau_direct):
    import matplotlib.pyplot as plt
    from sklift.viz import plot_uplift_curve

    # 描画設定
    fig, ax = plt.subplots(figsize=(10, 8))

    # SLモデルのアップリフト曲線
    plot_uplift_curve(y_true=y_test, uplift=tau_sl, treatment=T_test, perfect=False, random=False, ax=ax, name='SL Model')

    # XLモデルのアップリフト曲線
    plot_uplift_curve(y_true=y_test, uplift=tau_xl, treatment=T_test, perfect=False, random=False, ax=ax, name='XL Model')

    # causal forest でのアップリフト曲線
    # plot_uplift_curve(y_true=y_test, uplift=tau_cf, treatment=T_test, perfect=False, random=False, ax=ax, name='Causal Forest')

    # Directモデルのアップリフト曲線
    plot_uplift_curve(y_true=y_test, uplift=tau_direct, treatment=T_test, perfect=False, ax=ax, name='Direct Model')

    ax.set_title('Comparison of Uplift Models')
    ax.legend()
    plt.show()
