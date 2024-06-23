import init_data

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMRegressor
from econml.metalearners import SLearner
from econml.metalearners import XLearner
import matplotlib.pyplot as plt
from sklift.viz import plot_uplift_curve

# データの分割
def split_data(df, seed):
    X = df.drop(['treatment','exposure','visit','conversion'], axis=1)
    X = (X - X.mean()) / X.std()
    T = df['treatment']
    y = df['visit']

    X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(
        X, T, y, train_size=0.7, random_state=seed, stratify=T
    )

    # インデックスをリセット
    T_test = T_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, T_train, T_test, y_train, y_test

def create_custom_objective(T_train):
    def custom_objective(y_pred: np.ndarray, train_data: lgb.Dataset):
        y = train_data.get_label()
        t = T_train.values

        treatment = (t == 1)
        control = (t == 0)

        N_1 = np.sum(treatment)
        N_0 = np.sum(control)

        exp_scores = np.exp(y_pred - np.max(y_pred))
        sum_exp_scores = np.sum(exp_scores)
        p = exp_scores / sum_exp_scores

        sum_y_treat = np.sum(y[treatment])
        sum_y_control = np.sum(y[control])

        grad = (p * sum_y_treat / N_1) - (p * sum_y_control / N_0)

        # (-1)^t(k) * y_k / N_t(k) 項の計算
        last_term = ((-1) ** treatment) * y / np.where(treatment, N_1, N_0)
        grad += last_term
        p_one_minus_p = p * (1 - p)
        hess = (p_one_minus_p * sum_y_treat / N_1) - (p_one_minus_p * sum_y_control / N_0)

        return grad, hess
    return custom_objective

def get_tau_direct(X_train, y_train, X_test, custom_objective):
    # データセットの作成
    dtrain = lgb.Dataset(X_train, label=y_train)
    # LightGBMのパラメータ
    params = {
        'objective': lambda y_pred, train_data: custom_objective(y_pred, train_data)
        }

    # モデルの訓練
    bst = lgb.train(params, dtrain)

    # テストデータセットに対する予測
    tau_direct = bst.predict(X_test, num_iteration=bst.best_iteration)

    return tau_direct

def get_tau_sl(y_train, T_train, X_train, X_test):
    models = LGBMRegressor()
    S_learner = SLearner(overall_model = models)
    S_learner.fit(y_train, T_train, X = X_train)

    # 効果の推定
    tau_sl = S_learner.effect(X_test)

    return tau_sl

def get_tau_xl(y_train, T_train, X_train, X_test):
    models = LGBMRegressor()
    propensity_model = LogisticRegression()
    X_learner = XLearner(models=models, propensity_model=propensity_model)
    X_learner.fit(y_train, T_train, X = X_train)

    # 効果の推定
    tau_xl = X_learner.effect(X_test)

    return tau_xl

def uplift(tau_sl, tau_xl, tau_direct, y_test, T_test):

    # 描画設定
    fig, ax = plt.subplots(figsize=(10, 8))

    # SLモデルのアップリフト曲線
    plot_uplift_curve(y_true=y_test, uplift=tau_sl, treatment=T_test, perfect=False, random=False, ax=ax, name='SL Model', color='orange')

    # XLモデルのアップリフト曲線
    plot_uplift_curve(y_true=y_test, uplift=tau_xl, treatment=T_test, perfect=False, random=False, ax=ax, name='XL Model', color='green')

    # causal forest でのアップリフト曲線
    # plot_uplift_curve(y_true=y_test, uplift=tau_cf, treatment=T_test, perfect=False, random=False, ax=ax, name='Causal Forest')

    # Directモデルのアップリフト曲線
    plot_uplift_curve(y_true=y_test, uplift=tau_direct, treatment=T_test, perfect=False, ax=ax, name='Direct Model', color='red')

    ax.set_title('Comparison of Uplift Models')
    ax.legend()
    plt.show()

    # AUCの計算
    from sklift.metrics import uplift_auc_score
    print('SL Model:', uplift_auc_score(y_true=y_test, uplift=tau_sl, treatment=T_test))
    print('XL Model:', uplift_auc_score(y_true=y_test, uplift=tau_xl, treatment=T_test))
    print('Direct Model:', uplift_auc_score(y_true=y_test, uplift=tau_direct, treatment=T_test))

def main():
    df = init_data.get_data()
    seed = 42
    X_train, X_test, T_train, T_test, y_train, y_test = split_data(df, seed)
    tau_sl = get_tau_sl(y_train, T_train, X_train, X_test)
    tau_xl = get_tau_xl(y_train, T_train, X_train, X_test)
    custom_objective = create_custom_objective(T_train)
    tau_direct = get_tau_direct(X_train, y_train, X_test, custom_objective)
    uplift(tau_sl, tau_xl, tau_direct, y_test, T_test)

    
