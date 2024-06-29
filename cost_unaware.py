import init_data

# ライブラリのインポート
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMRegressor
from econml.metalearners import SLearner
from econml.metalearners import XLearner
import matplotlib.pyplot as plt
from sklift.viz import plot_uplift_curve

# 前処理
def preprocess(df):
    X = df.drop(['treatment','exposure','visit','conversion'], axis=1)
    X = (X - X.mean()) / X.std()
    T = df['treatment']
    y = df['visit']
    return X, T, y

# データの分割
def split_data(X, T, y, seed):
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

class BaseModel:
    def __init__(self, X_train, y_train, T_train=None, X_test=None, custom_objective=None):
        self.X_train = X_train
        self.y_train = y_train
        self.T_train = T_train
        self.X_test = X_test
        self.custom_objective = custom_objective
        self.model = None

    def fit(self):
        raise NotImplementedError("Subclasses should implement this method")
    
    def predict(self):
        raise NotImplementedError("Subclasses should implement this method")
    
class TauDirectModel(BaseModel):
    def fit(self):
        # データセットの作成
        dtrain = lgb.Dataset(self.X_train, label=self.y_train)
        # LightGBMのパラメータ
        params = {
            'objective': lambda y_pred, train_data: self.custom_objective(y_pred, train_data)
            }

        # モデルの訓練
        self.model = lgb.train(params, dtrain)
    
    def predict(self):
        # テストデータセットに対する予測
        tau_direct = self.model.predict(self.X_test, num_iteration=self.model.best_iteration)
        return tau_direct
    
class TauSLModel(BaseModel):
    def fit(self):
        models = LGBMRegressor()
        self.model = SLearner(overall_model = models)
        self.model.fit(self.y_train, self.T_train, X = self.X_train)
    
    def predict(self):
        # 効果の推定
        tau_sl = self.model.effect(self.X_test)
        return tau_sl
    
class TauXLModel(BaseModel):
    def fit(self):
        models = LGBMRegressor()
        propensity_model = LogisticRegression()
        self.model = XLearner(models=models, propensity_model=propensity_model)
        self.model.fit(self.y_train, self.T_train, X = self.X_train)
    
    def predict(self):
        # 効果の推定
        tau_xl = self.model.effect(self.X_test)
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
    X, T, y = preprocess(df)
    tau_list = []
    X_train, X_test, T_train, T_test, y_train, y_test = split_data(X, T, y, seed)
    for model in [TauSLModel, TauXLModel, TauDirectModel]:
        model_instance = model(X_train, y_train, T_train, X_test, create_custom_objective(T_train))
        model_instance.fit()
        tau = model_instance.predict()
        tau_list.append(tau)
    uplift(*tau_list, y_test, T_test)
    
