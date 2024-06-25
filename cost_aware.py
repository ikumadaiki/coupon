import init_data

# 必要なライブラリのインポート
import numpy as np
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from econml.metalearners import SLearner
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

# データの前処理
def preprocess(df):
    X = df.drop(['treatment', 'exposure', 'visit', 'conversion'], axis=1)
    X = (X - X.mean()) / X.std()
    T = df['treatment']
    y_r = df['conversion']
    y_c = df['visit']
    return X, T, y_r, y_c

# データの分割
def split_data(X, T, y_r, y_c, seed):
    X_train, X_test, T_train, T_test, y_r_train, y_r_test, y_c_train, y_c_test = train_test_split(
        X, T, y_r, y_c, train_size=0.7, random_state=seed, stratify=T
    )

    # インデックスをリセット
    T_test = T_test.reset_index(drop=True)
    y_r_test = y_r_test.reset_index(drop=True)
    y_c_test = y_c_test.reset_index(drop=True)

    return X_train, X_test, T_train, T_test, y_r_train, y_r_test, y_c_train, y_c_test

def create_custom_objective(y_r_train, y_c_train, T_train):
    def custom_objective(y_pred: np.ndarray, train_data: lgb.Dataset):
        y_r = y_r_train
        y_c = y_c_train
        t = T_train.values
        treatment = (t == 1)
        control = (t == 0)

        N_1 = np.sum(treatment)
        N_0 = np.sum(control)

        p = 1 / (1 + np.exp(-y_pred))

        grad = ((-1) ** t) * (y_r - y_c * p) / np.where(treatment, N_1, N_0)

        hess = y_c * p * (1 - p) / np.where(treatment, N_1, N_0)

        return grad, hess
    return custom_objective

class BaseModel:
    def __init__(self, X_train, X_test, T_train=None, y_r_train=None, y_c_train=None, custom_objective=None):
        self.X_train = X_train
        self.X_test = X_test
        self.T_train = T_train
        self.y_r_train = y_r_train
        self.y_c_train = y_c_train
        self.custom_objective = custom_objective
        self.model = None

    def fit(self):
        raise NotImplementedError("Subclass must implement abstract method")
    
    def predict(self):
        raise NotImplementedError("Subclass must implement abstract method")
    
class ROIDirectModel(BaseModel):
    def fit(self):
        dtrain = lgb.Dataset(self.X_train, free_raw_data=False)
        params = {
            'objective': lambda y_pred, train_data: self.custom_objective(y_pred, train_data),
            'verbose': -1,
        }
        self.model = lgb.train(params, dtrain)
    
    def predict(self):
        x = self.model.predict(self.X_test, num_iteration=self.model.best_iteration)
        return 1 / (1 + np.exp(-x))
    
class ROITPMSLModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = MinMaxScaler()
        self.moder_r = None
        self.model_c = None

    def fit(self):
        models_r = LGBMRegressor()
        self.model_r = SLearner(overall_model=models_r)
        self.model_r.fit(self.y_r_train, self.T_train, X=self.X_train)

        models_c = LGBMRegressor()
        self.model_c = SLearner(overall_model=models_c)
        self.model_c.fit(self.y_c_train, self.T_train, X=self.X_train)

    def predict(self):
        tau_r = self.model_r.effect(self.X_test)
        tau_c = self.model_c.effect(self.X_test)
        roi_tpmsl = tau_r / tau_c
        roi_tpmsl = self.scaler.fit_transform(roi_tpmsl.reshape(-1, 1)).flatten()
        return roi_tpmsl

def calculate_values(roi_scores, T_test, y_r_test, y_c_test):
    sorted_indices = np.argsort(roi_scores)[::-1]
    p_values = np.linspace(0, 1, 30)
    incremental_costs = []
    incremental_values = []
    
    for p in p_values:
        top_p_indices = sorted_indices[:int(p * len(roi_scores))]
        treatment_indices = (T_test[top_p_indices] == 1)
        
        # ATE (Average Treatment Effect) の計算
        ATE_Yr = np.mean(y_r_test[top_p_indices][treatment_indices]) - np.mean(y_r_test[top_p_indices][~treatment_indices])
        ATE_Yc = np.mean(y_c_test[top_p_indices][treatment_indices]) - np.mean(y_c_test[top_p_indices][~treatment_indices])
        
        incremental_costs.append(ATE_Yc * np.sum(treatment_indices))
        incremental_values.append(ATE_Yr * np.sum(treatment_indices))
        # print(ATE_Yr , ATE_Yc,np.sum(treatment_indices))
        incremental_costs[0] = 0
        incremental_values[0] = 0
        
    return incremental_costs, incremental_values

# Cost Curve の描画
def plot_cost_curve(incremental_costs_direct, incremental_values_direct, incremental_costs_tpmsl, incremental_values_tpmsl):
    plt.figure(figsize=(10, 6))
    plt.plot(incremental_costs_tpmsl / max(incremental_costs_tpmsl), incremental_values_tpmsl / max(incremental_values_tpmsl), marker='o', color='orange',  label='TPMSL')
    plt.plot(incremental_costs_direct / max(incremental_costs_direct), incremental_values_direct / max(incremental_values_direct), marker='x', color='red', label='Direct')
    # y = x の直線
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title('Cost Curve Comparison')
    plt.xlabel('Incremental Cost')
    plt.ylabel('Incremental Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    df = init_data.get_data()
    # パラメータの設定
    seed = 42
    roi_list = []
    X, T, y_r, y_c = preprocess(df)
    X_train, X_test, T_train, T_test, y_r_train, y_r_test, y_c_train, y_c_test = split_data(X, T, y_r, y_c, seed)
    for model in [ROIDirectModel, ROITPMSLModel]:
        model_instance = model(X_train, X_test, T_train, y_r_train, y_c_train, create_custom_objective(y_r_train, y_c_train, T_train))
        model_instance.fit()
        roi_scores = model_instance.predict()
        roi_list.append(roi_scores)
    
    incremental_costs_direct, incremental_values_direct = calculate_values(roi_list[0], T_test, y_r_test, y_c_test)
    incremental_costs_tpmsl, incremental_values_tpmsl = calculate_values(roi_list[1], T_test, y_r_test, y_c_test)
    plot_cost_curve(incremental_costs_direct, incremental_values_direct, incremental_costs_tpmsl, incremental_values_tpmsl)    