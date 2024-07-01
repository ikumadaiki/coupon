import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import LGBMRegressor
from econml.metalearners import SLearner
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import polars as pl

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_data(n, p, dic, seed=42):
    np.random.seed(seed)
    features = np.random.normal(size=(n, p))
    dic["features"] = features
    return dic

def generate_treatment(dic, seed=42):
    np.random.seed(seed)
    logistic_model = LogisticRegression(max_iter=1000)
    dic["target"] = (np.dot(dic["features"], np.random.uniform(0.1, 0.5, size=dic["features"].shape[1])) - 0.5 + np.random.normal(0, 0.5, size = len(dic["features"])) > 0).astype(int)
    logistic_model.fit(dic["features"], dic["target"])
    dic["T_prob"] = logistic_model.predict_proba(dic["features"])[:, 1]
    dic["T_prob"] = dic["T_prob"].clip(0.01, 0.99)
    dic["T"] = np.random.binomial(1, dic["T_prob"])
    dic.pop("target")
    return dic

# T_Probを予測値に変更
def predict_treatment(dic):
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(dic["features"], dic["T"])
    dic["T_prob"] = logistic_model.predict_proba(dic["features"])[:, 1]
    dic["T_prob"] = dic["T_prob"].clip(0.01, 0.99)
    return dic

def generate_visit(dic, seed=42):
    np.random.seed(seed)
    interaction_effects = sigmoid(np.sum(dic["features"], axis=1))
    baseline_effect = 0.3 + dic["features"][:, 2] * 0.4 + dic["features"][:, 4] * 0.1
    treatment_effect = dic["T"] * (0.3 + interaction_effects)
    noise = np.random.normal(0, 0.5)
    prob_visit = np.clip(baseline_effect + treatment_effect, 0, 1)
    dic["visit"] = np.random.binomial(1, prob_visit)
    return dic

def generate_conversion(dic, seed=42):
    np.random.seed(seed)
    interaction_effects_purchase = sigmoid(np.sum(dic["features"], axis=1))
    baseline_effect_purchase = 0.1 + dic["features"][:, 5] * 0.3 + dic["features"][:, 7] * 0.3
    treatment_effect_purchase = dic["T"] * (0.2 + interaction_effects_purchase)
    noise = np.random.normal(0, 0.5)
    prob_purchase = np.clip(baseline_effect_purchase + treatment_effect_purchase, 0, 1)
    dic["purchase"] = np.where(dic["visit"] == 1, np.random.binomial(1, prob_purchase), 0)
    return dic

def predict_outcome(dic):
    dic_t0 = {k: v[dic["T"]==0] for k, v in dic.items()}
    dic_t1 = {k: v[dic["T"]==1] for k, v in dic.items()}
    mu_r_0 = LGBMRegressor(verbose=-1).fit(dic_t0["features"], dic_t0["purchase"])
    mu_r_1 = LGBMRegressor(verbose=-1).fit(dic_t1["features"], dic_t1["purchase"])
    mu_c_0 = LGBMRegressor(verbose=-1).fit(dic_t0["features"], dic_t0["visit"])
    mu_c_1 = LGBMRegressor(verbose=-1).fit(dic_t1["features"], dic_t1["visit"])
    return mu_r_0, mu_r_1, mu_c_0, mu_c_1

def preprocess_data(dic, mu_r_0, mu_r_1, mu_c_0, mu_c_1):
    X, T, y_r, y_c = dic["features"], dic["T"], dic["purchase"], dic["visit"]
    dic["y_r_ipw"] = np.where(T==1, dic["purchase"] / dic["T_prob"], dic["purchase"] / (1 - dic["T_prob"]))
    dic["y_c_ipw"] = np.where(T==1, dic["visit"] / dic["T_prob"], dic["visit"] / (1 - dic["T_prob"]))
    dic["y_r_dr"] = np.where(T==1, (dic["purchase"] - mu_r_1.predict(X)) / dic["T_prob"] + mu_r_1.predict(X), (dic["purchase"] - mu_r_0.predict(X)) / (1 - dic["T_prob"]) + mu_r_0.predict(X))
    dic["y_c_dr"] = np.where(T==1, (dic["visit"] - mu_c_1.predict(X)) / dic["T_prob"] + mu_c_1.predict(X), (dic["visit"] - mu_c_0.predict(X)) / (1 - dic["T_prob"]) + mu_c_0.predict(X))
    return dic

def split_data(dic, seed):
    X_train, X_test, T_train, T_test, y_r_train, y_r_test, y_c_train, y_c_test, y_r_ipw_train, y_r_ipw_test, y_c_ipw_train, y_c_ipw_test, y_r_dr_train, y_r_dr_test, y_c_dr_train, y_c_dr_test = train_test_split(
        dic["features"], dic["T"], dic["purchase"], dic["visit"], dic["y_r_ipw"], dic["y_c_ipw"], dic["y_r_dr"], dic["y_c_dr"], train_size=0.7, random_state=seed, stratify=dic["T"]
    )
    return X_train, X_test, T_train, T_test, y_r_train, y_r_test, y_c_train, y_c_test, y_r_ipw_train, y_c_ipw_train, y_r_dr_train, y_c_dr_train

class CustomDataset(Dataset):
    def __init__(self, X_treated, y_r_treated, y_c_treated, X_control, y_r_control, y_c_control):
        self.X_treated = X_treated
        self.y_r_treated = y_r_treated
        self.y_c_treated = y_c_treated
        self.X_control = X_control
        self.y_r_control = y_r_control
        self.y_c_control = y_c_control

        # teratment_idxからcontrol_idxをランダム二つ選択する辞書を作成
        self.treatment_idx_to_control_idx = {}
        # unused_control_idx = set(list(range(len(self.X_control))))
        ununsed_control_idx = list(range(len(self.X_control)))
        for i in range(len(self.X_treated)):
            control_idx_i = np.random.choice(ununsed_control_idx, 2, replace=False)
            self.treatment_idx_to_control_idx[i] = control_idx_i
            # unused_control_idx -= set(control_idx_i)

    
    def __len__(self):
        return len(self.X_treated)
    
    def __getitem__(self, idx):
        # treatmentのデータを取得
        X_treated = self.X_treated[idx]
        y_r_treated = self.y_r_treated[idx]
        y_c_treated = self.y_c_treated[idx]

        # controlのデータを取得
        control_idx = self.treatment_idx_to_control_idx[idx]
        X_control = self.X_control[control_idx]
        y_r_control = self.y_r_control[control_idx]
        y_c_control = self.y_c_control[control_idx]

        return X_treated, y_r_treated, y_c_treated, X_control, y_r_control, y_c_control
    

class CustomCollator:
    def __call__(self, batch):
        # バッチを作成
        X_treated = torch.tensor([x[0] for x in batch], dtype=torch.float32)
        y_r_treated = torch.tensor([x[1] for x in batch], dtype=torch.float32)
        y_c_treated = torch.tensor([x[2] for x in batch], dtype=torch.float32)

        # controlは2つあるので、それぞれのデータを取得
        X_control = torch.tensor([x[3] for x in batch], dtype=torch.float32)
        y_r_control = torch.tensor([x[4] for x in batch], dtype=torch.float32)
        y_c_control = torch.tensor([x[5] for x in batch], dtype=torch.float32)

        # reshape
        N, D = X_treated.shape
        X_control = X_control.reshape(N * 2, D)
        y_r_control = y_r_control.reshape(N * 2, 1)
        y_c_control = y_c_control.reshape(N * 2, 1)

        return X_treated, y_r_treated, y_c_treated, X_control, y_r_control, y_c_control



def loader(X_train, T_train, y_r_train, y_c_train):
    # treatmentとcontrolのデータを分割
    treatment_mask = T_train == 1
    X_train_treated = X_train[treatment_mask]
    y_r_train_treated = y_r_train[treatment_mask]
    y_c_train_treated = y_c_train[treatment_mask]

    control_mask = T_train == 0
    X_train_control = X_train[control_mask]
    y_r_train_control = y_r_train[control_mask]
    y_c_train_control = y_c_train[control_mask]

    # polarsに変換
    X_train_treated = pl.from_pandas(X_train_treated)
    y_r_train_treated = pl.from_pandas(y_r_train_treated)
    y_c_train_treated = pl.from_pandas(y_c_train_treated)

    X_train_control = pl.from_pandas(X_train_control)
    y_r_train_control = pl.from_pandas(y_r_train_control)
    y_c_train_control = pl.from_pandas(y_c_train_control)

    # import pdb; pdb.set_trace()

    # # データをテンソルに変換
    # X_train_tensor_treated = torch.from_numpy(X_train_treated.values).float().clone()
    # y_r_train_tensor_treated = torch.from_numpy(y_r_train_treated.values).float().clone()
    # y_c_train_tensor_treated = torch.from_numpy(y_c_train_treated.values).float().clone()

    # X_train_tensor_control = torch.from_numpy(X_train_control.values).float().clone()
    # y_r_train_tensor_control = torch.from_numpy(y_r_train_control.values).float().clone()
    # y_c_train_tensor_control = torch.from_numpy(y_c_train_control.values).float().clone()

    # import pdb; pdb.set_trace()
    # treatment_mask = T_train_tensor == 1
    # import pdb; pdb.set_trace()
    # X_train_tensor_treated = X_train_tensor[treatment_mask, :]
    # y_r_train_tensor_treated = y_r_train_tensor[treatment_mask, :]
    # y_c_train_tensor_treated = y_c_train_tensor[treatment_mask, :]

    # import pdb; pdb.set_trace()
    # control_mask = T_train_tensor == 0
    # X_train_tensor_control = X_train_tensor[control_mask]
    # y_r_train_tensor_control = y_r_train_tensor[control_mask]
    # y_c_train_tensor_control = y_c_train_tensor[control_mask]

    # データをテンソルに変換してDatasetを作成
    import pdb; pdb.set_trace()
    ds = CustomDataset(
        X_train_treated, y_r_train_treated, y_c_train_treated,
        X_train_control, y_r_train_control, y_c_train_control
    )
    collator = CustomCollator()

    import pdb; pdb.set_trace()
    # DataLoaderの定義
    dl = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=collator)
    

    return dl


# 非線形モデルの定義
class NonLinearModel(nn.Module):
    def __init__(self, input_dim):
        super(NonLinearModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 30)
        self.fc2 = nn.Linear(30, 15)
        self.fc3 = nn.Linear(15, 5)
        self.fc4 = nn.Linear(5, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

# 損失関数の定義
def custom_loss(y_r, y_c, q, group_size):
    q = torch.clamp(q, 1e-6, 1 - 1e-6)
    logit_q = torch.log(q / (1 - q))
    loss = -torch.sum(y_r * logit_q + y_c * torch.log(1 - q)) / group_size
    return loss

def get_loss(num_epochs, lr, X_train, dl):
    model = NonLinearModel(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    # 学習ループ
    for epoch in tqdm(range(num_epochs), desc='Training'):
        model.train()
        total_loss = 0
        count_batches = 0

        average_loss = 0
        total = len(dl)
        desc = f'Epoch {epoch} AVG Loss: {average_loss:.4f}'
        for x_1, y_r_1, y_c_1, x_0, y_r_0, y_c_0 in tqdm(dl, total=total, desc=desc, leave=False):
            optimizer.zero_grad()
            q_1 = model(x_1)
            q_0 = model(x_0)
            loss_1 = custom_loss(y_r_1, y_c_1, q_1, x_1.size(0))
            loss_0 = custom_loss(y_r_0, y_c_0, q_0, x_0.size(0))
            loss = loss_1 - loss_0
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count_batches += 1

        average_loss = total_loss / count_batches
        loss_history.append(average_loss)
    return model, loss_history

# 評価
def get_roi(model, X_test):
    model.eval()
    with torch.no_grad():
        q_test = model(torch.tensor(X_test.values, dtype=torch.float32))
        roi_direct = q_test.numpy()
        roi_direct = roi_direct.reshape(1, -1)[0]
        return roi_direct
    
def plot_loss(loss_history):
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def get_roi_tpmsl(X_train, y_r_train, y_c_train, T_train, X_test):
    models = LGBMRegressor()
    # models = LinearRegression()
    S_learner_r = SLearner(overall_model = models)
    S_learner_r.fit(y_r_train, T_train, X = X_train)

    S_learner_c = SLearner(overall_model = models)
    S_learner_c.fit(y_c_train, T_train, X = X_train)

    # 効果の推定
    tau_r = S_learner_r.effect(X_test)
    tau_c = S_learner_c.effect(X_test)
    roi_tpmsl = tau_r / tau_c

    scaler = MinMaxScaler()
    roi_tpmsl = scaler.fit_transform(roi_tpmsl.reshape(-1, 1)).flatten()

    # roi_tpmsl = sigmoid(roi_tpmsl)
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

def main(predict_treatment=False):
    seed = 42
    n = 10000
    p = 10
    dic = {}
    num_epochs = 50
    lr = 0.0005
    dic = generate_data(n, p, dic, seed)
    dic = generate_treatment(dic, seed)
    dic = generate_visit(dic, seed)
    dic = generate_conversion(dic, seed)
    if predict_treatment:
        dic = predict_treatment(dic)
    mu_r_0, mu_r_1, mu_c_0, mu_c_1 = predict_outcome(dic)
    dic = preprocess_data(dic, mu_r_0, mu_r_1, mu_c_0, mu_c_1)
    X_train, X_test, T_train, T_test, y_r_train, y_r_test, y_c_train, y_c_test, y_r_ipw_train, y_c_ipw_train, y_r_dr_train, y_c_dr_train = split_data(dic, seed)
    print("=====================================")
    dl = loader(X_train, T_train, y_r_train, y_c_train)
    print("=====================================")
    model, loss_history = get_loss(num_epochs, lr, X_train, dl)
    plot_loss(loss_history)
    roi_direct = get_roi(model, X_test)
    roi_tpmsl = get_roi_tpmsl(X_train, y_r_train, y_c_train, T_train, X_test)
    incremental_costs, incremental_values = calculate_values(roi_direct, T_test, y_r_test, y_c_test)
    incremental_costs_tpmsl, incremental_values_tpmsl = calculate_values(roi_tpmsl, T_test, y_r_test, y_c_test)
    plt.plot(incremental_costs / max(incremental_costs), incremental_values / max(incremental_values), label="Direct Method", marker="x")
    plt.plot(incremental_costs_tpmsl / max(incremental_costs_tpmsl), incremental_values_tpmsl / max(incremental_values_tpmsl), label="TPMSL", marker="o")
    # 対角線を描画
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Incremental Costs")
    plt.ylabel("Incremental Values")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main(predict_treatment=False)