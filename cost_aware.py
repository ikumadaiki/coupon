# print('cost_aware.py')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

import init_data
df = init_data.get_data()

# パラメータの設定
seed = 42

# データの分割
def split_data(df, seed):
    # 訓練データとテストデータに分割
    from sklearn.model_selection import train_test_split

    X = df.drop(['treatment', 'exposure', 'visit', 'conversion'], axis=1)
    # Xの特徴量を正規化
    X = (X - X.mean()) / X.std()
    T = df['treatment']
    y_r = df['conversion']
    y_c = df['visit']

    X_train, X_test, T_train, T_test, y_r_train, y_r_test, y_c_train, y_c_test = train_test_split(
        X, T, y_r, y_c, train_size=0.7, random_state=seed, stratify=T
    )

    # インデックスをリセット
    T_test = T_test.reset_index(drop=True)
    y_r_test = y_r_test.reset_index(drop=True)
    y_c_test = y_c_test.reset_index(drop=True)

    # データをテンソルに変換
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_r_train_tensor = torch.tensor(y_r_train.values, dtype=torch.float32)
    y_c_train_tensor = torch.tensor(y_c_train.values, dtype=torch.float32)
    T_train_tensor = torch.tensor(T_train.values, dtype=torch.float32)

    # データをテンソルに変換してDatasetを作成
    dataset_1 = TensorDataset(X_train_tensor[T_train_tensor == 1], y_r_train_tensor[T_train_tensor == 1], y_c_train_tensor[T_train_tensor == 1])
    dataset_0 = TensorDataset(X_train_tensor[T_train_tensor == 0], y_r_train_tensor[T_train_tensor == 0], y_c_train_tensor[T_train_tensor == 0])

    # DataLoaderの定義
    loader_1 = DataLoader(dataset_1, batch_size=170, shuffle=True)
    loader_0 = DataLoader(dataset_0, batch_size=30, shuffle=True)

    return X_train, X_test, T_train, T_test, y_r_train, y_r_test, y_c_train, y_c_test, loader_1, loader_0

X_train, X_test, T_train, T_test, y_r_train, y_r_test, y_c_train, y_c_test, loader_1, loader_0 = split_data(df, seed)


# 非線形モデルの定義
class NonLinearModel(nn.Module):
    def __init__(self, input_dim):
        super(NonLinearModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 36)
        self.fc2 = nn.Linear(36, 18)
        self.fc3 = nn.Linear(18, 6)
        self.fc4 = nn.Linear(6, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

# 損失関数の定義
def custom_loss(y_r, y_c, q, group_size):
    q = torch.clamp(q, 1e-8, 1 - 1e-8)
    logit_q = torch.log(q / (1 - q))
    loss = -torch.sum(y_r * logit_q + y_c * torch.log(1 - q)) / group_size
    return loss

def get_loss(num_epochs, lr=0.0001):
    model = NonLinearModel(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    # 学習ループ
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        count_batches = 0

        for (x_1, y_r_1, y_c_1), (x_0, y_r_0, y_c_0) in tqdm(zip(loader_1, loader_0)):
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
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Average Loss: {average_loss}')
    return model, loss_history

# 評価
def get_roi(model):
    model.eval()
    with torch.no_grad():
        q_test = model(torch.tensor(X_test.values, dtype=torch.float32))
        roi_direct = q_test.numpy()
        roi_direct = roi_direct.reshape(1, -1)[0]
        return roi_direct
    
def get_roi_tpmsl():
    # 必要なライブラリのインポート
    from sklearn.ensemble import RandomForestRegressor
    from econml.metalearners import SLearner
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import MinMaxScaler
    from lightgbm import LGBMRegressor


    # モデルの構築
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

def calculate_values(roi_scores):
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