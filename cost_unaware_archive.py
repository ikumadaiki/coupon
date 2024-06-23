# print('cost_unaware.py')

import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# from torch.optim import lr_scheduler
# from tqdm import tqdm
# from torch.optim.lr_scheduler import StepLR

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


    X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(
        X, T, y, train_size=0.7, random_state=seed, stratify=T
    )

    # インデックスをリセット
    T_test = T_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)


    # # データをテンソルに変換
    # X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    # y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    # T_train_tensor = torch.tensor(T_train.values, dtype=torch.float32)


    # # データをテンソルに変換してDatasetを作成
    # dataset_1 = TensorDataset(X_train_tensor[T_train_tensor == 1], y_train_tensor[T_train_tensor == 1])
    # dataset_0 = TensorDataset(X_train_tensor[T_train_tensor == 0], y_train_tensor[T_train_tensor == 0])

    # # DataLoaderの定義
    # loader_1 = DataLoader(dataset_1, batch_size=680, shuffle=True)
    # loader_0 = DataLoader(dataset_0, batch_size=120, shuffle=True)

    return X_train, X_test, T_train, T_test, y_train, y_test

X_train, X_test, T_train, T_test, y_train, y_test = split_data(df, seed)


# モデルの構築
# class Model(nn.Module):
#     def __init__(self, input_dim):
#         super(Model, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 3)
#         self.fc2 = nn.Linear(3, 1)
#         # self.fc3 = nn.Linear(18, 6)
#         # self.fc4 = nn.Linear(6, 1)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         # x = torch.relu(self.fc2(x))
#         # x = torch.relu(self.fc3(x))
#         x = self.fc2(x)
#         return x
    

# def custom_loss(y, q, group_size):
#     q = torch.clamp(q, 1e-8, 1 - 1e-8)
#     loss = -torch.sum(y * torch.log(q)) / group_size
#     return loss


# def get_loss(num_epochs, lr=0.001):
#     print(lr)
#     model = Model(X_train.shape[1])
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     # lambda_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
#     loss_history = []

#     for epoch in tqdm(range(num_epochs)):
#         model.train()
#         total_loss = 0
#         count_batches = 0

#         for (x_1, y_1), (x_0,y_0) in zip(loader_1, loader_0):
#             optimizer.zero_grad()
#             s_1 = model(x_1)
#             s_0 = model(x_0)
#             q_1 = torch.softmax(s_1, dim=0)
#             q_0 = torch.softmax(s_0, dim=0)
#             loss_1 = custom_loss(y_1, q_1, x_1.size(0))
#             loss_0 = custom_loss(y_0, q_0, x_0.size(0))
#             loss = loss_1 - loss_0
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#             count_batches += 1

#         average_loss = total_loss / count_batches
#         loss_history.append(average_loss)
#         # lambda_scheduler.step()

#         if epoch % 5 == 0:
#             print(f'Epoch {epoch}: Average Loss: {average_loss}')

    # return model, loss_history

# def get_tau_direct_nn(model):
#     model.eval()
#     # 最適な重み

#     with torch.no_grad():
#         s = model(torch.tensor(X_test.values, dtype=torch.float32))
#         tau_direct_nn = torch.exp(s)
#         tau_direct_nn = tau_direct_nn.numpy().reshape(-1)
#     return tau_direct_nn


# N1 = np.sum(T_train == 1)
# N0 = np.sum(T_train == 0)

# y_train_1 = y_train[T_train == 1]
# y_train_0 = y_train[T_train == 0]

# X_train_1 = X_train[T_train == 1]
# X_train_0 = X_train[T_train == 0]

# def get_tau_direct():
#     import numpy as np
#     from scipy.optimize import minimize


#     # 初期重み
#     w_initial = np.random.rand(X_train.shape[1])

#     # 最適化
#     result = minimize(loss_function, w_initial, method='BFGS')

#     if result.success:
#         print("Optimized weights:", result.x)
#         print("Minimum loss:", result.fun)
#     else:
#         print("Optimization failed:", result.message)

#     # sとqの最適解の導出
#     w = result.x
#     scores = np.dot(X_test, w)
#     probabilities = softmax(scores)
#     tau_direct = probabilities * np.sum(np.exp(scores))

#     return tau_direct


# # 損失関数 L(s)
# def loss_function(w):
#     scores_1 = np.dot(X_train_1, w)
#     scores_0 = np.dot(X_train_0, w)
#     probabilities_1 = softmax(scores_1)
#     probabilities_0 = softmax(scores_0)
#     probabilities_1 = np.clip(probabilities_1, 1e-8, 1 - 1e-8)
#     probabilities_0 = np.clip(probabilities_0, 1e-8, 1 - 1e-8)
#     return -np.sum(y_train_1 * np.log(probabilities_1)) / N1 + np.sum(y_train_0 * np.log(probabilities_0)) / N0

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