# 必要ライブラリのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import matplotlib.style as style
import seaborn as sns

# Jupyter Notebookのマジックコマンドは削除し、必要に応じて設定を調整
plt.style.use('default')  # ggplotやfivethirtyeightなど他のスタイルを使用できます
sns.set()  # seabornのスタイルを使用

# データの読み込み
df = pd.read_csv('criteo-uplift-v2.1.csv')

def get_data():
    return df