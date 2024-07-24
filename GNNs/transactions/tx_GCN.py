
import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.nn import BatchNorm1d
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.data import Data,DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score,confusion_matrix,roc_curve,accuracy_score,auc


import pandas as pd

def load_node_csv(df, index_col,y=None, encoders=None,**kwargs):
    mapping = {index: i for i, index in enumerate(df[index_col].unique())}
    x = df.drop([y,index_col],axis=1)
    x = torch.tensor(x.values.astype(np.float32))
    y = torch.tensor(df[y].values.astype(np.int64))#.int()

    return x, y, mapping

def load_edge_csv(df, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


tx = pd.read_csv("trainData/df_tx.csv")
tx_e = pd.read_csv("trainData/df_tx_edge.csv")

try:
    tx.drop('Unnamed: 0',axis=1,inplace=True)
except:
    pass

tx.dropna(inplace=True,how='any')
txIdlist = tx.txId.unique().tolist()
tx_e = tx_e[tx_e.txId1.isin(txIdlist)]
tx_e = tx_e[tx_e.txId2.isin(txIdlist)]

tx['class'] = tx['class'].replace(2,0)
tx = tx.sort_values(by=['Time step'])
index_range = tx[tx['Time step'].between(1, 34)].index.tolist()

for column in tx.columns[2:184]:
    feature = np.array(tx[column]).reshape(-1,1)
    scaler = MinMaxScaler()
    scaler.fit(feature)
    feature_scaled = scaler.transform(feature)
    tx[column] = feature_scaled.reshape(1,-1)[0]

tx.describe()

def create_graphs(tx, tx_e, time_steps):
    graphs = []
    for t in time_steps:
        tx_t = tx[tx['Time step'] == t]
        target_nodes = tx_t['txId'].tolist()
        tx_e_t = tx_e[(tx_e['txId1'].isin(target_nodes)) & (tx_e['txId2'].isin(target_nodes))]

        tx_t = tx_t.drop('Time step', axis=1)

        tx_t_x, tx_t_y, tx_t_mapping = load_node_csv(tx_t, index_col='txId', y='class')

        edge_index, _ = load_edge_csv(tx_e_t, 'txId1', tx_t_mapping, 'txId2', tx_t_mapping,
                                      encoders=None)

        graph_data = Data(x=tx_t_x, y=tx_t_y, edge_index=edge_index)
        graphs.append(graph_data)

    return graphs


    return graphs


# GNN model
class GNN(torch.nn.Module):
    def __init__(self, num_features, hidden_layer, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features,hidden_layer )
        self.conv2 = GCNConv(hidden_layer, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


# 創建訓練圖和測試圖
train_time_steps = list(range(1, 35))
test_time_steps = list(range(35, 50))

train_graphs = create_graphs(tx, tx_e, train_time_steps)
test_graphs = create_graphs(tx, tx_e, test_time_steps)

# 使用 DataLoader 進行批次處理
train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)



class GAT_Net(torch.nn.Module):
  def __init__(self, features, hidden, classes, heads=1):
    super(GAT_Net, self).__init__()
    self.gat1 = GATConv(features, hidden, heads=heads)
    self.gat2 = GATConv(hidden * heads, classes)
  def forward(self, data):
      x, edge_index = data.x, data.edge_index

      x = self.gat1(x, edge_index)
      x = F.relu(x)
      x = F.dropout(x, training=self.training)
      x = self.gat2(x, edge_index)

      return F.log_softmax(x, dim=1)


def drawROC(test_y,y_pred):
    fpr, tpr, thersholds = roc_curve(test_y,y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)
    return roc_auc




# 指定裝置為 GPU 如果可用，否則使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 定義損失函數和優化器
criterion = torch.nn.CrossEntropyLoss()
label_weight = torch.tensor([0.3,0.7],dtype=torch.float).to(device)
# data = data.to(device)
 

# 定義訓練和測試的函式
def train(model, criterion, train_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.to(device))
        loss = criterion(out, data.y.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = out.argmax(dim=1)
        total_correct += (pred == data.y).sum().item()
        total_samples += len(data.y)

    avg_loss = running_loss / len(train_loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def test(model, criterion, test_loader):
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data in test_loader:
            out = model(data)
            loss = criterion(out, data.y)

            running_loss += loss.item()
            pred = out.argmax(dim=1)
            total_correct += (pred == data.y).sum().item()
            total_samples += len(data.y)

    avg_loss = running_loss / len(test_loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy




for num_epochs in [100,200]:
    for hidden in [90,100,200,360]:
        for heads in [4,8,12]:
                print_every = 50
                # 初始化模型
                model = GAT_Net(features=tx.shape[1]-3, hidden=hidden, classes=2, heads=heads)

                for epoch in range(1, num_epochs + 1):
                    train_loss, train_acc = train(model,criterion, train_loader)

                    if epoch % print_every == 0:
                        test_loss, test_acc = test(model, criterion, test_loader)
                        print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

                # 測試模型
                model.eval()
                y_true = []
                y_pred = []
                for data in test_loader:
                    out = model(data)
                    pred = out.argmax(dim=1).tolist()  # 將預測結果轉換為 Python 列表
                    y_pred.extend(pred)  # 使用 extend() 方法添加到 y_pred 列表中
                    y_true.extend(data.y.tolist())  # 將真實標籤轉換為 Python 列表並添加到 y_true 中
            
            

                # 將預測結果和真實標籤轉換為 NumPy 數組
                y_true_np = np.array(y_true)
                y_pred_np = np.array(y_pred)
                # 計算指標
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                print('hidden layer:%d, heads:%d'%(hidden,heads))
                print("Final Metrics:")
                print("Accuracy:", accuracy)
                print("Precision:", precision)
                print("Recall:", recall)
                print("F1 Score:", f1)
                print('*'*40)
                del model,accuracy,precision,recall,f1
