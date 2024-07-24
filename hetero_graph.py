
import pandas as pd
import numpy as np


import torch
import scipy.sparse as sp
import torch.nn.functional as F

import torch_geometric.transforms as T

# from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import HANConv, Linear,HeteroConv
from torch_geometric.data import Data, HeteroData,DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score,confusion_matrix,roc_curve,accuracy_score,auc


import warnings
warnings.filterwarnings("ignore")

df_AddrTx_edgelist = pd.read_csv("data/AddrTx_edgelist.csv")

df_TxAddr_edgelist = pd.read_csv("data/TxAddr_edgelist.csv")

actor = pd.read_csv("data/wallets_features_classes_combined.csv")
df_edge = pd.read_csv("data/AddrAddr_edgelist.csv")

actor.dropna(inplace=True)
actor = actor[actor['class']!=3].reindex()

actList = actor['address'].unique().tolist()

l = actList
df_edge = df_edge[df_edge['input_address'].isin(l)]
df_edge = df_edge[df_edge['output_address'].isin(l)]
df_AddrTx_edgelist = df_AddrTx_edgelist[df_AddrTx_edgelist['input_address'].isin(l)]
df_TxAddr_edgelist = df_TxAddr_edgelist[df_TxAddr_edgelist['output_address'].isin(l)]


tx = pd.read_csv("df_tx.csv")

tx_e= pd.read_csv("df_tx_edge.csv")

tx = tx[tx['class']!=3]
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

# tx.describe()

l = txIdlist
df_AddrTx_edgelist = df_AddrTx_edgelist[df_AddrTx_edgelist['txId'].isin(l)]
df_TxAddr_edgelist = df_TxAddr_edgelist[df_TxAddr_edgelist['txId'].isin(l)]


# import networkx as nx

# G = nx.from_pandas_edgelist(df_edge, source='input_address', target='output_address')

# # Calculate betweenness centrality
# betweenness_centrality = nx.betweenness_centrality(G)

# # Calculate closeness centrality
# closeness_centrality = nx.closeness_centrality(G)

# # Add the centrality measures to the actor dataframe
# actor['betweenness_centrality'] = [betweenness_centrality.get(addr, 0) for addr in actor['address']]
# actor['closeness_centrality'] = [closeness_centrality.get(addr, 0) for addr in actor['address']]

# # Display the actor dataframe with the added centrality measures
# actor.head()



"""# node loading for homogeneous data"""

def load_node_csv(df, index_col,y=None, encoders=None,**kwargs):
    mapping = {index: i for i, index in enumerate(df[index_col].unique())}
    if y != None:
      x = df.drop([y],axis=1)
      y = torch.tensor(df[y].values.astype(np.int64))#.int()
    x = df.drop([index_col,'Time step'],axis=1)
    x = torch.tensor(x.values.astype(np.float32))

    return x,y, mapping


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

# 归一化
for column in actor.columns[3:]:
    feature = np.array(actor[column]).reshape(-1,1)
    scaler = MinMaxScaler()
    scaler.fit(feature)
    feature_scaled = scaler.transform(feature)
    actor[column] = feature_scaled.reshape(1,-1)[0]



# 数据处理
actor['class'] = actor['class'].replace(2,0)
index_range_act = actor[actor['Time step'].between(1, 34)].index.tolist()

"""### tx
keeping the class in the tx.x since the task is to classify wallet nodes
"""

tx_x,_, tx_mapping = load_node_csv(tx,index_col='txId')

tx_edge_index,_ = load_edge_csv(tx_e, 'txId1',  tx_mapping , 'txId2',  tx_mapping ,
                  encoders=None)

"""### wallet (actor)"""

df_actor_x, df_actor_y, df_actor_mapping = load_node_csv(actor,index_col='address',y='class')

act_edge_index,_ = load_edge_csv(df_edge, 'input_address',  df_actor_mapping , 'output_address',  df_actor_mapping ,
                  encoders=None)

"""# heterogeneous graph using PyG

### train set and test set
"""



data = HeteroData()
data['wallet'].x = df_actor_x
data['wallet'].y = df_actor_y
data['tx'].x = tx_x



train_mask = torch.zeros(len(df_actor_x), dtype=torch.bool)
train_mask[0:len(index_range_act)] = True
test_mask = torch.zeros(len(df_actor_x), dtype=torch.bool)
test_mask[len(index_range_act):] = True
data['wallet'].train_mask = train_mask
data['wallet'].test_mask = test_mask


act_tx,_ = load_edge_csv(df_AddrTx_edgelist, 'input_address',  df_actor_mapping , 'txId',  tx_mapping ,
                  encoders=None)
tx_act,_ = load_edge_csv(df_TxAddr_edgelist, 'txId',  tx_mapping ,'output_address',  df_actor_mapping ,
                  encoders=None)

data['wallet', 'to', 'tx'].edge_index = act_tx
data['tx', 'to', 'wallet'].edge_index = tx_act

data['wallet', 'interact with', 'wallet'].edge_index = act_edge_index
data['tx', 'to', 'wallet'].edge_index = tx_act

data

"""# HAN"""

from typing import Dict, List, Union
from torch import nn

metapaths = [('wallet', 'to', 'tx'),
             ('tx', 'to', 'wallet'),
            ('wallet', 'interact with', 'wallet')]

class HAN(nn.Module)／
    def __init__(self, in_channels: Union[int, Dict[str, int]],
                 out_channels: int, hidden_channels=128, heads=8):
        super().__init__()
        self.han_conv = HANConv(in_channels, hidden_channels, heads=heads,
                                dropout=0.1, metadata=data.metadata())
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        out = self.han_conv(x_dict, edge_index_dict)
        out = self.lin(out['wallet'])
        return out



def train(model,weight,optimizer) -> float:
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data['wallet'].train_mask
    loss = F.cross_entropy(out[mask], data['wallet'].y[mask],weight=torch.tensor([0.1, weight]))
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model): #-> List[float]:
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)
    accs = []
    for split in ['train_mask', 'test_mask']:
        true_labels = data['wallet'].y[data['wallet'][split]].tolist()
        predictions = pred[data['wallet'][split]].tolist()
        mask = data['wallet'][split]
        acc = accuracy_score(true_labels,predictions)
        #acc = (pred[mask] == data['wallet'].y[mask]).sum() / mask.sum()
        accs.append(float(acc))


    # Compute precision, recall, F1 score, and micro-F1
    #print('true_labels: ',true_labels)
    true_labels = data['wallet'].y[data['wallet']['test_mask']].tolist()
    predictions = pred[data['wallet']['test_mask']].tolist()
    precision = precision_score(true_labels, predictions,pos_label=1)
    recall = recall_score(true_labels, predictions,pos_label=1)
    f1 = f1_score(true_labels, predictions)
    micro_f1 = f1_score(true_labels, predictions, average='micro')

    return accs, precision, recall, f1, micro_f1

def runningHan(hidden_channels,epochs,weight):
    model = HAN(in_channels=-1,hidden_channels=hidden_channels, out_channels=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, model = data.to(device), model.to(device)

    with torch.no_grad():  # Initialize lazy modules.
        out = model(data.x_dict, data.edge_index_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

    start_patience = patience = 100
    for epoch in range(1,epochs):
        loss = train(model,weight,optimizer)
        train_acc, precision, recall, f1, micro_f1 = test(model)
        if epoch % 5 == 0 or epoch == 1:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc[0]:.4f}, '
                    f'Test: {train_acc[1]:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, '
                    f'F1: {f1:.4f}, Micro-F1: {micro_f1:.4f}')
        
        patience -= 1

        if patience <= 0:
            print('Stopping training as validation accuracy did not improve '
                f'for {start_patience} epochs')
            break

for hidden_channels in [20,50,100]:
    for weight in [1.0,2.0,2.5,3.0]:
        runningHan(hidden_channels,epochs=200,weight=weight)