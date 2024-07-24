
import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
# import torch_geometric.transforms as T

from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import GCNConv,GATConv,SAGEConv
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
from sklearn.metrics import f1_score, recall_score, precision_score,confusion_matrix,roc_curve,accuracy_score,auc


def load_node_csv(df, index_col,y=None, encoders=None,**kwargs):
    mapping = {index: i for i, index in enumerate(df[index_col].unique())}
    x = df.drop([y,index_col,'Time step'],axis=1)
    x = torch.tensor(x.values.astype(np.float32))
    y = torch.tensor(df[y].values.astype(np.int64))#.int()
    # if encoders is not None:
    #     xs = [encoder(df[col]) for col, encoder in encoders.items()]
    #     x = torch.cat(xs, dim=-1)

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


tx = pd.read_csv('trainData/df_tx.csv')
tx_e = pd.read_csv('trainData/df_tx_edge.csv')

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
index_range = tx[tx['Time step'].between(1, 35)].index.tolist()

for column in tx.columns[2:184]:
    feature = np.array(tx[column]).reshape(-1,1)
    scaler = MinMaxScaler()
    scaler.fit(feature)
    feature_scaled = scaler.transform(feature)
    tx[column] = feature_scaled.reshape(1,-1)[0]

tx.describe()

# read data to graph
tx_x, tx_y, tx_mapping = load_node_csv(tx,index_col='txId',y='class')

edge_index,_ = load_edge_csv(tx_e, 'txId1',  tx_mapping , 'txId2',  tx_mapping ,
                  encoders=None)

data = Data(x=tx_x,y=tx_y,edge_index=edge_index)


train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
train_mask[0:len(index_range)] = True

test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
test_mask[len(index_range):] = True

data.train_mask = train_mask
data.test_mask = test_mask


# GAT model

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

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# GATmodel = GAT_Net(data.num_node_features, 16, 2, heads=4).to(device)
# optimizer = torch.optim.Adam(GATmodel.parameters(), lr=0.001)

# GATmodel.train()
# for epoch in range(200):
#     optimizer.zero_grad()
#     out = GATmodel(data)
#     loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask],weight=label_weight)
#     loss.backward()
#     optimizer.step()
    
def drawROC(test_y,y_pred):
    fpr, tpr, thersholds = roc_curve(test_y,y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def eval_node_classifier(model, graph, mask):
    model.eval()
    pred = model(graph).argmax(dim=1)
    #correct = (pred[mask] == graph.y[mask]).sum()
    #acc = int(correct) / int(mask.sum())

    y_pred = pred[mask]
    test_y = graph.y[mask]
    print('y_pred:',y_pred)
    print('\ntest_y',test_y)
    roc_auc = drawROC(test_y,y_pred)
    f1 = f1_score(test_y , y_pred, average='binary', pos_label=1)
    recall = recall_score(test_y , y_pred, average='binary', pos_label=1)
    precision= precision_score(test_y , y_pred, average='binary', pos_label=1)
    acc = accuracy_score(test_y , y_pred)

    # test_acc,test_recall,test_f1,test_precision = eval_node_classifier(model, data, data.test_mask)
    print(f'Accuracy: {acc :.4f}')
    print(f'Recall: {recall :.4f}')
    print(f'Precision: {precision :.4f}')
    print(f'F1: {f1 :.4f}')
    print(f'AUC_ROC: {roc_auc :.4f}')
    print('\n')
    
def GATtrain(epoches,weight,head,hidden_layer,data):
    label_weight = torch.tensor([1, weight],dtype=torch.int64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    label_weight = torch.tensor([1, weight],dtype=torch.float)
    model = GAT_Net(features=tx.shape[1]-3, hidden=hidden_layer, classes=2,heads=head).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epoches):
        optimizer.zero_grad()
        out = model(data)
        # if epoch == 0:
        #   print('out:',out[data.train_mask])
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask],weight=label_weight.float())
        loss.backward()
        optimizer.step()
        if epoch %100 == 0:
            print(f"epoch:{epoch+1}, loss:{loss.item()}")

    print('tx model with %d heads, %f weight, %d hidden layer, and%d epoche'%(head,weight,hidden_layer,epoch+1))
    eval_node_classifier(model, data, data.test_mask)

for head in [1,2,4]:
  for weight in [1,3,5]:
    for hidden_layer in [16,100,200,360,500]:
        GATtrain(500,weight,head,hidden_layer=hidden_layer,data=data)
    
    
    




    # return acc,recall,f1,precision


    # plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)

    # plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend(loc="lower right")
    # plt.show()
    # plt.savefig('roc-actor with val old weight 3 times features and 2000 epoche.png')





# torch.save(model, './model/tx-2000.pth')

