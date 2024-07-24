import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch_geometric.nn import GCNConv,GATConv,SAGEConv
from torch_geometric.data import Data
from sklearn.metrics import f1_score, recall_score, precision_score,confusion_matrix,roc_curve


import pandas as pd

def load_node_csv(df, index_col,y=None, encoders=None,**kwargs):
    mapping = {index: i for i, index in enumerate(df[index_col].unique())}
    x = df.drop(y,axis=1)
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
#    tx_e.drop('index',axis=1,inplace=True)
except:
    pass

# read data to graph
tx_x, tx_y, tx_mapping = load_node_csv(tx,index_col='txId',y='class')

edge_index,_ = load_edge_csv(tx_e, 'txId1',  tx_mapping , 'txId2',  tx_mapping ,
                  encoders=None)

data = Data(x=tx_x,y=tx_y,edge_index=edge_index)

train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
train_mask[0:154843] = True

test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
test_mask[154843:] = True

data.train_mask = train_mask
data.test_mask = test_mask


# GNN model
class GNN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features,num_features*3 )
        self.conv2 = GCNConv(num_features*3, num_classes)
        # self.conv3 = GCNConv(12, num_classes)
        # self.double()
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN(num_features=tx.shape[1]-1, num_classes=3).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    optimizer.zero_grad()
    out = model(data)
    #print(out)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if epoch %100 == 0:
      print(f"epoch:{epoch+1}, loss:{loss.item()}")





def eval_node_classifier(model, graph, mask):
    model.eval()
    pred = model(graph).argmax(dim=1)
    #correct = (pred[mask] == graph.y[mask]).sum()
    #acc = int(correct) / int(mask.sum())
    y_pred = pred[mask]
    test_y = graph.y[mask]
    # recall = recall_score(graph.y[mask], pred[mask], average='binary')
    # f1 = f1_score(graph.y[mask], pred[mask], average='micro')
    # precision = precision_score(graph.y[mask], pred[mask], average='micro')
    f1 = f1_score(test_y , y_pred, average='binary', pos_label=1)
    recall = recall_score(test_y , y_pred, average='binary', pos_label=1)
    precision= precision_score(test_y , y_pred, average='binary', pos_label=1)
    acc = accuracy_score(test_y , y_pred)
    
    drawROC(test_y,y_pred)

    return acc,recall,f1,precision

def drawROC(test_y,y_pred):
    fpr, tpr, thersholds = roc_curve(test_y,y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)
 
    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
    
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('roc-actor.png')


test_acc,test_recall,test_f1,test_precision = eval_node_classifier(model, data, data.test_mask)
print(f'Accuracy: {test_acc :.4f}')
print(f'Recall: {test_recall :.4f}')
print(f'F1: {test_f1 :.4f}')
print(f'Precision: {test_precision :.4f}')

torch.save(model, './model/tx.pth')

# torch.save(model.state_dict(), 'GNN-actor-2.pth')



