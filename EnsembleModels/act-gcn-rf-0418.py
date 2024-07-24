import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import torch_geometric.transforms as T

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import GCNConv, Linear
from torch_geometric.data import Data
from sklearn.metrics import f1_score, recall_score, precision_score,confusion_matrix,roc_curve,accuracy_score,auc


import pandas as pd

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

def drawROC(test_y,y_pred):
    fpr, tpr, thersholds = roc_curve(test_y,y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)
    return roc_auc


# read data
df_actor = pd.read_csv('trainData/df_actor.csv')
df_edges = pd.read_csv('trainData/df_edges.csv')

try:
    df_actor.drop('Unnamed: 0',axis=1,inplace=True)
except:
    pass

# 尝试剔除不重要features
# df_actor.drop(['blocks_btwn_input_txs_median',
#  'first_sent_block',
#  'first_received_block',
#  'fees_mean',
#  'first_block_appeared_in',
#  'fees_total',
#  'fees_min',
#  'fees_as_share_min',
#  'blocks_btwn_txs_max',
#  'fees_max'],axis=1,inplace=True)

# 归一化
for column in df_actor.columns[3:]:
    feature = np.array(df_actor[column]).reshape(-1,1)
    scaler = MinMaxScaler()
    scaler.fit(feature)
    feature_scaled = scaler.transform(feature)
    df_actor[column] = feature_scaled.reshape(1,-1)[0]





# 数据处理
df_actor['class'] = df_actor['class'].replace(2,0)
index_range = df_actor[df_actor['Time step'].between(1, 34)].index.tolist()


train = df_actor.iloc[index_range]
test =  df_actor.drop(index=index_range)
print('train shape: ',train.shape[0],'\ntest shape: ',test.shape[0])

y_train = train['class']
X_train = train.drop(['class','address','Time step'],axis=1)
y_test = test['class']
X_test = test.drop(['class','address','Time step'],axis=1)


# read data to graph
df_actor_x, df_actor_y, df_actor_mapping = load_node_csv(df_actor,index_col='address',y='class')

edge_index,_ = load_edge_csv(df_edges, 'input_address',  df_actor_mapping , 'output_address',  df_actor_mapping ,
                  encoders=None)

data = Data(x=df_actor_x,y=df_actor_y,edge_index=edge_index)


train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
train_mask[0:len(index_range)] = True
test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
test_mask[len(index_range):] = True
data.train_mask = train_mask
data.test_mask = test_mask



# GNN model
class GNN(torch.nn.Module):
    def __init__(self, num_features, hidden_layer, output_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_layer)
        self.conv2 = GCNConv(hidden_layer, hidden_layer)
        self.linear = torch.nn.Linear(hidden_layer, output_channels) # 新增

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.1) # 新增
        x_feats = self.conv2(x, edge_index)
        x = F.relu(x_feats) # 新增
        x = F.dropout(x, training=self.training, p=0.1) # 新增
        x = self.linear(x) # 新增
        return x, x_feats

# evaluation 函数
def eval_classifier(y_pred,y_test):
    roc_auc = drawROC(y_test,y_pred)
    f1 = f1_score(y_test , y_pred, average='binary', pos_label=1)
    recall = recall_score(y_test , y_pred, average='binary', pos_label=1)
    precision= precision_score(y_test , y_pred, average='binary', pos_label=1)
    acc = accuracy_score(y_test , y_pred)

    print(f'Accuracy: {acc :.4f}')
    print(f'Recall: {recall :.4f}')
    print(f'Precision: {precision :.4f}')
    print(f'F1: {f1 :.4f}')
    print(f'AUC_ROC: {roc_auc :.4f}')
    
# gcn的评估函数
def eval_node_classifier(model, graph, mask):
    model.eval()
    pred = model(graph).argmax(dim=1)

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

def train(epoches,weight,hidden_layer,data,output_channels):
    # 设定输出评估权重
    ary = np.ones(output_channels-1)
    ary = np.append(ary,[weight])
    label_weight = torch.tensor(ary,dtype=torch.float)
    del ary
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gcn_model = GNN(num_features=data.x.size(dim=1), hidden_layer=hidden_layer, output_channels=output_channels).to(device)
    data = data.to(device)
    label_weight = label_weight.to(device)
    optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.001)

    for epoch in range(epoches):
        optimizer.zero_grad()
        out, _ = gcn_model(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask],weight=label_weight)
        loss.backward()
        optimizer.step()
        if epoch %100 == 0:
            print(f"epoch:{epoch+1}, loss:{loss.item()}")
    return gcn_model

for hidden_layer in [100,150,200]:
    gcn_model = train(epoches=2000,weight=1,hidden_layer=hidden_layer,data=data,output_channels=50)#gcn_model(data.x, edge_index)
    gcn_out , convoluted_features = gcn_model(data)
    rf_convoluted = convoluted_features[train_mask].detach()
    convoluted = gcn_out[train_mask].detach()

    rf_conv = RandomForestClassifier(n_estimators=50,random_state=42)
    rf_conv.fit(rf_convoluted, data.y[train_mask])
    pred_conv = rf_conv.predict(convoluted_features[test_mask].detach().numpy())
    print('gcn_out: ',gcn_out)
    print('convoluted: ',convoluted)
    print('Resutl With GCN+RF')
    eval_classifier(pred_conv,y_test)
# print('Result with only GCN\n')
# eval_classifier(convoluted,y_test)
#eval_node_classifier(gcn_model,data, data.test_mask)



# for h_l in [100]:
#     for weight in [3]:
#         for output_channels in [16,57]:       
#                 # Generate convoluted features using trained GCN model
#                 convoluted_features = 
#                  = convoluted_features(data)
                

#                 rf_conv = RandomForestClassifier(n_estimators=50,random_state=42)
                

print('\n','*'*50)
print('Resutl With only RF')
 # Train Random Forest models
rf_no_conv = RandomForestClassifier(n_estimators=50,random_state=42)
rf_no_conv.fit(X_train, y_train)
pred_no_conv = rf_no_conv.predict(X_test)
eval_classifier(pred_no_conv,y_test)
del rf_conv, rf_no_conv, convoluted_features






