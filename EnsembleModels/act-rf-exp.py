from sklearn.ensemble import RandomForestClassifier
import torch
import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch.nn.functional as F
import torch_geometric.transforms as T

from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score,confusion_matrix,roc_curve,accuracy_score,auc


def load_node_csv(df, index_col,y=None, encoders=None,**kwargs):
    mapping = {index: i for i, index in enumerate(df[index_col].unique())}
    x = df.drop([y,index_col],axis=1)
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


df_actor = pd.read_csv('trainData/df_actor.csv')
df_edges = pd.read_csv('trainData/df_edges.csv')
try:
    df_actor.drop('Unnamed: 0',axis=1,inplace=True)
#    df_edges.drop('index',axis=1,inplace=True)
except:
    pass

df_actor['class'] = df_actor['class'].replace(2,0)
# df_actor= df_actor.sort_values(by=['Time step'])
# index_range = df_actor[df_actor['Time step'].between(1, 34)].index.tolist()




for column in df_actor.columns[3:]:
    feature = np.array(df_actor[column]).reshape(-1,1)
    scaler = MinMaxScaler()
    scaler.fit(feature)
    feature_scaled = scaler.transform(feature)
    df_actor[column] = feature_scaled.reshape(1,-1)[0]




# # read data to graph
# df_actor_x, df_actor_y, df_actor_mapping = load_node_csv(df_actor,index_col='address',y='class')

# edge_index,_ = load_edge_csv(df_edges, 'input_address',  df_actor_mapping , 'output_address',  df_actor_mapping ,
#                   encoders=None)

# data = Data(x=df_actor_x,y=df_actor_y,edge_index=edge_index)
def create_graphs(df_actor, df_edges, time_steps):
    graphs = []
    for t in time_steps:
        df_t = df_actor[df_actor['Time step'] == t]
        target_nodes = df_t['address'].tolist()

        df_edges_t = df_edges[df_edges['input_address'].isin(target_nodes) & df_edges['output_address'].isin(target_nodes)]

        df_t = df_t.drop('Time step', axis=1)

        df_t_x, df_t_y, df_t_mapping = load_node_csv(df_t, index_col='address', y='class')

        edge_index, _ = load_edge_csv(df_edges_t, 'input_address', df_t_mapping, 'output_address', df_t_mapping,
                                      target_nodes=target_nodes, encoders=None)

        graph_data = Data(x=df_t_x, y=df_t_y, edge_index=edge_index)
        graphs.append(graph_data)

    return graphs

# 創建訓練圖和測試圖
train_time_steps = list(range(1, 35))
test_time_steps = list(range(35, 50))

train_graphs = create_graphs(df_actor, df_edges, train_time_steps)
test_graphs = create_graphs(df_actor, df_edges, test_time_steps)

# 使用 DataLoader 進行批次處理
train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)



class GAT_Net(torch.nn.Module):
    def __init__(self, features, hidden, classes, heads):
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
    
    def get_convoluted_features(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        convoluted_features = self.gat2(x, edge_index)

        return convoluted_features



def drawROC(test_y,y_pred):
    fpr, tpr, thersholds = roc_curve(test_y,y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)
    return roc_auc


# 指定裝置為 GPU 如果可用，否則使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 定義損失函數和優化器
weight = 0.5
label_weight = torch.tensor([0.1,weight],dtype=torch.float).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=label_weight)


# data = data.to(device)
# Define the RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=50,random_state=42)

# Train and test function for GCN + RF
def train_test_gcn_rf(model, criterion, train_loader, test_loader, rf_model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Lists to store convoluted features and labels
    all_convoluted_features = []
    all_labels = []

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

        # Get convoluted features and store them
        convoluted_features = model.get_convoluted_features(data.to(device))
        all_convoluted_features.append(convoluted_features.cpu().detach().numpy())
        all_labels.append(data.y.cpu().detach().numpy())

    avg_loss = running_loss / len(train_loader)
    accuracy = total_correct / total_samples

    # Train the Random Forest model with convoluted features
    all_convoluted_features = np.vstack(all_convoluted_features)
    all_labels = np.concatenate(all_labels)
    rf_model.fit(all_convoluted_features, all_labels)

    # Test the model
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in test_loader:
            out = model(data)
            pred = out.argmax(dim=1).tolist()
            y_pred.extend(pred)
            y_true.extend(data.y.tolist())

    # Calculate metrics for GCN + RF
    accuracy_gcn_rf = accuracy_score(y_true, y_pred)
    precision_gcn_rf = precision_score(y_true, y_pred)
    recall_gcn_rf = recall_score(y_true, y_pred)
    f1_gcn_rf = f1_score(y_true, y_pred)
    auc_gcn_rf = drawROC(y_true,y_pred)

    return avg_loss, accuracy, accuracy_gcn_rf, precision_gcn_rf, recall_gcn_rf, f1_gcn_rf, auc_gcn_rf

# Train and test function for RF alone
def train_test_rf(train_loader, test_loader, rf_model):
    # Lists to store features and labels
    all_features = []
    all_labels = []

    for data in train_loader:
        all_features.append(data.x.cpu().detach().numpy())
        all_labels.append(data.y.cpu().detach().numpy())

    # Flatten and concatenate features and labels
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)

    # Train the Random Forest model
    rf_model.fit(all_features, all_labels)

    # Test the model
    y_true = []
    y_pred = []

    for data in test_loader:
        pred = rf_model.predict(data.x.cpu().detach().numpy())
        y_pred.extend(pred)
        y_true.extend(data.y.cpu().detach().numpy())

    # Calculate metrics for RF alone
    accuracy_rf = accuracy_score(y_true, y_pred)
    precision_rf = precision_score(y_true, y_pred)
    recall_rf = recall_score(y_true, y_pred)
    f1_rf = f1_score(y_true, y_pred)
    auc_rf = drawROC(y_true,y_pred)
    return accuracy_rf, precision_rf, recall_rf, f1_rf, auc_rf

# Experiment setup
print('Weight: %.2f' % weight)
for num_epochs in [400]:
    for hidden in [120]:
        for heads in [8]:
            print_every = 50
            # Initialize GCN + RF model
            model_gcn_rf = GAT_Net(features=df_actor.shape[1] - 3, hidden=hidden, classes=2, heads=heads)
            # Initialize RF model
            rf_model_alone = RandomForestClassifier(n_estimators=100)

            for epoch in range(1, num_epochs + 1):
                train_loss, train_acc, acc_gcn_rf, prec_gcn_rf, rec_gcn_rf, f1_gcn_rf, auc_gcn_rf = train_test_gcn_rf(
                    model_gcn_rf, criterion=criterion, train_loader=train_loader, test_loader=test_loader,
                    rf_model=rf_model)
                
                if epoch % print_every == 0:
                    acc_rf, prec_rf, rec_rf, f1_rf, auc_rf = train_test_rf(train_loader, test_loader, rf_model_alone)
                    print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                          f'GCN+RF Acc: {acc_gcn_rf:.4f}, RF Acc: {acc_rf:.4f}')
                if epoch == num_epochs:
                    print('ACT model hidden layer:%d, heads:%d'%(hidden,heads))
                    print('GCN+RF \n acc:%.3f, precision:%.3f, recall:%.3f,  f1:%.3f, auc:%.3f'%(acc_gcn_rf, prec_gcn_rf, rec_gcn_rf, f1_gcn_rf, auc_gcn_rf))
                    print('only RF \n acc:%.3f, precision:%.3f, recall:%.3f,  f1:%.3f, auc:%.3f'%(acc_rf, prec_rf, rec_rf, f1_rf, auc_rf))

            del model_gcn_rf, acc_gcn_rf, prec_gcn_rf, rec_gcn_rf, f1_gcn_rf
            del rf_model_alone, acc_rf, prec_rf, rec_rf, f1_rf
