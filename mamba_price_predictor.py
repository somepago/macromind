import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Dimension of representations')
parser.add_argument('--layer', type=int, default=2,
                    help='Num of layers')
parser.add_argument('--n-test', type=int, default=300,
                    help='Size of test set')
parser.add_argument('--datapath', type=str, default='data_prep/commodity_data_60days.csv',
                    help='Path to the data')                    

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

def evaluation_metric(y_test,y_hat):
    MSE = mean_squared_error(y_test, y_hat)
    RMSE = MSE**0.5
    MAE = mean_absolute_error(y_test,y_hat)
    R2 = r2_score(y_test,y_hat)
    print('%.4f %.4f %.4f %.4f' % (MSE,RMSE,MAE,R2))

def set_seed(seed,cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def dateinf(series, n_test):
    lt = len(series)
    print('Training start',series[0])
    print('Training end',series[lt-n_test-1])
    print('Testing start',series[lt-n_test])
    print('Testing end',series[lt-1])

set_seed(args.seed,args.cuda)

class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=args.hidden,
            num_layers=args.layer,
            batch_first=True
        )
        self.fc = nn.Linear(args.hidden, out_dim)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  
        
        lstm_out, _ = self.lstm(x)
        
        out = self.fc(lstm_out[:, -1, :])
        out = self.tanh(out)
        
        return out.squeeze(0)

def PredictWithData(trainX, trainy, testX):
    # Create a simpler model for this task
    input_dim = trainX.shape[1]
    
    # Define a simple model
    model = nn.Sequential(
        nn.Linear(input_dim, args.hidden),
        nn.ReLU(),
        nn.Linear(args.hidden, 1)
    )
    
    # Convert data to PyTorch tensors
    X_train = torch.FloatTensor(trainX)
    y_train = torch.FloatTensor(trainy).view(-1, 1)  # Reshape to column vector
    X_test = torch.FloatTensor(testX)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    # Training loop
    for epoch in range(args.epochs):
        # Forward pass
        model.train()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if epoch % 10 == 0 and epoch > 0:
            print(f'Epoch {epoch} | Loss: {loss.item():.4f}')
    
    # Prediction
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy().flatten()
    
    # Ensure predictions has the right size
    if len(predictions) != args.n_test:
        print(f"Warning: predictions size ({len(predictions)}) doesn't match n_test ({args.n_test})")
        if len(predictions) < args.n_test:
            # Pad with the last value if needed
            predictions = np.pad(predictions, (0, args.n_test - len(predictions)), 'edge')
        else:
            # Truncate if too long
            predictions = predictions[:args.n_test]
    
    return predictions

data = pd.read_csv(args.datapath)
data['Date'] = pd.to_datetime(data['Date'])  

label_encoder_ticker = LabelEncoder()
label_encoder_commodity = LabelEncoder()

data['Ticker'] = label_encoder_ticker.fit_transform(data['Ticker'])
data['Commodity'] = label_encoder_commodity.fit_transform(data['Commodity'])

data['Date'] = data['Date'].astype('int64') // 10**9
# Z-normalize all columns except 'Date', 'Ticker', and 'Commodity'
numeric_columns = data.columns.drop(['Date'])
data[numeric_columns] = (data[numeric_columns] - data[numeric_columns].mean()) / data[numeric_columns].std()
close = data.pop('Close').values
data.drop(columns=['Average'],inplace=True)
data = data.sort_values(by='Date', ascending=True).reset_index(drop=True)

dat = data.iloc[:,1:].values
trainX, testX = dat[:-args.n_test, :], dat[-args.n_test:, :]
trainy = close[:-args.n_test]

predictions = PredictWithData(trainX, trainy, testX)
time = data['Date'][-args.n_test:].values
groundtruth = close[-args.n_test:]


# Print shapes for debugging
print(f"predictions shape: {predictions.shape}")
print(f"args.n_test: {args.n_test}")
print(f"groundtruth shape: {groundtruth.shape}")


print('MSE RMSE MAE R2')
evaluation_metric(groundtruth, predictions)
plt.figure(figsize=(10, 6))
plt.plot(time, groundtruth, label='Stock Price')
plt.plot(time, predictions, label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Close', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()