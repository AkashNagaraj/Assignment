import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import tqdm


class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(80,30) # Input features, output features
        self.act = nn.ReLU()
        self.hidden2 = nn.Linear(30,10)
        self.output = nn.Linear(10,4)

    def forward(self,x):
        x = self.act(self.hidden1(x))
        x = self.act(self.hidden2(x))
        x = self.output(x)
        return x


def train(X,y):

    batch_size = 8
    X_train = X[:len(X)//batch_size*batch_size]
    y_train = y[:len(y)//batch_size*batch_size]

    n_epochs = 50  
    model = Multiclass()
    
    optimizer = optim.Adam(model.parameters(),lr=0.005)
    loss = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        for i in range(0,len(X)-batch_size,batch_size): 
            X_batch = X[i:i+batch_size]
            y_batch = torch.tensor(y[i:i+batch_size]).long()
            y_pred = model(X_batch)
            output = loss(y_pred, y_batch)
            optimizer.zero_grad()
            output.backward()
            optimizer.step()
        if epoch%10==0:
            print(output)
    
    # Use the last batch to test the model
    m = nn.Softmax(dim=1)
    valid = model(X[i+batch_size:])
    valid = m(valid)
    print("Accuracy is :",sum(y[i+batch_size:]==valid.argmax(1))/batch_size)



def implement_pytorch(X,y):
    X = torch.tensor(X,requires_grad=True)
    X = X.float()
    y = torch.from_numpy(y)
    train(X,y)


def main():
    X = np.random.rand(100,50)
    y = np.random.randint(0,2,(100,))
    implement_pytorch(X,y)


if __name__=="__main__":
    main()

