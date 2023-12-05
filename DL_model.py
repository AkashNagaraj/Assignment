import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import tqdm

import gensim.downloader as api
glove_model = api.load("glove-wiki-gigaword-100")

# Build a neural network with two hidden layers
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


# Train the neural network
def train(X,y,batch_size):

    #batch_size = 8
    X_train = X[:len(X)//batch_size*batch_size]
    y_train = y[:len(y)//batch_size*batch_size]

    n_epochs = 10  
    model = Multiclass()
    
    optimizer = optim.Adam(model.parameters(),lr=0.001)
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


# The function that is called from the jupyte-notebook
def implement_pytorch(X,y,batch_size):
    X = torch.tensor(X,requires_grad=True)
    X = X.float()
    y = torch.from_numpy(y)
    train(X,y,batch_size)

## The below is the incomplete code to implement RNN with GloVe 

class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        #self.embedding = nn.Embedding(input_dim,embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self,x):
        print("Inside forward pass",x.shape)
        embedded = self.embedding(x) # x = [sent_length, batch_size]
        output, hidden = self.rnn(embedded)
        out = self.fc(hidden)
        return out


def get_glove_values(df):
    input_data = []
    for idx in range(0,len(df)):
        sentence = df[idx].split()
        sentence_vector = []
        for word in sentence:
            vector = glove_model[word]
            sentence_vector.append(vector)
        input_data.append(sentence_vector)
    return np.array(input_data).astype(np.float32)


def implement_with_GloVe(df, INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, batch_size):
    
    col1, col2, target = "subject", "body", "opened"

    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for idx in range(0,len(df),batch_size):
        input_data, target = df[idx:idx+batch_size][col1] +" "+  df[idx:idx+batch_size][col2], df[idx:idx+batch_size][target]
        input_glove_data = get_glove_values(input_data)
        target_data = target.astype(np.float32)
        #train_glove(input_data, target_data)
        
        preds = model(input_glove_data)
        print("The RNN prediction is :",preds)

        y_pred = model(X_batch)
        output = loss(y_pred, y_batch)
        optimizer.zero_grad()
        output.backward()
        optimizer.step()


# This function passes dummy data to check if the model architectures are correct
def main():
    """
    X = np.random.rand(100,50)
    y = np.random.randint(0,2,(100,))
    implement_pytorch(X,y,8)
    """

    list1 = ["this is a good day"]*10
    list2 = ["today is"]*10
    list3 = [1,2,0,2,3,0,0,1,1,3]
    df = pd.DataFrame({"subject":list1,"body":list2,"opened":list3})
    INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, batch_size = 200, 300, 300, 4, 5

    implement_with_GloVe(df, INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, batch_size)

if __name__=="__main__":
    main()

