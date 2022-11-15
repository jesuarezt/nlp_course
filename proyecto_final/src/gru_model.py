class RNN_GRU_BI(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 
                                      embed_dim, 
                                      padding_idx=0) 
        self.rnn = nn.GRU(embed_dim, rnn_hidden_size, 
                           batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(rnn_hidden_size*2, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        out = self.embedding(text)
        out = nn.utils.rnn.pack_padded_sequence(out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True)
        _, (hidden) = self.rnn(out)
        out = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
 

def train_GRU_BI(dataloader):
    model2_1.train()
    total_acc, total_loss = 0, 0
    losses,acc_acum,nums=0,0,0

    for text_batch, label_batch, lengths in dataloader:
        optimizer.zero_grad()
        y_test=label_batch.long()
        y_pred = torch.argmax(model2_1(text_batch, lengths), 1)

        pred = model2_1(text_batch, lengths)
        loss = loss_func(pred, y_test)
        loss.backward()
        optimizer.step()
  
        total_acc += (y_pred == label_batch).float().sum().item()
        total_loss += loss.item()*label_batch.size(0)

    return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)

def evaluate_GRU_BI(dataloader):
    model2_1.eval()
    total_acc, total_loss = 0, 0
    with torch.no_grad():
        for text_batch, label_batch, lengths in dataloader:
            #revisar funcion de perdida
            y_pred = torch.argmax(model2_1(text_batch, lengths), 1)
            y_test=label_batch.long()
            pred = model2_1(text_batch, lengths)
            loss = loss_func(pred, y_test)
            total_acc += (y_pred == label_batch).float().sum().item()
            total_loss += loss.item()*label_batch.size(0)
    return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)