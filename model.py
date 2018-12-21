import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class EncoderCNN(nn.Module):
    def __init__(self, embed_size, model_name):
        super(EncoderCNN, self).__init__()
          
        # TO DO: inception_v3, vgg19_bn, alexnet, 
        if model_name == 'densenet':
              model = models.densenet161(pretrained=True)
        elif model_name == 'resnet':
              model = models.resnet152(pretrained=True)
        else:
              print("{} is not one of accepted model names: 'densenet', 'resnet'".format(model_name))
        self.model_name = model_name
          
        #freeze by name
#         ct = []
#         for name, child in model.named_children():
#             if "Conv2d_4a_3x3" in ct:
#                 for params in child.parameters():
#                     params.requires_grad = False
#             ct.append(name)

        #freeze by number
#         for i, child in enumerate(model.children()):
#             if i<=4:
#                 for param in child.parameters():
#                     param.requires_grad = False
                    
        #freeze all
        for param in model.parameters():
            param.requires_grad_(False)
        
        #drop last layer to match output size 
        modules = list(model.children())[:-1]
        self.model = nn.Sequential(*modules)
        if model_name == 'densenet':
            self.embed = nn.Linear(model.classifier.in_features, embed_size) #classifier for densenet but fc for others
        elif model_name == 'resnet':
            self.embed = nn.Linear(model.fc.in_features, embed_size)
        else:
            print("{} is not one of accepted model names: 'densenet', 'resnet'".format(model_name))
        

    def forward(self, images):
        features = self.model(images)
        if self.model_name == 'densenet':
              features = F.avg_pool2d(features, kernel_size=7) #to get to appropriate size
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, bias=True, dropout=.25, batch_first=True) 
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def init_hidden(self, batch_size):
        ''' At the start of training, we need to initialize a hidden state;
           there will be none because the hidden state is formed based on perviously seen data.
           So, this function defines a hidden state with all zeroes and of a specified size.'''
        # The axes dimensions are (n_layers, batch_size, hidden_size)
        return (torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device), \
                torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device))

    def forward(self, features, captions):
        #LSTM accepts all input vectors of feature size and then initialized hidden state
        captions = captions[:,:-1] #remove <END> b/c we dont actually run end marker through lstm
        
        batch_size = features.shape[0] # features is of shape (batch_size, embed_size)
        
        self.hidden = self.init_hidden(batch_size) 
        c_embeddings = self.embed(captions) #(batch_size, captions length - 1, embed_size)
        
        inputs = torch.cat((features.unsqueeze(1), c_embeddings), dim=1) #add dimension to first axis of features so 
                                                                         ##(batch_size, 1, embed_size)
        #inputs size: (batch_size, # words in caption, embed_size)
        lstm_out, self.hidden = self.lstm(inputs, self.hidden) #hidden already initialized
        final_outputs = self.fc(lstm_out)
        return final_outputs

    def sample(self, inputs, states=None, max_len=20):
        ''' 
        accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len)
        (greedy search of taking next word in the sequence as the one with the maximum probability in vocabulary)
        '''
        
        output = []
        self.batch_size = inputs.shape[0]
        self.hidden = self.init_hidden(self.batch_size)
        
        while True:
            lstm_out, self.hidden = self.lstm(inputs, self.hidden)
            final_outputs = self.fc(lstm_out) # (1, 1, vocab_size)
            final_outputs = final_outputs.squeeze(1) # (1, vocab_size)
            #add most likely word
            _, new_word_torch = torch.max(final_outputs, dim=1)
            new_word = int(new_word_torch.cpu().numpy()[0]) #convert to numpt int
            output.append(new_word)
            # <end> index is 1
            if new_word == 1 or len(output)>max_len:
                break
                
            inputs = self.embed(new_word_torch)
            inputs = inputs.unsqueeze(1) # (1, 1, embed_size)
                
        return output
                
        