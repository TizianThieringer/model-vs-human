import torch
import torch.distributed as dist
from torch import nn

#import eval_linear as lin

#Setup linear layer
def build_model():
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
    
    return model

class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)
    


def build_linear_classifier():
    linear_classifier = LinearClassifier(1536, 1000)
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier)
    linear_classifier.eval()
    state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_linearweights.pth")['state_dict']
    linear_classifier.load_state_dict(state_dict, strict=True)
    
    return linear_classifier




#Sequentialise
@torch.no_grad()
class MyModel(nn.Module):
    
    def __init__(self):
        nn.Module.__init__(self)
        self.facebook_model = build_model()
        self.lin_layer = build_linear_classifier()
            
    # forward
    def forward(self, x):
        intermediate_output = self.facebook_model.get_intermediate_layers(x)
        output = torch.cat([i_o[:, 0] for i_o in intermediate_output], dim=-1)
        
        output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
        output = output.reshape(output.shape[0], -1)

            
        output = self.lin_layer(output)
        return output



#Build whole model

#Prepare DDP
#dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)


model = MyModel()


