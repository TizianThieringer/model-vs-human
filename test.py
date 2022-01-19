import torch
import torch.distributed as dist
from torch import nn


class LinearClassifier(torch.nn.Module):
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = torch.nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)
        intermediate_output = model.get_intermediate_layers(x, 1)
        x = torch.cat([a[:, 0] for a in intermediate_output], dim=-1)
        
        x = torch.cat((x.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
        x = x.reshape(x.shape[0], -1)

        # linear layer
        return self.linear(x)
    
def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)
# load backbone
model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')

#Setup linear layer
linear_classifier = LinearClassifier(1536, 1000)
linear_classifier = linear_classifier.cuda()
linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier)
linear_classifier.eval()
state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_linearweights.pth")['state_dict']
linear_classifier.load_state_dict(state_dict, strict=True)



#Sequentialise
model = torch.nn.Sequential(model,
                            linear_classifier)




x = torch.ones((1, 3, 224, 224))
out = model(x)
print("model out", out)