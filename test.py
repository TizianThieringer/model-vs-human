import torch
import torch.distributed as dist

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

        # linear layer
        return self.linear(x)


dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)
# load backbone
model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')

#Setup linear layer
linear_classifier = LinearClassifier(2048, 1000)
linear_classifier = linear_classifier.cuda()
linear_classifier = torch.nn.parallel.DistributedDataParallel(linear_classifier)
state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_linearweights.pth")['state_dict']
linear_classifier.load_state_dict(state_dict, strict=True)

#Sequentialise
model = torch.nn.Sequential(model,
                            linear_classifier)

#inp = torch.ones((1, 3, 224, 224))
#intermediate_output = model.get_intermediate_layers(inp, 1)
#output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
#output = linear_classifier(output)
#print(output)


x = torch.ones((1, 3, 224, 224))
out = model(x)
print("out: ", out)