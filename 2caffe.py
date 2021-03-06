import torch
import sys
sys.path.insert(0,'/opt/caffe/python') #caffe 路径
import caffe
from torch.autograd import Variable
import torchvision
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
from network import PoseNet
import os
from p2c import pytorch2caffe


net = PoseNet(3)
net.train(False)
net.eval()
print(net)
net.cuda()
net.load_state_dict(torch.load('./models/model_29.pkl'))
net.cpu()
input_var = Variable(torch.rand(1, 3, 22, 22))
state_dicts = net.state_dict()
pretrained_dict = {k: v for k, v in state_dicts.items() if k in state_dicts}

caffe_model = caffe.Net('./posenet.prototxt',caffe.TRAIN)

for param_name in caffe_model.params.keys():
    print param_name
    weight = param_name+'.weight'
    bias = param_name+'.bias'
    caffe_model.params[param_name][0].data[...] = pretrained_dict[weight].numpy()
    caffe_model.params[param_name][1].data[...] = pretrained_dict[bias].numpy()
caffe_model.save('posenet.caffemodel')
