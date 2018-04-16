import torch
import torch.nn as nn
from torch.autograd import Variable
from my_package.modules.add import MyAddModule
from my_package._ext import my_lib

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.add = MyAddModule()

    def forward(self, input1, input2):
        return self.add(input1, input2)


def test_net():
    model = MyNetwork()
    x = torch.range(1, 25).view(5, 5)
    input1, input2 = Variable(x), Variable(x * 4)
    print(model(input1, input2))
    print(input1 + input2)

    if torch.cuda.is_available():
        input1, input2, = input1.cuda(), input2.cuda()
        print(model(input1, input2))
        print(input1 + input2)


def test_aten():
    if torch.cuda.is_available():
        t = torch.cuda.FloatTensor([[1.0, 2.0], [3.0, 4.0]])
        my_lib.my_lib_aten(t)


# test_net()
test_aten()
