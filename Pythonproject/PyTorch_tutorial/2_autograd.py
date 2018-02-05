
## torch tutorial : 2_Autograd
## 참고 - http://bob3rdnewbie.tistory.com/315
## anaconda 기본환경 설정 후 실행

import torch
from torch.autograd import Variable

x = Variable(torch.ones(2,2),requires_grad=True)
print(x)

y = x+2
print(y)

print(y.creator) # error

z = y*y*3
out = z.mean()

print(z,out)

out.backward()

print(x.grad)


x = torch.randdn(3)
x = Variable(x,requires_grad=True)

y = x*2
whiel y.data.norm() < 1000 :
    y = y*2


print(y)


gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)

print(x.grad)

