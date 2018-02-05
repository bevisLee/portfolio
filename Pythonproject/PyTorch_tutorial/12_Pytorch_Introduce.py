
## 참고 - http://bob3rdnewbie.tistory.com/326
## 원본 - http://pytorch.org/tutorials/beginner/nlp/pytorch_tutorial.html#introduction-to-torch-s-tensor-library
## Anaconda로 실행

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


torch.manual_seed(1)

V_data = [1., 2., 3.]
V = torch.Tensor(V_data)
print(V)

len(V)

x = torch.randn(2,3,4)
print(x)
print(x.view(2,12))

print(x.view(2,-1))



x = autograd.Variable(torch.Tensor([1., 2., 3]), requires_grad=True)

print(x.data)

y = autograd.Variable(torch.Tensor([4., 5., 6]), requires_grad=True)
z = x + y
print(z.data)

print(z.grad_fn)


s = z.sum()
print(s)
print(s.grad_fn)

s.backward()
print(x.grad)


