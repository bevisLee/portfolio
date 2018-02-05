
## 참고 사이트 - http://bob3rdnewbie.tistory.com/318

import torch
a = torch.FloatTensor(5, 7)

print(a)
print(a.size())

x = torch.ones(5, 5)
print(x)

z = torch.Tensor(5, 2)
z[:,0] = 10
z[:,1] = 100
print(z)