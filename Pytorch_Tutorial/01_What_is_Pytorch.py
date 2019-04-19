import torch
import numpy as np

x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.float32)
print(x)

x = torch.Tensor([5.5, 3])
print(x.dtype)

x = torch.rand_like(x, dtype=torch.float32)
print(x)

print(x.size())
y = torch.rand(2)
print(x + y)
print(torch.add(x, y))
torch.add(x, y, out=y)
print(y)
y.add_(x)
print(type(y))

# You can use standard NumPy-like indexing with all bells and whistles!
print(y[:])

x = torch.rand(4, 4)
y = x.view([16,])
print(y.size())
z = x.view([-1, 8])
print(z.size())

z = torch.rand(1)
print(z.item())

print(x.numpy())

y = torch.rand(4, 4)
y.add_(x)
print(y)

a = np.ones((4, 4))
b = torch.from_numpy(a)
print(b.size())

if torch.cuda.is_available():
    device = torch.device('cuda')
    y = torch.ones_like(b, device=device)
    x = b.to(device)
    z = x + y
    print(z)
    print(x.to('cpu'), torch.float32)