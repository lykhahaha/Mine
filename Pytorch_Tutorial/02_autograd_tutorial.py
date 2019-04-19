import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)

y =  x + 2
print(y)

print(y.grad_fn)
print(y.requires_grad)

z = y * y * 3
out = z.mean()
# x.requires_grad_(False)
print(z, out)

out.backward(torch.tensor(2.))
print(x.grad)

x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1600:
    y = y * 2
y.backward(torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float32))
print(x.grad)

print(x.requires_grad)
print((x*2).requires_grad)
with torch.no_grad():
    print((x*2).requires_grad)
    print((x*2).grad_fn)