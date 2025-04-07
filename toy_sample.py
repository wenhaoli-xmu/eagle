import torch
x= torch.tensor([1., 2., 3.,4.,5.,6.], requires_grad=True)
inp = x[3:]
f = torch.nn.Linear(3, 1)
y = f(inp)
y.backward()
 
print(x.grad)
print(inp.grad)

