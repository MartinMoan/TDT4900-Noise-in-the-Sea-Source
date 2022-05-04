import torch
import numpy as np

softmax = torch.nn.Softmax(dim=1)

pred = torch.tensor([[10, 80, 0], [2, 70, 90]], dtype=torch.float)
predicted = softmax(pred)

target = torch.tensor([1, 0]).type(torch.LongTensor)
print(predicted, target)

loss = torch.nn.NLLLoss()

print(loss(predicted, target))

