import torch
import time
import torch.nn as nn

device = torch.device('cuda')

model = nn.Sequential( # 参数0
    nn.Linear(1024, 8192),
    nn.ReLU(),
    nn.Linear(8192, 8192), 
    nn.ReLU(), 
    nn.Linear(8192, 1024),
)
model = model.to(device)
model = model.half() # 用半精度加速
optimizer = torch.optim.Adam(model.parameters()) 

torch.backends.cudnn.benchmark = True

batch_size = 32768 # 参数1
cnt = 0

print(model)
while True:
    inputs = torch.randn(batch_size, 1024, dtype=torch.float16, device=device)
    targets = torch.randn(batch_size, 1024, dtype=torch.float16, device=device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = torch.nn.functional.mse_loss(outputs, targets)
    loss.backward()
    optimizer.step()
    
    torch.cuda.synchronize() 
    time.sleep(0.01) # 参数2
    cnt += 1
    if cnt == 1 or cnt % 1000 == 0:
        print(f'Step {cnt:,}...')