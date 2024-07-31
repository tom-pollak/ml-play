# %%
# fmt: off
import torch as t
import torch.nn.functional as F
import torch.nn as nn
from tqdm import trange
import matplotlib.pyplot as plt

# %%

bs = 512
nsteps = 10_000
lookback = 4
pos = 5e-1
device = 'cpu' # runs faster on cpu
start_pos = -5*t.pi
end_pos = 5*t.pi
range_size = end_pos - start_pos

def create_model(device=device):
    return nn.Sequential(
        nn.Linear(lookback, 12),
        nn.ReLU(),
        nn.Linear(12, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Tanh(),
    ).to(device)


# %%

def to_np(x): return x.detach().cpu().numpy()

X = t.arange(start_pos, end_pos, step=pos)
y = t.cos(X)
plt.plot(to_np(X), to_np(y))
plt.show()

# %%

sample_start = t.rand(1) * (end_pos - lookback * pos - start_pos) + start_pos
print("start of sample", sample_start)
x = t.arange(lookback) * pos + sample_start # (lookback,)
y = t.cos(x[-1])
print("batch", x, y)

# %%

def get_sample(bs, device='cpu'):
    sample_start = t.rand(bs) * (range_size - lookback * pos) + start_pos
    x = sample_start.unsqueeze(-1) + (t.arange(lookback) * pos).unsqueeze(0) # (bs, lookback)
    y = t.cos(x[:, -1]).unsqueeze(-1) # (bs, lookback), (bs, 1)
    x_norm = 2*x / range_size
    return x_norm.to(device), y.to(device)

x,y = get_sample(10)
print("input", x)
print("target", y)

net = create_model('cpu')
y_pred = net(x)
print("prediction", y_pred) # (bs,1)

loss = F.mse_loss(y, y_pred)
print("loss", loss)

# %%

def add_noise_(x, stdev=1e-1):
    x += t.randn_like(x) * stdev
    return x

x,y = get_sample(1000)
add_noise_(x, stdev=1e-1)
plt.scatter(to_np(x[:, -1]),to_np(y))


# %%

def train(sampler, noise=None):
    net = create_model(device)
    net.compile()
    net.train()

    pb = trange(nsteps)
    optimizer = t.optim.Adam(list(net.parameters()))

    for step in pb:
        x,y = sampler(bs, device=device)
        if noise is not None:
            add_noise_(x, stdev=noise)
        y_pred = net(x)
        loss = F.mse_loss(y, y_pred)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 500:
            pb.set_postfix(dict(loss=loss.item()))

    return net.eval()

net = train(get_sample, noise=1e-1)

with t.inference_mode():
    x,y = get_sample(1000, device=device)
    y_pred = net(x)
    plt.scatter(to_np(x[:, -1]),to_np(y_pred))

# %%

sample_start = t.arange(start_pos, end_pos * 2, step=pos)
x = sample_start.unsqueeze(-1) + (t.arange(lookback) * pos).unsqueeze(0) # (bs, lookback)
y = t.cos(x[:, -1]).unsqueeze(-1) # (bs, lookback), (bs, 1)
x_norm = 2*x / range_size
y_pred = net(x_norm)

plt.plot(to_np(x_norm[:, -1]), to_np(y_pred), label="Pred")
plt.plot(to_np(x_norm[:, -1]), to_np(y), linestyle="--", alpha=0.5, label="True (cos)")
plt.vlines(end_pos / (0.5*range_size), ymin=-1, ymax=1, linestyles='--', label="end of train set")
plt.legend(loc='upper right')
plt.show()

# %%

def get_sample2(bs, device='cpu'):
    sample_start = t.rand(bs) * (range_size - (lookback+1) * pos) + start_pos
    samples = sample_start.unsqueeze(-1) + (t.arange(lookback+1) * pos).unsqueeze(0) # (bs, lookback+1)
    samples_cos = t.cos(samples)
    x, y = samples_cos[:, :-1], samples_cos[:, -1].unsqueeze(-1) # (bs, lookback), (bs, 1)
    return x.to(device), y.to(device)


# %%

net = train(get_sample2, noise=1e-1)

@t.inference_mode()
def generate(nsteps):
    seq = start_pos + (t.arange(lookback) * pos).unsqueeze(0) # (lookback,)
    seq = t.cos(seq)
    for step in range(nsteps):
        x = seq[:,-lookback:].clone()
        add_noise_(x, 1e-1)
        pred = net(x)
        seq = t.cat((seq, pred), dim=1)
    return seq


steps = int(2*(range_size / pos))
seq = generate(steps)[0]
plt.plot(to_np(t.linspace(start_pos, end_pos*2, steps=steps+lookback)), to_np(seq))
plt.show()

# %%
