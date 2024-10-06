import torch

def beta_scheduler(T=50):
    beta_schedule = torch.linspace(0, 1.0, T+1).to('cuda')
    return beta_schedule

def forward_diffusion(x0, t, T=50):
    beta_schedule = beta_scheduler(T)
    alphas = 1. - beta_schedule
    alpha_bars = alphas.cumprod(dim=0)
    
    epsilon = torch.randn_like(x0)

    alpha_bar_t = torch.gather(alpha_bars, dim=0, index=t)
    alpha_bar_t = torch.reshape(alpha_bar_t,(-1,1,1,1))
    
    noisy_image = torch.sqrt(alpha_bar_t)*x0 + torch.sqrt(1 - alpha_bar_t)*epsilon
    return noisy_image, epsilon


def reverse_diffusion(n_sample, model, T=50):
    beta_schedule = beta_scheduler(T)
    x_t = torch.randn((n_sample, 1, 28, 28))

    alphas = 1. - beta_schedule
    alpha_bars = alphas.cumprod(dim=0)

    for t in range(T - 1):

        t = T - t - 1

        e_t = model(x_t, torch.full((n_sample), t))
        x_t_1 = (1/torch.sqrt(alphas[t]))*(x_t - ((1-alphas[t])/torch.sqrt(1-alpha_bars[t])*e_t))

        sigma_t = torch.sqrt(beta_schedule[t])
        z = torch.randn((n_sample, 1, 28, 28))

        if t>1:
          x_t = x_t_1 + sigma_t * z
        else:
          x_t = x_t_1

    x0 = x_t

    return x0
