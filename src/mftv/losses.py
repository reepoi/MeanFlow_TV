import torch
import torch.autograd.functional as func


def combined_loss(model, source_samples, target_samples, t, r):

    velocity = source_samples - target_samples
    x_interp = (1 - t) * target_samples + t * source_samples
    tau = torch.clamp(t - r, min=1e-8)

    def f(x, t_val, r_val):
        return model(x, t_val, r_val)

    tangents_partial_t = (
        torch.zeros_like(velocity),
        torch.ones_like(t),
        torch.zeros_like(r)
    )
    tangents_partial_x = (
        velocity,
        torch.zeros_like(t),
        torch.zeros_like(r)
    )

    # Compute u and derivatives
    u, dudt = func.jvp(f, (x_interp, t, r), tangents_partial_t, create_graph=True)
    _, nabla_x_u_dot_v = func.jvp(f, (x_interp, t, r), tangents_partial_x, create_graph=True)


    target_1 = velocity - tau * (nabla_x_u_dot_v + dudt)
    loss_1 = torch.nn.functional.mse_loss(u, target_1)

    target_2 = velocity - u  - tau * dudt
    loss_2 = torch.nn.functional.mse_loss(tau * nabla_x_u_dot_v, target_2)

    return loss_1 + 0.2 * loss_2

def backward_loss(model, source_samples, target_samples, t, r):

    x_interp = (1 - t) * target_samples + t * source_samples
    velocity = source_samples - target_samples

    def f(x, t_val, r_val):
        return model(x, t_val, r_val)

    tangents = (
        velocity,                  # dx = velocity (full dim)
        torch.ones_like(t),        # dt = 1
        torch.zeros_like(r)        # dr = 0
    )

    u, dudt = func.jvp(f, (x_interp, t, r), tangents , create_graph=True)  # ← REQUIR)


    u_target = velocity - (t - r) * dudt

    loss =  torch.nn.functional.mse_loss(u, u_target.detach())


    return loss

