import torch


@torch.no_grad()
def model_lerp_(model, model_other, coeff):
    for p1, p2 in zip(model.parameters(), model_other.parameters()):
        p1.data.lerp_(p2.data, 1-coeff)


@torch.no_grad()
def model_add_(model, model_other):
    for p1, p2 in zip(model.parameters(), model_other.parameters()):
        p1.data.add_(p2.data)


@torch.no_grad()
def model_sub_(model, model_other):
    for p1, p2 in zip(model.parameters(), model_other.parameters()):
        p1.data.sub_(p2.data)


def slerp(tensor1, tensor2, lam):
    cos_omega = torch.sum(tensor1 * tensor2) / (torch.norm(tensor1) * torch.norm(tensor2))
    omega = torch.acos(torch.clamp(cos_omega, -1.0, 1.0))
    coeff1 = (torch.sin((1 - lam) * omega) / torch.sin(omega))
    coeff2 = (torch.sin(lam * omega) / torch.sin(omega))
    return coeff1 * tensor1 + coeff2 * tensor2


@torch.no_grad()
def model_slerp_(model, model_other, coeff=0.5):
    for p1, p2 in zip(model.parameters(), model_other.parameters()):
        p_new = slerp(p1.data, p2.data, coeff)
        p1.data.copy_(p_new)
