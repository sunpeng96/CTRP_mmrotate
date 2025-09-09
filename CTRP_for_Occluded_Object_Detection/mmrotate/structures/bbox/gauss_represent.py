import torch
import math

torch.pi = math.pi


def rbox2gauss(rboxes):
    cx, cy, w, h, angle = rboxes.unbind(dim=-1)
    gGaussA, gGaussB, gGaussC = gauss_encode(w, h, angle)
    return torch.stack([cx, cy, gGaussA, gGaussB, gGaussC], dim=-1).view(-1, 5)


def gauss2rbox(gauss):
    cx, cy, A, B, C = gauss.unbind(dim=-1)
    w, h, theta = gauss_decode(A, B, C)
    return torch.stack([cx, cy, w, h, theta], dim=-1).view(-1, 5)


def gauss_encode(w, h, angle):
    A = 0.25 * (w.pow(2.0) * torch.cos(angle).pow(2.0) + h.pow(2.0) * torch.sin(angle).pow(2.0))
    B = 0.25 * (h.pow(2.0) * torch.cos(angle).pow(2.0) + w.pow(2.0) * torch.sin(angle).pow(2.0))
    C = 0.25 * (w.pow(2.0) - h.pow(2.0)) * torch.cos(angle) * torch.sin(angle)
    return A, B, C


def gauss_decode(a, b, c):
    e = 0.5 * (b - a) / c
    theta = torch.zeros_like(c)
    mid = torch.sqrt(1.0 + e.pow(2.0))
    theta = torch.where(c != 0, torch.atan(e + torch.sign(c) * mid), theta)
    theta[(c == 0) & (a < b)] = -torch.pi / 2.0
    mid_term = 2.0 * torch.abs(c) * mid
    le = torch.where(c != 0, torch.sqrt(2.0 * (a + b + mid_term)), torch.sqrt(4.0 * a))
    ls = torch.where(c != 0, torch.sqrt(2.0 * (a + b - mid_term)), torch.sqrt(4.0 * b))
    le = torch.where(le > ls, le, ls)
    ls = torch.where(le > ls, ls, le)
    return le, ls, theta