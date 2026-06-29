"""A single GSD-aware SAR physical-statistics auxiliary loss."""

import torch
import torch.nn.functional as F


def _odd_window(meters, gsd, minimum, maximum):
    value = int(round(float(meters) / max(float(gsd), 1e-6)))
    value = max(minimum, min(maximum, value))
    if value % 2 == 0:
        value = value + 1 if value < maximum else value - 1
    return value


def _statistics(image, window, eps):
    mean = F.avg_pool2d(image, window, stride=1, padding=window // 2)
    second = F.avg_pool2d(image.square(), window, stride=1, padding=window // 2)
    variance = (second - mean.square()).clamp_min(0.0)
    cv = torch.sqrt(variance + eps) / mean.clamp_min(eps)
    return mean, variance, cv


def sar_physical_loss(prediction, target, sar_gsd, cfg):
    """Match local mean/variance/CV over approximately equal ground extents."""
    enabled = tuple(cfg.get("physical_terms", ["mean", "variance", "cv"]))
    if not set(enabled).issubset({"mean", "variance", "cv"}):
        raise ValueError(f"Unknown physical_terms: {enabled}")
    eps = float(cfg.get("physical_eps", 1e-4))
    meters = float(cfg.get("physical_window_meters", 50.0))
    minimum = int(cfg.get("physical_window_min", 5))
    maximum = int(cfg.get("physical_window_max", 15))
    pred_01 = ((prediction + 1.0) * 0.5).clamp(0.0, 1.0)
    target_01 = ((target.detach() + 1.0) * 0.5).clamp(0.0, 1.0)
    totals = {name: prediction.new_zeros(()) for name in enabled}
    windows = []
    for index, gsd in enumerate(sar_gsd.detach().cpu().tolist()):
        window = _odd_window(meters, gsd, minimum, maximum)
        windows.append(window)
        pred_stats = _statistics(pred_01[index:index + 1], window, eps)
        target_stats = _statistics(target_01[index:index + 1], window, eps)
        for name, pred_value, target_value in zip(("mean", "variance", "cv"), pred_stats, target_stats):
            if name in totals:
                totals[name] = totals[name] + F.l1_loss(pred_value, target_value)
    for name in totals:
        totals[name] = totals[name] / len(prediction)
    total = sum(float(cfg.get(f"physical_{name}_weight", 1.0)) * value for name, value in totals.items())
    metrics = {f"physical_{name}": value for name, value in totals.items()}
    metrics["physical_window_px"] = prediction.new_tensor(sum(windows) / len(windows))
    return total, metrics


class AdaptivePhysicalWeight:
    """Bounded EMA gradient-ratio controller; never a free learnable scalar."""

    def __init__(self, cfg):
        self.enabled = bool(cfg.get("adaptive_physical", True))
        self.target_ratio = float(cfg.get("physical_target_grad_ratio", 0.05))
        self.minimum = float(cfg.get("physical_lambda_min", 0.0))
        self.maximum = float(cfg.get("physical_lambda_max", 0.1))
        self.ema = float(cfg.get("physical_lambda_ema", 0.95))
        self.value = float(cfg.get("lambda_physical", 0.05))

    @staticmethod
    def _norm(gradient):
        return gradient.detach().float().norm() if gradient is not None else torch.tensor(0.0)

    def update(self, diffusion_loss, physical_loss, probe_parameter):
        if not self.enabled:
            return self.value
        grad_diff = torch.autograd.grad(diffusion_loss, probe_parameter, retain_graph=True, allow_unused=True)[0]
        grad_phy = torch.autograd.grad(physical_loss, probe_parameter, retain_graph=True, allow_unused=True)[0]
        diff_norm = float(self._norm(grad_diff).cpu())
        phy_norm = float(self._norm(grad_phy).cpu())
        if diff_norm > 0.0 and phy_norm > 0.0:
            raw = self.target_ratio * diff_norm / (phy_norm + 1e-12)
            raw = max(self.minimum, min(self.maximum, raw))
            self.value = self.ema * self.value + (1.0 - self.ema) * raw
            self.value = max(self.minimum, min(self.maximum, self.value))
        return self.value

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = float(state.get("value", self.value))
