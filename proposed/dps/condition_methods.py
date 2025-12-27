from abc import ABC, abstractmethod
import torch
import logging

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):  # 通过**kwargs传入额外参数
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            distance = torch.norm(difference)  # scalar
            # 原文是2范数的平方，而这里是2范数，似乎是为了和原文中对2范数做归一化一致，因为后面没有除以自身模。
            norm_grad = torch.autograd.grad(outputs=distance, inputs=x_prev)[0]  # (B, C, H, W)
            
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            distance = torch.norm(difference) / measurement.abs()
            distance = distance.mean()
            norm_grad = torch.autograd.grad(outputs=distance, inputs=x_prev)[0]

        else:
            raise NotImplementedError

        return norm_grad, distance

    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t
    
@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, distance = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        
        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, distance
        
@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, distance = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        return x_t, distance
    
@register_conditioning_method(name='sg')
class SphericalGaussianConstraint(ConditioningMethod):
    # Guidance with spherical gaussian constraint for conditional diffusion》的实现，但没使用
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, distance = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        norm_norm_grad = norm_grad.flatten(1).norm(dim=1, keepdim=True)  # (B, C, H, W) -> (B, C*H*W) -> (B,1)
        norm_norm_grad = norm_norm_grad.unsqueeze(-1).unsqueeze(-1)      # (B,1) -> (B,1,1,1)
        # norm_norm_grad = torch.linalg.norm(norm_grad, dim=(1,2,3), keepdim=True)
        sigma_4_dsg = kwargs.get('sigma_4_dsg')  # torch.Size([])
        n = x_t.shape[1] * x_t.shape[2] * x_t.shape[3]
        radius = (n**0.5 * sigma_4_dsg).to(x_t.device)  # torch.Size([])
        print(f"radius: {radius.shape}")
        x_t -= radius * norm_grad / (norm_norm_grad + 1e-9)

        return x_t, distance
        
@register_conditioning_method(name='ps+')
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get('num_sampling', 5)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        distance = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            distance += torch.norm(difference) / self.num_sampling
        
        norm_grad = torch.autograd.grad(outputs=distance, inputs=x_prev)[0]  # 求对x_prev求梯度就是靠这个实现的？
        x_t -= norm_grad * self.scale
        return x_t, distance
