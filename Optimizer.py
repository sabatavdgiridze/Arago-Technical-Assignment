import torch
from abc import ABC, abstractmethod

class Distribution(ABC):
    @abstractmethod
    def sample(self, n, device):
        """Sample n elements from the distribution on given device(cpu or gpu)"""
        pass

    @abstractmethod
    def inverse_cdf(self, u):
        """Inverse cumulative distribution function"""
        pass

class UniformDistribution(Distribution):
    """Uniform distribution on [0,1]"""
    def sample(self, n, device):
        return torch.rand(n, device=device)
    def inverse_cdf(self, u):
        return u

class LinearDistribution(Distribution):
    """Distribution with density p(x) = 2x on [0,1]"""
    def sample(self, n, device):
        # using inverse transform sampling: U ~ Uniform(0,1) -> sqrt(U) ~ p(x) = 2x (as F(x) = x^2 for the linear distribution)
        return torch.sqrt(torch.rand(n, device=device))
    def inverse_cdf(self, u):
        return torch.sqrt(u)

class Optimizer:
    def __init__(self, bits, distribution, initial_values, g_func=None, n_samples=1_000_000, epochs=10_000, learning_rate=1e-2, use_fixed_sampling=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.levels_per_segment = 2 ** bits // 4

        self.distribution = distribution
        self.g_func = g_func if g_func is not None else lambda x: x

        self.n_samples, self.epochs, self.learning = n_samples, epochs, learning_rate
        self.use_fixed_sampling = use_fixed_sampling

        self.breakpoints = torch.tensor(initial_values, requires_grad=True, dtype=torch.float32, device=self.device)
        self.optimizer = torch.optim.Adam([self.breakpoints], lr=learning_rate)

    def fixed_sample(self, n, device):
        """
        Generate fixed samples (for every iteration) to reduce variance in Monte Carlo integration.
        """

        # Create uniform samples: [0, 1/n), [1/n, 2/n), ..., [(n-1)/n, 1)
        base_grid = torch.arange(n, device=device, dtype=torch.float32) / n
        uniform_offsets = torch.rand(n, device=device) / n
        stratified_uniform = base_grid + uniform_offsets

        # Transform through inverse CDF to get samples from target distribution
        return self.distribution.inverse_cdf(stratified_uniform)

    def make_levels(self, breakpoints):
        """builds values f takes from breakpoints = Tensor([a, b, c])"""
        points = torch.cat([
            torch.tensor([0.0], device=self.device, dtype=breakpoints.dtype),
            breakpoints,
            torch.tensor([1.0], device=self.device, dtype=breakpoints.dtype),
        ])
        segments = []
        t = torch.arange(self.levels_per_segment + 1, device=self.device, dtype=breakpoints.dtype) / self.levels_per_segment
        for idx in range(4):
            start, end = points[idx], points[idx + 1]
            vals = start + (end - start) * t
            if idx < 3:
                vals = vals[:-1]  # exclude right endpoint
            segments.append(vals)
        return torch.cat(segments)

    def quantize(self, x, levels):
        """
        Returns the value of the interval, for the step-function f, in which x belongs.
        To find the index of that interval, we find the greatest i such that:
            g^{-1}((g(levels[i]) + g(levels[i+1])) / 2) <= x or
            (g(levels[i]) + g(levels[i+1])) / 2 <= g(x) as g is increasing
        """

        g_levels = self.g_func(levels)
        mid_points = (g_levels[:-1] + g_levels[1:]) / 2

        # Find the correct index
        idx = torch.bucketize(self.g_func(x), mid_points)

        # Return the step function value
        return levels[idx]

    def compute_loss(self, samples, quantized):
        """we use Monte Carlo integration in here. for complicated pdf"""
        g_quantized = self.g_func(quantized)  # (g o f)(x)
        g_samples = self.g_func(samples)      # g(x)
        return torch.mean((g_quantized - g_samples) ** 2)

    def optimize(self, verbose=True):
        for epoch in range(self.epochs + 1):
            # Forward pass
            levels = self.make_levels(self.breakpoints)

            if self.use_fixed_sampling:
                samples = self.fixed_sample(self.n_samples, self.device)
            else:
                samples = self.distribution.sample(self.n_samples, self.device)

            quantized = self.quantize(samples, levels)

            # Compute the loss
            loss = self.compute_loss(samples, quantized)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging
            if verbose and epoch % (self.epochs // 10) == 0:
                with torch.no_grad():
                    a, b, c = self.breakpoints
                    print(f"Epoch {epoch:4d}  loss={loss:.12f}  a={a:.10f}  b={b:.10f}  c={c:.10f}")

        # Return final values for [a, b, c]
        with torch.no_grad():
            return self.breakpoints.tolist()

if __name__ == "__main__":
    print("=== UNIFORM DISTRIBUTION ===")
    uniform_dist = UniformDistribution()

    initial_conditions = [
        [0.3, 0.7, 0.9],
        [0.2, 0.4, 0.8],
        [0.1, 0.6, 0.9],
    ]

    for i, initial in enumerate(initial_conditions):
        print(f"\nUniform - Initial condition {i + 1}: {initial}")
        optimizer = Optimizer(bits=8, distribution=uniform_dist, initial_values=initial)
        result = optimizer.optimize()
        print(f"Final breakpoints: {result}")

    print("\n\n=== LINEAR DISTRIBUTION ===")
    linear_dist = LinearDistribution()

    for i, initial in enumerate(initial_conditions):
        print(f"\nLinear - Initial condition {i + 1}: {initial}")
        optimizer = Optimizer(bits=8, distribution=linear_dist, initial_values=initial)
        result = optimizer.optimize()
        print(f"Final breakpoints: {result}")
