import torch
import itertools

class PolynomialFeaturesTorch:
    def __init__(self, degree=2, interaction_only=False, include_bias=False):
        """
        PyTorch simplified implementation of PolynomialFeatures.
        Args:
            degree (int): The degree of the polynomial features.
            interaction_only (bool): If True, only interaction features are produced.
            include_bias (bool): If True, include a bias column (all ones).
        """
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias

    def fit_transform(self, X):
        """
        Generate polynomial features.
        Args:
            X (torch.Tensor): Input tensor of shape (n_samples, n_features).
        Returns:
            torch.Tensor: Transformed tensor with polynomial features.
        """
        n_samples, n_features = X.shape
        # Generate all combinations of feature indices
        combinations = self._combinations(n_features)
        # Compute polynomial features
        features = [torch.ones((n_samples, 1), dtype=X.dtype, device=X.device)] if self.include_bias else []
        for comb in combinations:
            features.append(torch.prod(X[:, comb], dim=1, keepdim=True))
        features = torch.cat(features, dim=1)
        features = torch.sum(features, dim = -1)
        # print("feature shape:",features.shape)
        return features

    def _combinations(self, n_features):
        """
        Generate index combinations for polynomial features.
        Args:
            n_features (int): Number of input features.
        Returns:
            list of tuples: Index combinations for polynomial features.
        """
        comb_func = itertools.combinations if self.interaction_only else itertools.combinations_with_replacement
        return [comb for degree in range(1, self.degree + 1) for comb in comb_func(range(n_features), degree)]
    

if __name__ == "__main__":
    metrics=torch.tensor([[1,2]])
    poly = PolynomialFeaturesTorch(degree=2,include_bias=False)
    # print(metrics.shape)
    normalized_metrics = poly.fit_transform(metrics)
    print(normalized_metrics)