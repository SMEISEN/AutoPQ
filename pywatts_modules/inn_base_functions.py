import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch
from torch import nn


class CondNet(nn.Module):
    """
    The conditional sub-network of the conditional Invertible Neural Network (cINN).
    """
    def __init__(self, cond_features: int, horizon: int):
        """
        Initialize the neural network.
        :param cond_features:
            Number of conditional features.
        :type cond_features: int
        :param horizon:
            Forecasting horizon.
        :type horizon: int
        """
        super().__init__()
        self.horizon = horizon
        self.condition = nn.Sequential(nn.Linear(cond_features,  8),
                                       nn.Tanh(),
                                       nn.Linear(8, 4),
                                       )

    def forward(self, conds: torch.Tensor) -> torch.Tensor:
        """
        Feed forward through the network.
        :param conds:
            Conditional information tensor.
        :type conds: torch.Tensor
        :return:
            Output of the network.
        :rtype: torch.Tensor
        """
        return self.condition(conds)


def default_subnet(ch_in: int, ch_out: int) -> nn.Sequential:
    """
    Create default sub-network.
    :param ch_in:
        Number of input neurons.
    :type ch_in: int
    :param ch_out:
        Number of output neurons.
    :type ch_out: int
    :return:
        Initialized sub-network
    :rtype: nn.Sequential
    """
    return nn.Sequential(nn.Linear(ch_in, 32),
                         nn.Tanh(),
                         nn.Linear(32, ch_out))


class INN(nn.Module):
    """
    Initializes the conditional Invertible Neural Network (cINN).
    """
    def __init__(self, lr: float, cond_features: int, horizon: int, n_layers_cond: int = 5, n_layers_without_cond: int = 0,
                 subnet: nn.Sequential = default_subnet):
        """
        Initializes the pyWATTS module.
        :param lr:
            Learning rate.
        :type lr: float
        :param cond_features:
            Number of conditional features.
        :type cond_features: int
        :param horizon:
            Forecasting horizon.
        :type horizon: int
        :param n_layers_cond:
            Number of conditional layers.
        :type n_layers_cond: int
        :param n_layers_without_cond:
            Number of non-conditional layers.
        :type n_layers_without_cond: int
        :param subnet:
            Feed-forward sub-network.
        :type subnet: nn.Sequential
        """
        super().__init__()
        self.horizon = horizon
        if cond_features > 0:
            self.cond_net = CondNet(cond_features, horizon=self.horizon)
        else:
            self.cond_net = None

        self.no_layer_cond = n_layers_cond
        self.no_layer_without_cond = n_layers_without_cond
        self.subnet = subnet
        self.cinn = self.build_inn()

        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        for p in self.trainable_parameters:
            p.data = 0.01 * torch.randn_like(p)
        if self.cond_net:
            self.trainable_parameters += list(self.cond_net.parameters())

        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=lr, weight_decay=1e-5)

    def build_inn(self) -> Ff.ReversibleGraphNet:
        """
        Build the cINN.
        :return:
            Trainable cINN.
        :rtype: Ff.ReversibleGraphNet
        """
        nodes = [Ff.InputNode(self.horizon)]

        if self.cond_net:
            # Add conditional layers
            cond = Ff.ConditionNode(4)
            for k in range(self.no_layer_cond):
                nodes.append(
                    Ff.Node(nodes[-1], Fm.GLOWCouplingBlock, {"subnet_constructor": self.subnet, "clamp": 0.5},
                            conditions=cond))
                nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}))
            for k in range(self.no_layer_without_cond):
                nodes.append(
                    Ff.Node(nodes[-1], Fm.GLOWCouplingBlock, {"subnet_constructor": self.subnet}))
                nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}))
            return Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)
        else:
            for k in range(self.no_layer_cond):
                nodes.append(
                    Ff.Node(nodes[-1], Fm.GLOWCouplingBlock, {"subnet_constructor": self.subnet}))
                nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}))
            return Ff.ReversibleGraphNet(nodes + [Ff.OutputNode(nodes[-1])], verbose=False)

    def forward(self, x: torch.Tensor, conds: torch.Tensor, rev: bool = False) -> (torch.Tensor, torch.Tensor):
        """
        Forward operation through the INN.
        :param x:
            Target time series tensor.
        :type x: torch.Tensor
        :param conds:
            Conditional information tensor.
        :type conds: torch.Tensor
        :param rev:
            Whether to inverse transform or not.
        :type rev: bool
        :return:
            Latent tensor and Jacobian.
        :rtype: (torch.Tensor, torch.Tensor)
        """
        c = self._calculate_condition(conds)
        z, jac = self.cinn(x.float(), c=c, rev=rev)
        return z, jac

    def _calculate_condition(self, conds: torch.Tensor) -> torch.Tensor:
        """
        Calculate output from conditional sub-net.
        :param conds:
            Conditional information tensor.
        :type conds: torch.Tensor
        :return:
            Output of the conditional sub-network.
        :rtype: torch.Tensor
        """
        return self.cond_net(conds)
