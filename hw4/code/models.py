import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, input_dim: int, num_frequencies: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies

    @property
    def output_dim(self) -> int:
        return self.input_dim * (2 * self.num_frequencies + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encodings = [x]
        for level in range(self.num_frequencies):
            freq = (2**level) * torch.pi
            encodings.append(torch.sin(freq * x))
            encodings.append(torch.cos(freq * x))
        return torch.cat(encodings, dim=-1)


class NeuralField2D(nn.Module):
    def __init__(
        self,
        coord_dim: int = 2,
        pe_frequencies: int = 10,
        hidden_dim: int = 256,
        hidden_layers: int = 4,
    ):
        super().__init__()
        self.pe = PositionalEncoding(input_dim=coord_dim, num_frequencies=pe_frequencies)

        layers: list[nn.Module] = []
        in_dim = self.pe.output_dim
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 3))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pe(coords))


class NeRFMLP(nn.Module):
    def __init__(
        self,
        xyz_pe_frequencies: int = 10,
        dir_pe_frequencies: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 8,
        skip_layer: int = 4,
    ):
        super().__init__()
        self.xyz_pe = PositionalEncoding(input_dim=3, num_frequencies=xyz_pe_frequencies)
        self.dir_pe = PositionalEncoding(input_dim=3, num_frequencies=dir_pe_frequencies)
        self.skip_layer = skip_layer

        xyz_dim = self.xyz_pe.output_dim
        dir_dim = self.dir_pe.output_dim

        self.xyz_layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                in_dim = xyz_dim
            elif layer_idx == skip_layer:
                in_dim = hidden_dim + xyz_dim
            else:
                in_dim = hidden_dim
            self.xyz_layers.append(nn.Linear(in_dim, hidden_dim))

        self.feature_layer = nn.Linear(hidden_dim, hidden_dim)
        self.sigma_layer = nn.Linear(hidden_dim, 1)
        self.color_hidden = nn.Linear(hidden_dim + dir_dim, hidden_dim // 2)
        self.color_layer = nn.Linear(hidden_dim // 2, 3)
        self.relu = nn.ReLU()
        self.sigma_activation = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

        # A small positive bias helps the density branch avoid collapsing to zero
        # at initialization, especially on longer runs.
        nn.init.constant_(self.sigma_layer.bias, 0.1)

    def forward(self, xyzs: torch.Tensor, r_ds: torch.Tensor):
        num_rays, num_samples, _ = xyzs.shape

        xyz_flat = xyzs.reshape(-1, 3)
        dirs = r_ds[:, None, :].expand(-1, num_samples, -1).reshape(-1, 3)

        xyz_encoded = self.xyz_pe(xyz_flat)
        dir_encoded = self.dir_pe(dirs)

        h = xyz_encoded
        for layer_idx, layer in enumerate(self.xyz_layers):
            if layer_idx == self.skip_layer:
                h = torch.cat((h, xyz_encoded), dim=-1)
            h = self.relu(layer(h))

        sigma = self.sigma_activation(self.sigma_layer(h))
        features = self.feature_layer(h)
        color_hidden = self.relu(self.color_hidden(torch.cat((features, dir_encoded), dim=-1)))
        rgb = self.sigmoid(self.color_layer(color_hidden))

        return rgb.reshape(num_rays, num_samples, 3), sigma.reshape(num_rays, num_samples, 1)
