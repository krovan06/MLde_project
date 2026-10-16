"""
ML-модель для интерполяции погоды.
Используется в research, trainer и inference.
"""

import torch
import torch.nn as nn


class WeatherMLP(nn.Module):
    """
    MLP для интерполяции погоды.
    
    Используется:
    - В ml_research.py — исследование и эксперименты.
    - В ml_trainer.py — обучение для продакшна.
    - В ml_inference.py — применение модели.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: list = None, #type: ignore
        output_dim: int = 1,
        activation=nn.ReLU,
    ):
        """
        Args:
            input_dim: Число входных фич.
                       По умолчанию 6: lat_norm, lon_norm,
                       month_sin, month_cos, weekday_sin, weekday_cos.
            hidden_dims: Список размеров скрытых слоёв.
                         По умолчанию [64, 128, 64].
            output_dim: Число выходных значений.
                        По умолчанию 1 (temperature).
            activation: Класс функции активации.
                        По умолчанию nn.ReLU.
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128, 64]

        # Собираем слои
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim

        # Выходной слой
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Прямой проход."""
        return self.net(x)