import torch
import torch.nn as nn

class SimpleTestNet(nn.Module):
    def __init__(self, cfg):
        super(SimpleTestNet, self).__init__()
        self.final_proj = nn.Sequential(
            nn.ConvTranspose1d(in_channels=1024, out_channels=256, kernel_size=8, stride=8, padding=0, output_padding=0),
            nn.GELU()
        )

    def forward(self, x):
        return self.final_proj(x)

# 假设 cfg 是一个配置对象，并且包含 encoder_embed_dim 属性
cfg = ...  # 初始化或获取您的配置对象

# 创建模型实例
model = SimpleTestNet(cfg)

# 创建一个随机输入张量
# 假设 batch_size = 1, seq_len = 10
batch_size, seq_len = 1, 100
input_tensor = torch.randn(batch_size, 1024, seq_len)

# 执行前向传播
output_tensor = model(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}")
