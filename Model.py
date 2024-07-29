from utils import *


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = SelfAttention(d_model)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(d_model) for _ in range(2)]
        )

    def forward(self, x):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention(x, x, x, "Encoder")
        )
        x = self.residual_connections[1](x, self.ffn)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(Encoder, self).__init__()
        self.encoderlayer = EncoderLayer(d_model, d_ff, dropout)

    def forward(self, x):
        x = self.encoderlayer(x)
        return x


# Decoder block
class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(DecoderBlock, self).__init__()
        self.self_attention = SelfAttention(d_model)
        self.cross_attention = SelfAttention(d_model)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(d_model) for _ in range(3)]
        )

    def forward(self, x, encoder_output):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention(x, x, x, "Decoder")
        )
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention(
                x, encoder_output, encoder_output, "Encoder"
            ),
        )
        x = self.residual_connections[2](x, self.ffn)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(Decoder, self).__init__()
        self.decoderlayer = DecoderBlock(d_model, d_ff, dropout)

    def forward(self, x, encoder_output):
        x = self.decoderlayer(x, encoder_output)
        return x
