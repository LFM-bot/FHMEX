import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math


class FilterBlock(nn.Module):
    def __init__(self, alpha=0.1, beta=0.2, n_layers=1, n_heads=2, hidden_size=64,
                 inner_size=256, hidden_dropout_prob=0., hidden_act='gelu', layer_norm_eps=1e-12,
                 initializer_range=0.02, kernel_size=7, MAX_ITEM_LIST_LENGTH=197, freq_dropout_prob=0.4,
                 num_bands=6, conv_layers=1):
        super(FilterBlock, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.inner_size = inner_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range

        config = {}
        config['alpha'] = alpha
        config['beta'] = beta
        config['n_layers'] = n_layers
        config['n_heads'] = n_heads
        config['hidden_size'] = hidden_size
        config['inner_size'] = inner_size
        config['hidden_dropout_prob'] = hidden_dropout_prob
        config['hidden_act'] = hidden_act
        config['layer_norm_eps'] = layer_norm_eps
        config['initializer_range'] = initializer_range

        config['kernel_size'] = kernel_size
        config['MAX_ITEM_LIST_LENGTH'] = MAX_ITEM_LIST_LENGTH
        config['freq_dropout_prob'] = freq_dropout_prob
        config['num_bands'] = num_bands
        config['conv_layers'] = conv_layers

        self.lfm_encoder = CrossModalLFMEncoder(config)
        self.gfm_encoder = CrossModalGFMEncoder(config)
        self.concat_layer = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

        self.freq_conv_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode='reflect'
            ),
            nn.BatchNorm1d(hidden_size),
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Conv1d):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, sequence_emb):
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        seq_mask = torch.ones(sequence_emb.shape[:-1], dtype=torch.bool).to(sequence_emb.device)
        # UAF
        frequency_emb = torch.fft.rfft(sequence_emb, dim=1, norm='ortho')
        filter = torch.sigmoid(self.freq_conv_encoder(frequency_emb.abs().permute(0, 2, 1)))

        # GFM
        gfm_layer = self.gfm_encoder(sequence_emb, seq_mask, filter, output_all_encoded_layers=True)
        gfm_output = gfm_layer[-1]

        # LFM
        item_encoded_layers, total_lb_loss = self.lfm_encoder(sequence_emb, seq_mask, filter,
                                                              output_all_encoded_layers=True)
        lfm_output = item_encoded_layers[-1]

        concate_output = torch.cat((lfm_output, gfm_output), dim=-1)
        output = self.concat_layer(concate_output)

        output = self.LayerNorm(output)
        output = self.dropout(output)
        # return output, gfm_output, lfm_output, total_lb_loss
        return output


class CrossModalGlobalFilterBlock(nn.Module):
    def __init__(self, alpha=0.1, beta=0.2, n_layers=1, n_heads=2, hidden_size=64,
                 inner_size=256, hidden_dropout_prob=0., hidden_act='gelu', layer_norm_eps=1e-12,
                 initializer_range=0.02, kernel_size=7, MAX_ITEM_LIST_LENGTH=197, freq_dropout_prob=0.4,
                 num_bands=6, conv_layers=1):
        super(CrossModalGlobalFilterBlock, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.inner_size = inner_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range

        config = {}
        config['alpha'] = alpha
        config['beta'] = beta
        config['n_layers'] = n_layers
        config['n_heads'] = n_heads
        config['hidden_size'] = hidden_size
        config['inner_size'] = inner_size
        config['hidden_dropout_prob'] = hidden_dropout_prob
        config['hidden_act'] = hidden_act
        config['layer_norm_eps'] = layer_norm_eps
        config['initializer_range'] = initializer_range

        config['kernel_size'] = kernel_size
        config['MAX_ITEM_LIST_LENGTH'] = MAX_ITEM_LIST_LENGTH
        config['freq_dropout_prob'] = freq_dropout_prob
        config['num_bands'] = num_bands
        config['conv_layers'] = conv_layers

        self.gfm_encoder = CrossModalGFMEncoder(config)
        # self.lfm_encoder = LFMEncoder(config)
        # self.concat_layer = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

        self.freq_conv_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode='reflect'
            ),
            nn.BatchNorm1d(hidden_size),
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, text_emb, img_emb):
        text_emb, img_emb = self.LayerNorm(text_emb), self.LayerNorm(img_emb)
        text_emb, img_emb = self.dropout(text_emb), self.dropout(img_emb)

        seq_mask = torch.ones(text_emb.shape[:-1], dtype=torch.bool).to(text_emb.device)
        # UAF
        text_frequency_emb = torch.fft.rfft(text_emb, dim=1, norm='ortho')
        img_frequency_emb = torch.fft.rfft(img_emb, dim=1, norm='ortho')

        text_filter = torch.sigmoid(self.freq_conv_encoder(text_frequency_emb.abs().permute(0, 2, 1)))
        img_filter = torch.sigmoid(self.freq_conv_encoder(img_frequency_emb.abs().permute(0, 2, 1)))

        # GFM
        gfm_layer = self.gfm_encoder(text_emb, img_emb, seq_mask, text_filter, img_filter)
        gfm_output = gfm_layer[-1]

        return gfm_output

    def get_filter(self, sequence_emb):
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        # UAF
        frequency_emb = torch.fft.rfft(sequence_emb, dim=1, norm='ortho')
        filter = torch.sigmoid(self.freq_conv_encoder(frequency_emb.abs().permute(0, 2, 1)))

        return filter


class CrossModalLocalFilterBlock(nn.Module):
    def __init__(self, alpha=0.1, beta=0.2, n_layers=1, n_heads=2, hidden_size=64,
                 inner_size=256, hidden_dropout_prob=0., hidden_act='gelu', layer_norm_eps=1e-12,
                 initializer_range=0.02, kernel_size=7, MAX_ITEM_LIST_LENGTH=197, freq_dropout_prob=0.4,
                 num_bands=6, conv_layers=1):
        super(CrossModalLocalFilterBlock, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.inner_size = inner_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range

        config = {}
        config['alpha'] = alpha
        config['beta'] = beta
        config['n_layers'] = n_layers
        config['n_heads'] = n_heads
        config['hidden_size'] = hidden_size
        config['inner_size'] = inner_size
        config['hidden_dropout_prob'] = hidden_dropout_prob
        config['hidden_act'] = hidden_act
        config['layer_norm_eps'] = layer_norm_eps
        config['initializer_range'] = initializer_range

        config['kernel_size'] = kernel_size
        config['MAX_ITEM_LIST_LENGTH'] = MAX_ITEM_LIST_LENGTH
        config['freq_dropout_prob'] = freq_dropout_prob
        config['num_bands'] = num_bands
        config['conv_layers'] = conv_layers

        self.lfm_encoder = CrossModalLFMEncoder(config)
        # self.gfm_encoder = GFMEncoder(config)
        # self.concat_layer = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

        self.freq_conv_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode='reflect'
            ),
            nn.BatchNorm1d(hidden_size),
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, text_emb, img_emb):
        text_emb, img_emb = self.LayerNorm(text_emb), self.LayerNorm(img_emb)
        text_emb, img_emb = self.dropout(text_emb), self.dropout(img_emb)
        seq_mask = torch.ones(text_emb.shape[:-1], dtype=torch.bool).to(text_emb.device)
        # UAF
        text_frequency_emb = torch.fft.rfft(text_emb, dim=1, norm='ortho')
        img_frequency_emb = torch.fft.rfft(img_emb, dim=1, norm='ortho')

        text_filter = torch.sigmoid(self.freq_conv_encoder(text_frequency_emb.abs().permute(0, 2, 1)))
        img_filter = torch.sigmoid(self.freq_conv_encoder(img_frequency_emb.abs().permute(0, 2, 1)))

        # LFM
        item_encoded_layers, total_lb_loss = self.lfm_encoder(text_emb, img_emb, seq_mask, text_filter, img_filter,
                                                              output_all_encoded_layers=True)
        lfm_output = item_encoded_layers[-1]

        return lfm_output


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish, 'silu': F.silu}


class LFMGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_bands = config['num_bands']

        self.gate = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, self.num_bands)
        )

    def forward(self, x):
        magnitude = x.abs()
        phase = torch.angle(x)
        mag_features = torch.mean(magnitude, dim=1)
        phase_features = torch.mean(phase, dim=1)
        combined_features = torch.cat([mag_features, phase_features], dim=-1)

        gate_logits = self.gate(combined_features)
        probs = F.softmax(gate_logits, dim=-1)

        local_band_prob, prob_indices = torch.topk(probs, self.num_bands, dim=-1)
        local_band_prob_normalized = local_band_prob / local_band_prob.sum(dim=-1, keepdim=True)

        return local_band_prob_normalized, prob_indices


class CM_LFMfilterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.complex_weight1 = nn.Parameter(
            torch.randn(1, config['hidden_size'], config['MAX_ITEM_LIST_LENGTH'] // 2 + 1, 2,
                        dtype=torch.float32) * 0.02)
        self.complex_weight2 = nn.Parameter(
            torch.randn(1, config['hidden_size'], config['MAX_ITEM_LIST_LENGTH'] // 2 + 1, 2,
                        dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(config['freq_dropout_prob'])
        self.conv_layers = config['conv_layers']
        self.hidden_size = config['hidden_size']
        self.kernel_size = config['kernel_size']
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=1e-12)
        self.num_bands = config['num_bands']
        self.LFMgate = LFMGate(config)
        self.freq_conv_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                padding_mode='reflect'
            ),
            nn.BatchNorm1d(self.hidden_size),
        )
        self.LFMgate = LFMGate(config)

    def compute_balance_loss(self, local_band_indices, local_band_prob):
        batch_size = local_band_indices.size(0)
        mask = F.one_hot(local_band_indices, num_classes=self.num_bands).float()
        weighted_mask = mask * local_band_prob.unsqueeze(-1)
        band_usage = weighted_mask.sum(dim=[0, 1])
        band_usage = band_usage / batch_size
        ideal_usage = torch.ones_like(band_usage) * (1 / self.num_bands)
        usage_penalty = (band_usage - ideal_usage) ** 2
        balance_loss = usage_penalty.mean()
        return balance_loss, band_usage

    def forward(self, text_tensor, img_tensor, seq_mask, text_filter, img_filter):
        batch, max_len, hidden = text_tensor.shape
        text_x = torch.fft.rfft(text_tensor, dim=1, norm='ortho')
        img_x = torch.fft.rfft(img_tensor, dim=1, norm='ortho')

        local_band_prob, prob_indices = self.LFMgate(text_x + img_x)
        balance_loss, band_usage = self.compute_balance_loss(prob_indices, local_band_prob)

        # cross-modal filter
        text_weight = torch.view_as_complex(self.complex_weight1)
        img_weight = torch.view_as_complex(self.complex_weight2)
        text_filtered_w = torch.complex(img_filter * img_weight.real, img_filter * img_weight.imag)
        text_x_ = text_x * text_filtered_w.permute(0, 2, 1)
        img_filtered_w = torch.complex(text_filter * text_weight.real, text_filter * text_weight.imag)
        img_x_ = img_x * img_filtered_w.permute(0, 2, 1)
        mm_x_ = text_x_ + img_x_

        frequency_bands = torch.empty((batch, self.num_bands, max_len, hidden), device=text_tensor.device,
                                      dtype=text_tensor.dtype)
        for band in range(self.num_bands):
            frequency_output = torch.zeros_like(mm_x_)
            band_start = band * (max_len // 2 + 1) // self.num_bands
            band_end = (band + 1) * (max_len // 2 + 1) // self.num_bands
            frequency_output[:, band_start:band_end] = mm_x_[:, band_start:band_end]
            sequence_emb_fft = torch.fft.irfft(frequency_output, n=max_len, dim=1, norm='ortho')

            band_output = self.out_dropout(sequence_emb_fft)
            frequency_bands[:, band] = self.LayerNorm(band_output + text_tensor)

        selected = torch.gather(frequency_bands, dim=1,
                                index=prob_indices.view(batch, self.num_bands, 1, 1).expand(-1, -1, max_len, hidden))
        weighted_bands = local_band_prob.view(batch, self.num_bands, 1, 1) * selected
        LFM_output = weighted_bands.sum(dim=1)

        return LFM_output, balance_loss


class Intermediate(nn.Module):
    def __init__(self, config):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(config['hidden_size'], config['inner_size'])
        if isinstance(config['hidden_act'], str):
            self.intermediate_act_fn = ACT2FN[config['hidden_act']]
        else:
            self.intermediate_act_fn = config['hidden_act']

        self.dense_2 = nn.Linear(config['inner_size'], config['hidden_size'])
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=1e-12)
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class CrossModalLFMLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.filterlayer = CM_LFMfilterLayer(config)
        self.intermediate = Intermediate(config)

    def forward(self, text_emb, img_emb, seq_mask, text_filter, img_filter):
        LFM_output, balance_loss = self.filterlayer(text_emb, img_emb, seq_mask, text_filter, img_filter)
        output = self.intermediate(LFM_output)
        return output, balance_loss


class CrossModalLFMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = CrossModalLFMLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config['n_layers'])])

    def forward(self, text_emb, img_emb, seq_mask, text_filter, img_filter, output_all_encoded_layers=True):
        all_encoder_layers = []
        total_balance_loss = 0

        for layer_module in self.layer:
            text_emb, balance_loss = layer_module(text_emb, img_emb, seq_mask, text_filter, img_filter)
            total_balance_loss += balance_loss

            if output_all_encoded_layers:
                all_encoder_layers.append((text_emb))

        if not output_all_encoded_layers:
            all_encoder_layers.append((text_emb))

        return all_encoder_layers, total_balance_loss


class CM_GFMFilterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.complex_weight1 = nn.Parameter(
            torch.randn(1, config['hidden_size'], config['MAX_ITEM_LIST_LENGTH'] // 2 + 1, 2,
                        dtype=torch.float32) * 0.02)
        self.complex_weight2 = nn.Parameter(
            torch.randn(1, config['hidden_size'], config['MAX_ITEM_LIST_LENGTH'] // 2 + 1, 2,
                        dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(config['freq_dropout_prob'])
        self.conv_layers = config['conv_layers']
        self.hidden_size = config['hidden_size']
        self.kernel_size = config['kernel_size']
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=1e-12)
        self.num_bands = config['num_bands']
        self.freq_conv_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                padding_mode='reflect'
            ),
            nn.BatchNorm1d(self.hidden_size),
        )

    def forward(self, text_tensor, img_tensor, seq_mask, text_filter, img_filter):
        batch, max_len, hidden = text_tensor.shape
        text_x = torch.fft.rfft(text_tensor, dim=1, norm='ortho')
        text_weight = torch.view_as_complex(self.complex_weight1)
        img_x = torch.fft.rfft(img_tensor, dim=1, norm='ortho')
        img_weight = torch.view_as_complex(self.complex_weight2)

        # cross-modal filter
        text_filtered_w = torch.complex(img_filter * img_weight.real, img_filter * img_weight.imag)
        text_x_ = text_x * text_filtered_w.permute(0, 2, 1)
        img_filtered_w = torch.complex(text_filter * text_weight.real, text_filter * text_weight.imag)
        img_x_ = img_x * img_filtered_w.permute(0, 2, 1)

        # multi-modal fusion
        mm_x_ = text_x_ + img_x_

        whole_sequence_emb_irfft = torch.fft.irfft(mm_x_, n=max_len, dim=1, norm='ortho')
        whole_emb = self.out_dropout(whole_sequence_emb_irfft)
        whole_emb = self.LayerNorm(whole_emb + text_tensor)
        return whole_emb


class CrossModalGFMLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.filterlayer = CM_GFMFilterLayer(config)
        self.intermediate = Intermediate(config)
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=1e-12)

    def forward(self, text_emb, img_emb, seq_mask, text_filter, img_filter):
        gfm_output = self.filterlayer(text_emb, img_emb, seq_mask, text_filter, img_filter)
        output = self.intermediate(gfm_output)
        return output


class CrossModalGFMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = CrossModalGFMLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config['n_layers'])])

    def forward(self, text_emb, img_emb, seq_mask, text_filter, img_filter,
                output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            text_emb = layer_module(text_emb, img_emb, seq_mask, text_filter, img_filter)
            if output_all_encoded_layers:
                all_encoder_layers.append(text_emb)

        if not output_all_encoded_layers:
            all_encoder_layers.append(text_emb)

        return all_encoder_layers


from timm.models.vision_transformer import Block

if __name__ == '__main__':
    hidden_size = 768
    # block = FilterBlock(hidden_size=hidden_size, inner_size=hidden_size * 4)

    x = torch.randn(24, 197, hidden_size)

    # output, gfm_output, lfm_output, total_lb_loss = block(x)
    #
    # print(output.shape)
    # print()

    block = Block(dim=hidden_size, num_heads=4)
    output = block(x)
    print(output.shape)
    print()
