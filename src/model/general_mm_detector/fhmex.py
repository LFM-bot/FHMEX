import timm
import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
import torchvision
from torch.nn.init import xavier_normal_, xavier_uniform_
from torchvision.transforms import Resize
from transformers import BertModel, AutoModel
from src.model.abstract_detector import AbstractDetector
from src.model.vision_module import mae_vit
from src.model.vision_module.inceptionet import GoogLeNet
from src.utils.utils import HyperParamDict
from timm.models.vision_transformer import Block
from src.model.encode_module import GlobalFilterBlock, LocalFilterBlock, CrossModalGlobalFilterBlock, \
    CrossModalLocalFilterBlock
from src.model.loss import InfoNCELoss
from positional_encodings.torch_encodings import (
    PositionalEncoding1D,
)
from src.dataset.dataset import chinese_datasets


class FHMEX(AbstractDetector):
    def __init__(self, config, additional_data_dict):
        """
        增加跨模态对比
        """
        super(FHMEX, self).__init__(config)
        self.batch_size = config.train_batch
        self.dataset = config.dataset
        self.num_expert = config.num_expert
        self.n_layers = config.n_layers
        self.kernel_size = config.kernel_size
        self.num_bands = config.num_bands
        self.freq_dropout_prob = config.freq_dropout_prob
        self.lamda1 = config.lamda1
        self.lamda2 = config.lamda2

        self.unified_dim, self.text_dim = 768, 768
        self.text_token_len, self.image_token_len = 197, 197
        out_dim = 1
        self.depth = 1  # 2

        # ================ Initialize models ================s
        # Image encoder: MAE
        model_size = 'base'
        self.image_model = mae_vit.__dict__["mae_vit_{}_patch16".format(model_size)](norm_pix_loss=False)
        checkpoint = torch.load(
            '/mnt1/userhome/tangpang/shichenglong/proj/LLMs/mae_checkpoint/mae_pretrain_vit_{}.pth'.format(
                model_size), map_location="cpu"
        )
        self.image_model.load_state_dict(checkpoint["model"], strict=False)

        # Text encoder: BERT
        model_name = (
            '/mnt1/userhome/tangpang/shichenglong/proj/LLMs/bert-base-chinese'
            if self.dataset in chinese_datasets
            else "/mnt1/userhome/tangpang/shichenglong/proj/LLMs/bert-base-uncased"
        )
        print("BERT: using {}".format(model_name))
        self.text_model = BertModel.from_pretrained(model_name)

        self.text_attention = TokenAttention(self.unified_dim)
        self.image_attention = TokenAttention(self.unified_dim)
        self.mm_attention = TokenAttention(self.unified_dim)

        # Register position IDs as buffers
        self.register_buffer('positional_mm',
                             torch.zeros(self.batch_size, self.image_token_len + self.text_token_len, self.unified_dim))
        self.register_buffer('positional_image', torch.zeros(self.batch_size, self.image_token_len, self.unified_dim))
        self.register_buffer('positional_text', torch.zeros(self.batch_size, self.text_token_len, self.unified_dim))
        self.register_buffer('positional_modal_representation', torch.zeros(self.batch_size, 3, self.unified_dim))

        # GATE, EXPERTS for features
        blocks = [GlobalFilterBlock, LocalFilterBlock, LocalFilterBlock]
        mm_blocks = [CrossModalGlobalFilterBlock, CrossModalLocalFilterBlock, CrossModalLocalFilterBlock]
        image_expert_list, text_expert_list, mm_expert_list = [], [], []
        for i in range(self.num_expert):
            text_expert_list.append(
                blocks[i](hidden_size=self.unified_dim, inner_size=4 * self.unified_dim,
                          n_layers=self.n_layers, kernel_size=self.kernel_size, num_bands=self.num_bands * i,
                          freq_dropout_prob=self.freq_dropout_prob))
            image_expert_list.append(
                blocks[i](hidden_size=self.unified_dim, inner_size=4 * self.unified_dim,
                          n_layers=self.n_layers, kernel_size=self.kernel_size, num_bands=self.num_bands * i,
                          freq_dropout_prob=self.freq_dropout_prob))
            mm_expert_list.append(
                mm_blocks[i](hidden_size=self.unified_dim, inner_size=4 * self.unified_dim,
                             n_layers=self.n_layers, kernel_size=self.kernel_size, num_bands=self.num_bands * i,
                             freq_dropout_prob=self.freq_dropout_prob))

        self.text_experts = nn.ModuleList(text_expert_list)
        self.image_experts = nn.ModuleList(image_expert_list)
        self.mm_experts = nn.ModuleList(mm_expert_list)

        self.image_gate_mae = nn.Sequential(
            nn.Linear(self.unified_dim, self.unified_dim),
            nn.SiLU(),
            nn.Linear(self.unified_dim, self.num_expert),
        )
        self.text_gate = nn.Sequential(
            nn.Linear(self.unified_dim, self.unified_dim),
            nn.SiLU(),
            nn.Linear(self.unified_dim, self.num_expert),
        )
        self.mm_gate = nn.Sequential(
            nn.Linear(self.unified_dim, self.unified_dim),
            nn.SiLU(),
            nn.Linear(self.unified_dim, self.num_expert),
        )

        # Final MOE for feature aggregation
        self.final_attention = nn.ModuleList(
            [TokenAttention(self.unified_dim) for i in range(1)]
        )
        self.fusion_SE_network_main_task = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.unified_dim, self.unified_dim),
                    nn.SiLU(),
                    nn.Linear(self.unified_dim, self.num_expert),
                )
                for _ in range(1)
            ]
        )

        # Classification head and mlp
        self.mix_classifier = nn.Sequential(
            nn.Linear(64, out_dim),
        )
        self.mix_trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
        )
        self.text_trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
        )
        self.text_alone_classifier = nn.Sequential(
            nn.Linear(64, out_dim),
        )
        self.image_trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
        )
        self.image_alone_classifier = nn.Sequential(
            nn.Linear(64, out_dim),
        )

        # Mapping mlp
        self.mapping_IS_MLP_mu = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_IS_MLP_sigma = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_T_MLP_mu = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_T_MLP_sigma = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.adaIN = AdaIN()

        def _expert():
            fusing_expert_ls = []
            for i in range(self.num_expert):
                fusing_expert = []
                for j in range(self.depth):
                    fusing_expert.append(Block(dim=self.unified_dim, num_heads=4))
                fusing_expert = nn.ModuleList(fusing_expert)
                fusing_expert_ls.append(fusing_expert)
            return nn.ModuleList(fusing_expert_ls)

        self.final_fusing_experts = nn.ModuleList([_expert() for i in range(1)])

        pos_weight = torch.tensor(config.pos_weight)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.specialization_loss = nn.TripletMarginLoss(margin=1, p=2)
        self.alignment_loss = InfoNCELoss(temperature=1., similarity_type='dot')

        self.img_align_lin1 = nn.Identity()
        self.img_align_lin2 = nn.Linear(self.unified_dim, self.unified_dim)
        self.text_align_lin1 = nn.Identity()
        self.text_align_lin2 = nn.Linear(self.unified_dim, self.unified_dim)
        self.mm_align_lin1 = nn.Identity()
        self.mm_align_lin2 = nn.Linear(self.unified_dim, self.unified_dim)

    def inner_forward(self, data_dict):
        input_ids = data_dict['input_ids']
        attention_mask = data_dict['attention_mask']
        token_type_ids = data_dict['token_type_ids']
        image = data_dict['image']
        batch_size = image.shape[0]

        # Positional embeddings
        p_1d_mm = PositionalEncoding1D(self.unified_dim)
        x_mm = torch.rand(
            batch_size, self.image_token_len, self.unified_dim
        )
        self.positional_mm = p_1d_mm(x_mm).to(self.dev)
        p_1d_image = PositionalEncoding1D(self.unified_dim)
        x_image = torch.rand(batch_size, self.image_token_len, self.unified_dim)
        self.positional_image = p_1d_image(x_image).to(self.dev)
        p_1d_text = PositionalEncoding1D(self.unified_dim)
        x_text = torch.rand(batch_size, self.text_token_len, self.unified_dim)
        self.positional_text = p_1d_text(x_text).to(self.dev)
        p_1d = PositionalEncoding1D(self.unified_dim)
        x = torch.rand(batch_size, 3, self.unified_dim)
        self.positional_modal_representation = p_1d(x).to(self.dev)

        # Feature encoding
        raw_image_feature = self.image_model.forward_ying(image)
        raw_text_feature = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0]

        # Token-attention
        text_atn_feature, _ = self.text_attention(raw_text_feature)  # [B, D_t]
        image_atn_feature, _ = self.image_attention(raw_image_feature)  # [D, D_i]
        mm_atn_feature, _ = self.mm_attention(
            torch.cat((raw_image_feature, raw_text_feature), dim=1)
        )

        # Gate scores for MOE
        gate_image_feature = self.image_gate_mae(image_atn_feature)
        gate_text_feature = self.text_gate(text_atn_feature)  # 64 320
        gate_mm_feature = self.mm_gate(mm_atn_feature)

        # Image MOE
        image_expert_feature_1 = []
        image_expert_feature_2 = []
        for i in range(self.num_expert):
            image_expert = self.image_experts[i]
            image_expert_feature_1.append(image_expert(raw_image_feature + self.positional_image)[:, 0])
            image_expert_feature_2.append(image_expert(raw_image_feature + self.positional_image)[:, 0])

        image_expert_feature_1 = torch.stack(image_expert_feature_1, dim=0)  # [num_expert, B, D]
        image_expert_feature_2 = torch.stack(image_expert_feature_2, dim=0)
        image_spe_loss, image_align_loss = 0, 0
        shared_image_feature = 0
        for i in range(self.num_expert):
            anchor = image_expert_feature_1[i]  # [B, D]
            pos = image_expert_feature_2[i]
            neg = (image_expert_feature_1.sum(dim=0) - anchor) / (self.num_expert - 1)
            # specialization loss
            image_spe_loss += self.specialization_loss(anchor, pos, neg)
            # MOE output
            shared_image_feature += anchor * gate_image_feature[:, i].unsqueeze(1)

        # alignment loss
        for i in range(self.num_expert):
            image_align_loss += self.alignment_loss(self.img_align_lin1(image_expert_feature_2[i]),
                                                    self.img_align_lin2(shared_image_feature))

        image_spe_loss /= self.num_expert
        image_align_loss /= self.num_expert

        # Text MOE
        text_expert_feature_1 = []
        text_expert_feature_2 = []
        for i in range(self.num_expert):
            text_expert = self.text_experts[i]
            text_expert_feature_1.append(text_expert(raw_text_feature + self.positional_text)[:, 0])
            text_expert_feature_2.append(text_expert(raw_text_feature + self.positional_text)[:, 0])

        text_expert_feature_1 = torch.stack(text_expert_feature_1, dim=0)  # [num_expert, B, D]
        text_expert_feature_2 = torch.stack(text_expert_feature_2, dim=0)
        text_spe_loss, text_align_loss = 0, 0
        shared_text_feature = 0
        for i in range(self.num_expert):
            anchor = text_expert_feature_1[i]  # [B, D]
            pos = text_expert_feature_2[i]
            neg = (text_expert_feature_1.sum(dim=0) - anchor) / (self.num_expert - 1)
            # specialization loss
            text_spe_loss += self.specialization_loss(anchor, pos, neg)
            # MOE output
            shared_text_feature += anchor * gate_text_feature[:, i].unsqueeze(1)

        # alignment loss
        for i in range(self.num_expert):
            text_align_loss += self.alignment_loss(self.text_align_lin1(text_expert_feature_2[i]),
                                                   self.text_align_lin2(shared_text_feature))

        text_spe_loss /= self.num_expert
        text_align_loss /= self.num_expert

        # MM MOE
        mm_expert_feature_1 = []
        mm_expert_feature_2 = []
        for i in range(self.num_expert):
            mm_img_expert = self.mm_experts[i]
            mm_expert_feature_1.append(mm_img_expert(raw_text_feature + self.positional_mm,
                                                     raw_image_feature + self.positional_mm)[:, 0])
            mm_expert_feature_2.append(mm_img_expert(raw_text_feature + self.positional_mm,
                                                     raw_image_feature + self.positional_mm)[:, 0])

        mm_expert_feature_1 = torch.stack(mm_expert_feature_1, dim=0)  # [num_expert, B, D]
        mm_expert_feature_2 = torch.stack(mm_expert_feature_2, dim=0)
        mm_spe_loss, mm_align_loss = 0, 0
        shared_mm_feature = 0
        for i in range(self.num_expert):
            anchor = mm_expert_feature_1[i]
            pos = mm_expert_feature_2[i]
            neg = (mm_expert_feature_1.sum(dim=0) - anchor) / (self.num_expert - 1)
            # specialization loss
            mm_spe_loss += self.specialization_loss(anchor, pos, neg)
            # MOE output
            shared_mm_feature += anchor * gate_mm_feature[:, i].unsqueeze(1)

        # alignment loss
        for i in range(self.num_expert):
            mm_align_loss += self.alignment_loss(self.mm_align_lin1(mm_expert_feature_2[i]),
                                                 self.mm_align_lin2(shared_mm_feature))

        mm_spe_loss /= self.num_expert
        mm_align_loss /= self.num_expert

        shared_image_feature_lite = self.image_trim(shared_image_feature)
        shared_text_feature_lite = self.text_trim(shared_text_feature)

        # Single modality prediction
        image_only_output = self.image_alone_classifier(shared_image_feature_lite)
        text_only_output = self.text_alone_classifier(shared_text_feature_lite)

        # Re-standardization
        img_mu = self.mapping_IS_MLP_mu(torch.sigmoid(image_only_output).clone().detach())
        img_sigma = self.mapping_IS_MLP_sigma(torch.sigmoid(image_only_output).clone().detach())

        text_mu = self.mapping_T_MLP_mu(torch.sigmoid(text_only_output).clone().detach())
        text_sigma = self.mapping_T_MLP_sigma(torch.sigmoid(text_only_output).clone().detach())

        shared_image_feature = self.adaIN(shared_image_feature, img_mu, img_sigma)
        shared_text_feature = self.adaIN(shared_text_feature, text_mu, text_sigma)
        shared_mm_feature = shared_mm_feature

        # Final output MOE
        concat_feature_main_biased = torch.stack([shared_image_feature, shared_text_feature,
                                                  shared_mm_feature], dim=1)
        final_feature_main_task = 0
        fusion_tempfeat_main_task, _ = self.final_attention[0](concat_feature_main_biased)
        gate_main_task = self.fusion_SE_network_main_task[0](fusion_tempfeat_main_task)
        for i in range(self.num_expert):
            fusing_expert = self.final_fusing_experts[0][i]
            for j in range(self.depth):
                tmp_fusion_feature = fusing_expert[j](
                    concat_feature_main_biased + self.positional_modal_representation
                )
            tmp_fusion_feature = tmp_fusion_feature[:, 0]
            final_feature_main_task += tmp_fusion_feature * gate_main_task[:, i].unsqueeze(1)
        final_feature_main_task_lite = self.mix_trim(final_feature_main_task)

        final_output = self.mix_classifier(final_feature_main_task_lite)

        spe_loss = (image_spe_loss + text_spe_loss + mm_spe_loss) / 3.
        align_loss = (image_align_loss + text_spe_loss + mm_spe_loss) / 3.

        return final_output, image_only_output, text_only_output, spe_loss, align_loss, final_feature_main_task_lite

    def get_news_emb(self, data_dict):
        news_emb = self.inner_forward(data_dict)[-1]

        return news_emb

    def forward(self, data_dict):

        final_output = self.inner_forward(data_dict)[0]
        logits = torch.sigmoid(final_output)

        return logits

    def calc_loss(self, data_dict: dict):
        labels = data_dict['label']
        final_output, image_only_output, text_only_output, spe_loss, align_loss, _ = self.inner_forward(data_dict)

        loss_main = self.criterion(final_output, labels.float().unsqueeze(1))
        loss_image = self.criterion(image_only_output, labels.float().unsqueeze(1))
        loss_text = self.criterion(text_only_output, labels.float().unsqueeze(1))
        loss_single_modal = (loss_text + loss_image) / 2
        # loss_cl = self.nce_loss(image_feature, text_feature)
        # loss = loss_main + 1. * loss_single_modal + self.lamda * loss_cl
        loss = loss_main + 1. * loss_single_modal + self.lamda1 * spe_loss + (1 - self.lamda1) * align_loss
        # loss = loss_main + 1. * loss_single_modal + self.lamda1 * spe_loss
        # loss = loss_main + 1. * loss_single_modal

        return loss

    def get_pretrain_features(self, input_ids, attention_mask, token_type_ids, image):
        image_feature = self.image_model.forward_ying(image)
        text_feature = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0]

        return image_feature, text_feature


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        # discriminator (without classifier)
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # generator (MLP-based VAE structure)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)  # 假设隐变量维度为hidden_dim
        )
        self.enc_fc_z_mean = nn.Linear(hidden_dim, hidden_dim)
        self.enc_fc_z_log_var = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.discriminator(x)

    def encode(self, x):
        h = self.encoder(x)
        z_mean = self.enc_fc_z_mean(h)
        z_log_var = self.enc_fc_z_log_var(h)
        return z_mean, z_log_var

    def decode(self, z):
        h = self.decoder(z)
        return h

    def reparameterize(self, z_mean, z_log_var, num_samples=1):
        z_std = (z_log_var * 0.5).exp()
        z_std = z_std.unsqueeze(1).expand(-1, num_samples, -1)
        z_mean = z_mean.unsqueeze(1).expand(-1, num_samples, -1)
        unit_normal = torch.randn_like(z_std)
        z = z_mean + unit_normal * z_std
        z = z.view(-1, z_std.size(2))
        return z

    def forward_generate(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        recon_x = self.decode(z)
        return recon_x, z_mean, z_log_var


class SimpleGate(nn.Module):
    def __init__(self, dim=1):
        super(SimpleGate, self).__init__()
        self.dim = dim

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=self.dim)
        return x1 * x2


class TokenAttention(torch.nn.Module):
    """
    Compute attention layer
    """

    def __init__(self, input_shape):
        super(TokenAttention, self).__init__()
        self.attention_layer = nn.Sequential(
            torch.nn.Linear(input_shape, input_shape),
            SimpleGate(dim=2),
            torch.nn.Linear(int(input_shape / 2), 1),
        )

    def forward(self, inputs):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))  # [B, L]
        # scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        scores = scores.unsqueeze(1)  # [B, 1, L]
        outputs = torch.matmul(scores, inputs).squeeze(1)
        # scores = self.attention_layer(inputs)
        # outputs = scores*inputs
        return outputs, scores


class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction="none", log_target=True)

    def forward(self, p, q):
        eps = 1e-10
        p, q = p + eps, q + eps
        p = p / p.sum(dim=-1, keepdim=True)
        q = q / q.sum(dim=-1, keepdim=True)

        # Applying a small epsilon to avoid log(0)
        m = 0.5 * (p + q)
        log_m = m.log()
        kl_pm = self.kl(log_m, p.log())
        kl_qm = self.kl(log_m, q.log())
        jsd = 0.5 * (kl_pm + kl_qm).sum(dim=-1)
        return jsd


class InteractionModule(nn.Module):
    def __init__(self, unified_dim, agr_threshold, sem_threshold):
        agr_threshold = 0.3
        sem_threshold = 0.3
        balance_loss_coef = 0.1
        router_z_loss_coef = 0.01
        interaction_loss_coef = 0.7
        super(InteractionModule, self).__init__()
        self.jsd_module = JSD()
        self.kl = nn.KLDivLoss(reduction="none", log_target=True)
        # 256
        self.unified_dim = 64
        self.modality_attn = TokenAttention(self.unified_dim)
        self.soft_gate = nn.Sequential(
            nn.Linear(self.unified_dim, self.unified_dim),
            nn.SiLU(),
            nn.Linear(self.unified_dim, 4),
        )

        self.agr_threshold = torch.tensor(agr_threshold, requires_grad=False)
        self.sem_threshold = torch.tensor(sem_threshold, requires_grad=False)
        # self.noisy_gate = False
        # self.noise_scale = 1.5
        # self.log_scale = None
        self.interaction_loss = nn.CrossEntropyLoss()
        self.interaction_loss_coef = interaction_loss_coef
        self.balance_loss_coef = balance_loss_coef
        self.router_z_loss_coef = router_z_loss_coef

    def clip_similarity(self, text_embeds, image_embeds):
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        clip_scores = (image_embeds * text_embeds).sum(dim=-1)
        return clip_scores

    def compute_router_z_loss(self, gate1_logits, gate2_logits):
        logits = torch.cat([gate1_logits, gate2_logits], dim=1)
        max_logits = torch.logsumexp(logits, dim=1)
        router_z_loss = torch.mean(max_logits ** 2)
        return self.router_z_loss_coef * router_z_loss

    def compute_balance_loss(self, assignments):
        num_experts = 4
        batch_size = assignments.size(0)
        expert_counts = torch.zeros(num_experts)
        for i in range(num_experts):
            expert_counts[i] = (assignments == i).sum()
        distribution = expert_counts / batch_size
        target_distribution = torch.full_like(distribution, fill_value=1 / num_experts)
        balancing_loss = F.mse_loss(distribution, target_distribution)
        return balancing_loss

    def forward(self, p_t, p_i, e_t, e_i, m_t, m_i):
        """
        p_t: text_only_output,
        p_i: image_only_output,
        e_t: shared_text_feature_lite,
        e_i: shared_image_feature_lite,
        m_t: projected_mt,
        m_i: projected_mi,
        """
        # Compute supervision signal
        js_div = self.jsd_module(p_t, p_i)
        clip_score = self.clip_similarity(m_t, m_i)

        agr_gate_scores = (js_div < self.agr_threshold).type(torch.int64)
        sem_gate_scores = (clip_score > self.sem_threshold).type(torch.int64)

        stacked_features = torch.stack(
            (e_t, e_i, m_t, m_i),
            dim=1,
        )

        gate_inputs, _ = self.modality_attn(stacked_features)
        gate_logits = self.soft_gate(gate_inputs)
        targets = 2 * agr_gate_scores + sem_gate_scores

        agr_logits = gate_logits[:, :2]
        sem_logits = gate_logits[:, 2:]
        interaction_loss = self.interaction_loss_coef * (
            self.interaction_loss(gate_logits, targets)
        )

        agr_gate = torch.argmax(F.softmax(agr_logits, dim=1), dim=1)
        sem_gate = torch.argmax(F.softmax(sem_logits, dim=1), dim=1)
        dispatch_index = agr_gate * 2 + sem_gate
        router_z_loss = self.router_z_loss_coef * self.compute_router_z_loss(
            agr_logits, sem_logits
        )

        # no gradient for dispatch_index !!
        dispatch_index = dispatch_index.detach()
        balance_loss = self.balance_loss_coef * self.compute_balance_loss(
            dispatch_index
        )
        expert_mask = F.softmax(gate_logits, dim=1)
        gate_loss = interaction_loss + router_z_loss + balance_loss
        return expert_mask, gate_loss


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        """Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x, (1)) / (x.shape[1])

    def sigma(self, x):
        """Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""
        return torch.sqrt(
            (
                    torch.sum((x.permute([1, 0]) - self.mu(x)).permute([1, 0]) ** 2, (1))
                    + 0.000000023
            )
            / (x.shape[1])
        )

    def forward(self, x, mu, sigma):
        """Takes a content embeding x and a style embeding y and changes
        transforms the mean and standard deviation of the content embedding to
        that of the style. [See eq. 8 of paper] Note the permutations are
        required for broadcasting"""
        # print(mu.shape) # 12
        x_mean = self.mu(x)
        x_std = self.sigma(x)
        x_reduce_mean = x.permute([1, 0]) - x_mean
        x_norm = x_reduce_mean / x_std
        # print(x_mean.shape) # 768, 12
        return (sigma.squeeze(1) * (x_norm + mu.squeeze(1))).permute([1, 0])


class SimpleGate(nn.Module):
    def __init__(self, dim=1):
        super(SimpleGate, self).__init__()
        self.dim = dim

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=self.dim)
        return x1 * x2


class TokenAttention(torch.nn.Module):
    """
    Compute attention layer
    """

    def __init__(self, input_shape):
        super(TokenAttention, self).__init__()
        self.attention_layer = nn.Sequential(
            torch.nn.Linear(input_shape, input_shape),
            nn.SiLU(),
            torch.nn.Linear(input_shape, 1),
        )

    def forward(self, inputs):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        scores = scores.unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        return outputs, scores


def FHMEX_config():
    parser = HyperParamDict('Model hyper-parameters for FHMEX')
    parser.add_argument('--model', default='FHMEX', type=str)
    parser.add_argument('--model_type', default='General', choices=['General', 'LLM-based'], type=str)
    parser.add_argument('--num_expert', default=2, type=int)
    parser.add_argument('--n_layers', default=1, type=int)
    parser.add_argument('--kernel_size', default=7, type=int)
    parser.add_argument('--num_bands', default=6, type=int)
    parser.add_argument('--freq_dropout_prob', default=0.4, type=float)
    parser.add_argument('--lamda1', default=0.1, type=float)
    parser.add_argument('--lamda2', default=0.1, type=float)

    return parser


if __name__ == '__main__':
    print(FreqMOE_V3_config())
