# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.01,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        #所有 dropout 层（如 attention、MLP）的dropout概率
        self.dropout = dropout
        # tokenizer 起始token ID
        self.bos_token_id = bos_token_id
        # tokenizer 结束token ID
        self.eos_token_id = eos_token_id
        # 激活函数，默认 silu
        self.hidden_act = hidden_act
        # 隐藏层维度（神经元的数量），即d_model
        self.hidden_size = hidden_size
        # Transformer 中前馈神经网络的维度
        self.intermediate_size = intermediate_size
        # 最大位置编码
        self.max_position_embeddings = max_position_embeddings
        # 多头的数量
        self.num_attention_heads = num_attention_heads
        # 隐藏层的数量
        self.num_hidden_layers = num_hidden_layers
        # Key/Value 的头数
        self.num_key_value_heads = num_key_value_heads
        # 词表大小
        self.vocab_size = vocab_size
        # RMSNorm 层的eps
        self.rms_norm_eps = rms_norm_eps
        # RoPE Theta
        self.rope_theta = rope_theta
        # 推理时是否使用RoPE缩放
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        # 是否使用 Flash Attention
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # 缩放系数
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # RMS(x) 公式
        # x.pow(2).mean(-1, keepdim=True)计算了输入x的平方的均值
        # torch.rsqrt是平方根的倒数，这样就得到了RMSNorm的分母部分，再加上eps防止分母为0
        # 最后乘以x，得到RMSNorm的结果
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 强制转为 float32 计算 norm 以保证数值精度，最后再转回 x 的类型 (如 float16)
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    """
    获得 sin/cos函数表
    """
    # 1. 计算 Theta。dim 是 head_dim (hidden_size // num_heads), rope_base=1000000
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    # todo：czl 没看懂
    # 2. YaRN 算法 (长文本外推逻辑)
    if rope_scaling is not None:
        # 获取配置参数
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        # 仅当推理长度 end 超过训练长度 orig_max 时触发
        if end / orig_max > 1.0:
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    # 3. 生成位置编码
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    # 4. 拼接 Cos 和 Sin
    # 注意：这里拼接了两次，是为了适配下面的 rotate_half 实现
    # 形状变为 [seq_len, dim]
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    将q和k应用RoPE
    """
    def rotate_half(x):
        # 辅助函数：将向量切分为两半，并交换顺序、取负  [x1, x2] -> [-x2, x1]
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # 应用欧拉公式的实数形式
    # q * cos.unsqueeze(unsqueeze_dim): 前半部分 x1 * cos(theta) 后半部分 x2 * cos(theta)
    # rotate_half(q): 输入 [x1, x2]  输出 [-x2, x1]
    # rotate_half(q) * sin.unsqueeze(unsqueeze_dim)：前半部分 -x2 * sin(theta) 后半部分 x1 * sin(theta)
    # 完整公式最后结果: 前半部分 x1 * cos(theta) - x2 * sin(theta) 后半部分 x2 * cos(theta) + x1 * sin(theta)
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    """
    当 Key 和 Value 的头数少于 Query 的头数时，通过重复 K/V 来对齐维度，将K和V的维度扩展到和Q的相同的维度
    """
    # X 维度为 [batch_size, seq_len, Key/Value的头数, head_dim]
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # expand 维度为 [batch_size, seq_len, Key/Value的头数, 新维度n_rep, head_dim]
    # reshape 的维度为：[batch_size, seq_len, Key/Value的头数 × n_rep, head_dim]
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    注意力机制 todo:czl 没读完
    """
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        # Key/Value 的头数
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        # Query 的头数
        self.n_local_heads = args.num_attention_heads
        # Key/Value 的头数
        self.n_local_kv_heads = self.num_key_value_heads
        # 每个 KV 头要被重复多少次才能匹配 Q 头
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度
        self.head_dim = args.hidden_size // args.num_attention_heads
        # QKV 投影层
        # Q: [batch_size, seq_len, hidden_size] → [batch_size, seq_len, Query的头数 × head_dim]
        # K/V: [batch_size, seq_len, hidden_size] → [batch_size, seq_len, Key/Value的头数 × head_dim]
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # o: 把多头拼接后的输出 投影回 hidden_size
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        # x: [batch_size, seq_len, hidden_size]
        bsz, seq_len, _ = x.shape
        # 投影得到公式里的QKV矩阵
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 将 Q 拆分成多头，维度为 [batch_size, seq_len, Query的头数, head_dim]
        # 将 K、V 拆分成多头，维度为 [batch_size, seq_len, Key/Value的头数, head_dim]
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        # 将QK旋转位置
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # kv_cache实现，记录KV缓存，将当前的K和V拼接到缓存中
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # 扩展 Key 和 Value 头维度
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # 注意力机制的公式实现：
            # scores 的维度是 [batch_size, Query的头数, seq_len, seq_len]
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # full 创建一个全 -inf 的(seq_len, seq_len)方阵
            # triu 创建一个下三角（含对角线）为 0，上三角为 -inf
            # unsqueeze 增加维度后[1, 1, seq_len, seq_len]
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)  # scores+mask

            if attention_mask is not None:
                # attention_mask 的维度是 [batch_size, seq_len]
                # extended_attention_mask 的维度为 [batch_size, 1, 1, seq_len]
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            # softmax
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            # output 的维度是 [batch_size, Query的头数, seq_len, head_dim]
            output = scores @ xv

        # 将多头的结果拼接起来, 先交换维度为 [batch_size, seq_len, Query的头数, head_dim]，再拼接成 [batch_size, seq_len, Query的头数 * head_dim]
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        # 最终投影回残差流
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    """
    前馈神经网络FNN
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # 输入的x是 [batch_size, seq_len, hidden_size]
        # 维度升维转换 gate_proj(x) 维度：[batch_size, seq_len, hidden_size] → [batch_size, seq_len, intermediate_size]
        # 激活函数 act_fn(gate_proj(x))  # 维度保持 [batch_size, seq_len, intermediate_size]
        # 维度升维转换 up_proj(x)  维度：[batch_size, seq_len, hidden_size] → [batch_size, seq_len, intermediate_size]
        # 维度降维转换 down_proj(x)  维度：[batch_size, seq_len, intermediate_size] → [batch_size, seq_len, hidden_size]
        # dropout 维度保持 [batch_size, seq_len, hidden_size]
        # 输出的结果是 [batch_size, seq_len, hidden_size]
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    """
    控制每个Token要交给哪些专家处理，并计算计算用于负载均衡的辅助损失aux_loss
    使用 Softmax 计算 Token 对每个专家的亲和度分数，选出分数最高的 K 个专家。
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        # weight的维度是 [n_routed_experts, hidden_size] 将输入的hidden_states映射到专家权重
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 使用 Kaiming 为每个专家初始化一个“打分向量”，通过点积衡量 token 与专家的匹配度。
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        # hidden_states的维度是 [batch_size, seq_len, hidden_size]
        bsz, seq_len, h = hidden_states.shape
        # 将hidden_states的维度降维为 [batch_size * seq_len, hidden_size]
        hidden_states = hidden_states.view(-1, h)
        # hidden_states @ weight^T 所以logits代表每个 token 对每个专家的原始得分，维度是 [batch_size * seq_len, n_routed_experts]
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            # scores 维度是 [batch_size * seq_len, n_routed_experts]
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        # 用softmax选出前 K 个专家
        # topk_weight是 专家权重，维度是 [batch_size * seq_len, num_experts_per_tok]
        # topk_idx是 每个token选择的专家索引，维度是 [batch_size * seq_len, num_experts_per_tok]
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            # softmax后对topk_weight归一化
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            # 在训练时强迫token均匀分配给各个专家，防止“专家负载不均衡”
            # scores_for_aux 维度是 [batch_size * seq_len, n_routed_experts]
            scores_for_aux = scores
            aux_topk = self.top_k
            # topk_idx_for_aux_loss是 每个token选择的专家索引，维度是 [batch_size, seq_len * hidden_size]
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                # Sequence 级负载均衡：不仅仅整个 Batch 要均衡，每个单独的序列（Sequence）内部也要均衡。
                # scores_for_seq_aux 维度是 [batch_size, seq_len, n_routed_experts], 按 batch 中每个 sequence 单独计算，表示每个 Sequence 中每个Token分配给每个专家的分数
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                # ce 统计每个 Sequence 中每个专家被选中的次数，ce的维度是 [batch_size, n_routed_experts]
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                # 计算Loss，按scores_for_seq_aux的seq_len维度求平均, 再跟 ce相乘
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # Batch 级负载均衡
                # mask_ce 统计每个专家被token选中的情况，ce的维度是 [batch_size, seq_len * hidden_size, n_routed_experts]
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                # ce 统计每个专家被选中的次数，ce的维度是 [n_routed_experts]
                ce = mask_ce.float().mean(0)
                # Pi：门控网络给给专家 i 分配token的平均概率
                Pi = scores_for_aux.mean(0)
                # fi: 实际上有多少 token 被分配给了专家 i
                fi = ce * self.n_routed_experts
                # 计算Loss
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            # 推理阶段不计算aux_loss
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """
    多专家前馈神经网络，有共享专家和路由分配专家
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        # 输入x的维度是 [batch_size, seq_len, hidden_size]
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        # topk_idx是 每个token选择的TopK专家索引，维度是 [batch_size, seq_len, num_experts_per_tok]
        # topk_weight是 每个token选择的TopK专家权重，维度是 [batch_size, seq_len, num_experts_per_tok]
        # aux_loss是 辅助损失，鼓励专家负载均衡
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # x的维度是 [batch_size * seq_len, hidden_size]
        x = x.view(-1, x.shape[-1])
        # flat_topk_idx的维度是 [batch_size * seq_len * num_experts_per_tok]
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            # 训练阶段
            # 扩展x的第一维的长度是 [batch_size * seq_len * num_experts_per_tok, hidden_size]
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            # 创建一个空Tensor，用于存放各专家的输出，y的维度是 [batch_size * seq_len * num_experts_per_tok, hidden_dhidden_sizeim]
            y = torch.empty_like(x, dtype=x.dtype)
            for i, expert in enumerate(self.experts):
                # 专家开始处理
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
            # y的维度转为 [batch_size, seq_len, num_experts_per_tok, hidden_size]
            # 乘以权重topk_weight，再求和
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            # y的维度转为 [batch_size, seq_len, hidden_size]
            y = y.view(*orig_shape)
        else:
            # 推理阶段，代码手动实现了一个循环，根据索引将 Token 分发给对应的专家计算，然后再加权聚合回来。
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        # 处理共享专家
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        # 输出结果：共享专家输出 + Σ(路由专家输出 * 权重)
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        优化推理函数，高效地计算稀疏激活专家的输出（推理时不求梯度，且 Batch 可能较小），避免训练时那种对每个专家遍历所有 token 的低效方式。
        """
        # x的维度是 [batch_size * seq_len, hidden_size]
        # flat_expert_indices 维度是 [batch_size * seq_len * num_experts_per_tok]
        # flat_expert_weights 维度是 [batch_size * seq_len * num_experts_per_tok]
        # expert_cache 用于累加每个 token 的加权专家输出, 维度是 [batch_size * seq_len, hidden_size]
        expert_cache = torch.zeros_like(x)
        # 对专家索引排序
        idxs = flat_expert_indices.argsort()
        # 统计每个专家处理了多少 token
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 获得每个 token 所在的位置
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            # 获取第i个专家，以及这个专家处理的token 在 token_idxs 开始索引和结束索引
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            # 提取第i个专家以及这个专家要处理的x的部分片段
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            # 专家开始处理
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 专家处理完后加权
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 结果累计到 expert_cache里
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    """
    Decoder 第2个和第3个子层
    """
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        # Attention前的Norm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # FFN前的Norm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # FeedForward，可能是普通 FFN 或 MoE
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # x ──► LayerNorm ──► Self-Attention ──► Add ──► LayerNorm ──► FeedForward ──► Add ──► output
        #       (norm first)    ↑______Residual______↑     (norm first)   ↑____Residual____↑
        # 先临时保存，后面做残差
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            # 先做RMSNorm
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        # 做残差连接
        hidden_states += residual
        # 做Norm + FeedForward，再做残差连接
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """
    decoder的主体，包含了 Embedding → N×Block → Final RMSNorm
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        # 创建 self.num_hidden_layers 个 MiniMindBlock
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        # RMSNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算 RoPE（旋转位置编码）所需的 cos/sin 表
        # 维度是 [max_position_embeddings, hidden_size // num_attention_heads]
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        # input_ids ──► Embedding ──► Dropout  ──► MiniMindBlock 0  ──► MiniMindBlock 1  ──► ... ──► MiniMindBlock N  ──► Final RMSNorm ──► hidden_states(output)
        #                                               │                      │                         │
        #                                               ▼                      ▼                         ▼
        #                                             (K₀, V₀)              (K₁, V₁)         ...      (Kₙ, Vₙ)

        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        # input_ids → [batch_size, seq_len] → embed_tokens → [batch_size, seq_len, hidden_size] → dropout → 仍是 [batch_size, seq_len, hidden_size]
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 准备sin cos位置编码，根据传入的 start_pos 切片
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        # 逐层通过 Transformer Blocks
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        # 最终归一化
        hidden_states = self.norm(hidden_states)

        # MoE 辅助损失, 如果某层用了 MoE（混合专家），其 mlp 会计算一个 负载均衡辅助损失（auxiliary loss），要把所有层的 aux_loss 加起来，供训练时联合优化
        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )
        # 返回隐藏状态、当前的 (key, value) 缓存？ 和 MoE 辅助损失
        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    """
    完整的因果语言模型, MiniMindModel（Transformer 主干） + LM Head（输出层）
    """
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        # 线程层, 将隐藏状态映射回词表维度 (vocab_size)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 权重共享, 让 embedding 层 和 lm_head 层 共享同一组权重矩阵
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        # input_ids ──► [MiniMindModel] ──► hidden_states ──► [LM Head] ──► logits (vocab_size)
        #                ↑      ↑                ↑
        #                │      │                └── 可选：只取最后 k 个 token 的 logits
        #                │      └── 输出: past_key_values（用于下一次推理）
        #                └── 内部: Embedding → N×Block → Final RMSNorm

        # 调 MiniMindModel
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        # 动态截取hidden_states。不需要对整个序列做 lm_head。因为:
        # 在训练时，通常要计算 所有位置 的 loss（如 [x1,x2,x3] → 预测 [x2,x3,EOF]）
        # 在推理时，我们只关心 最后一个 token 的预测结果（因为只生成一个新词）
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        # 输出结果
        output = CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        output.aux_loss = aux_loss
        return output
