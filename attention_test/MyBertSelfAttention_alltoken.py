import torch
import torch.nn as nn
import math
from transformers.models.bert.modeling_bert import(
    BertLayer,
    BertEncoder,
    BertModel,
    BertForSequenceClassification,
    BertSelfAttention,
)
import torch.nn.functional as F

def inv_softplus(y):
    return math.log(math.exp(float(y))-1.0) if y>0 else -5.0


class MyBertSelfAttention(BertSelfAttention):
    def __init__(self, config, per_head=False, nonnegative=True):
        super().__init__(config)
        self.per_head = per_head
        self.nonnegative = nonnegative  # True→softplus；False→tanh/clamp
        #self.bias_scale = nn.Parameter(torch.tensor(0,0)) (可学习缩放)
    def forward(self, *args, **kwargs):
        hidden_states = args[0]
        attention_mask         = kwargs.get("attention_mask", None)  # 建议传加性mask
        head_mask              = kwargs.get("head_mask", None)
        encoder_hidden_states  = kwargs.get("encoder_hidden_states", None)
        encoder_attention_mask = kwargs.get("encoder_attention_mask", None)
        past_key_value         = kwargs.get("past_key_value", None)
        output_attentions      = kwargs.get("output_attentions", False)
        
        key_boost_table        = kwargs.get("key_boost_table",None)
        input_ids              = kwargs.get("input_ids", None)

        
        
        # 常规 q/k/v（省略 cross-attn 分支可照你之前补齐）
        mixed_query_layer = self.query(hidden_states)
        key_layer   = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)

        # === token-wise bias ===
        if input_ids is not None and key_boost_table is not None:
            # table(input_ids): [B, L, 1] 或 [B, L, H]
            raw = key_boost_table(input_ids)
            bias = F.softplus(raw) if self.nonnegative else torch.tanh(raw)
            if not self.per_head:
                #[B,L,1] to [B,1,1,L]
                bias = bias.squeeze(-1).unsqueeze(1).unsqueeze(2)
            else:
                #[B,L,H] to [B,H,1,L]
                bias = bias.permute(0,2,1).unsqueeze(2)
            
            attention_scores = attention_scores + bias.to(attention_scores.dtype)

        if attention_mask is not None:
    # 期望 [B,1,1,L]
            attention_scores = attention_scores + attention_mask.to(attention_scores.dtype)  # 加性mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(context_layer.size()[:-2] + (self.all_head_size,))
        return (context_layer,) + ((attention_probs,) if output_attentions else ())

    
    
    
from torch import nn
from transformers.models.bert.modeling_bert import BertLayer

class MyBertLayer(BertLayer):
    def __init__(self, config,        
                 per_head=False,
                 nonnegative=True):
        super().__init__(config)
        self.attention.self = MyBertSelfAttention(
            config,  
            per_head=per_head,
            nonnegative=nonnegative,
        )
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        key_boost_table=None,
        output_attentions=False,
        input_ids=None,
        **kwargs                # 捕获所有未显式声明的额外参数
    ):
        # 把 input_ids 也传给 attention
        self_attn_outputs = self.attention.self(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            input_ids=input_ids,   # 这里
            key_boost_table=key_boost_table
        )
        attention_output = self.attention.output(
            self_attn_outputs[0], hidden_states 
        )
        # 其余部分和父类一致
        
        
        # 其余部分复用父类中间层 & 输出层
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return (layer_output,) + ((self_attn_outputs[1],) if output_attentions else ())

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Union

from transformers.models.bert.modeling_bert import BertEncoder, BertLayer
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

class MyBertEncoder(BertEncoder):
    def __init__(self, config,
                 per_head=False,
                 nonnegative=True):
        super().__init__(config)
        self.layer = nn.ModuleList([
            MyBertLayer(config, per_head, nonnegative)
            for _ in range(config.num_hidden_layers)
        ])
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        input_ids: Optional[torch.LongTensor] = None,   # ★ 透传
        key_boost_table= None
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions    = () if output_attentions else None
        next_key_values   = () if use_cache else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i] if head_mask is not None else None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=(
                    past_key_values[i] if past_key_values is not None else None
                ),
                output_attentions=output_attentions,
                input_ids=input_ids,
                key_boost_table= key_boost_table,
            )
            # layer_outputs: (layer_output, attentions?)
            hidden_states = layer_outputs[0]

            if use_cache:
                # 如果你的 MyBertSelfAttention 不返回新的 key/value，保持 None
                next_key_values += (None,)

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            outputs = (hidden_states, next_key_values)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            if output_attentions:
                outputs += (all_attentions,)
            return outputs

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=None,
        )




class MyBertModel(BertModel):
    def __init__(self, config,
                 domain_token_ids=None,         # 为兼容旧调用，保留但无实际用途
                 boost_init=0.0,
                 learnable_boost=True,
                 per_head=False,
                 nonnegative=True):
        super().__init__(config)

        # 1) 共享 token→bias 查表
        if learnable_boost:
            dim = config.num_attention_heads if per_head else 1
            self.key_boost_table = nn.Embedding(config.vocab_size, dim)
            # softplus^-1(init) 使得 softplus(raw)≈boost_init
            inv_sp = math.log(math.exp(float(boost_init)) - 1.0) if boost_init > 0 else -5.0
            nn.init.constant_(self.key_boost_table.weight, inv_sp)
        else:
            self.register_buffer("key_boost_table", None)

        # 2) 自定义 encoder，传“共享表”的引用（✅ 关键改动）
        self.encoder = MyBertEncoder(
            config,
            per_head=per_head,
            nonnegative=nonnegative,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # 1) 计算 embeddings
        if input_ids is not None:
            input_shape = input_ids.size()
            embeddings = self.embeddings(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids
            )
        elif inputs_embeds is not None:
            embeddings = inputs_embeds
            input_shape = embeddings.size()[:-1]
        else:
            raise ValueError("必须提供 input_ids 或 inputs_embeds")

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device= embeddings.device)
        else:
            if attention_mask.dtype != torch.float32 and attention_mask.dtype != torch.float16:
                attention_mask = attention_mask.to(dtype=embeddings.dtype)
            if attention_mask.device != embeddings.device:
                attention_mask = attention_mask.to(embeddings.device)
        
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, device=embeddings.device
            # embeddings.dtype
        )
        
        if head_mask is not None:
            head_mask = self.get_head_mask(head_mask,self.config.num_hidden_layers)
        
        # 2) 调用自定义 encoder，并显式透传 input_ids
        encoder_outputs = self.encoder(
            embeddings,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            input_ids=input_ids,  # ★ 关键：透传 input_ids
            key_boost_table= self.key_boost_table,
        )
        from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

        # 3) 获取输出并构造返回值
        if return_dict:
            sequence_output = encoder_outputs.last_hidden_state
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
            return BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                past_key_values=encoder_outputs.past_key_values,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
                cross_attentions=None,
            )
        else:
            outputs = (encoder_outputs[0], encoder_outputs[1])  # (sequence_output, pooled_output)
            extra = ()
            if output_hidden_states:
                extra += (encoder_outputs.hidden_states,)
            if output_attentions:
                extra += (encoder_outputs.attentions,)
            return outputs + extra

        
        
from transformers.modeling_outputs import SequenceClassifierOutput

class MyBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self,
                 config,
                 domain_token_ids=None,
                 boost_init=0.0,
                 learnable_boost=True,
                 per_head=False,          # ✅ 新增
                 nonnegative=True,        # ✅ 新增
                 **kwargs):               # 预留扩展
        super().__init__(config)
        self.bert = MyBertModel(
            config,
            domain_token_ids=domain_token_ids,
            boost_init=boost_init,
            learnable_boost=learnable_boost,
            per_head=per_head,           # ✅ 透传
            nonnegative=nonnegative,     # ✅ 透传
            **kwargs
        )

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs
        )
        pooled_output = outputs.pooler_output
        logits = self.classifier(self.dropout(pooled_output))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            out = (logits,)
            if output_hidden_states:
                out += (outputs.hidden_states,)
            if output_attentions:
                out += (outputs.attentions,)
            if loss is not None:
                out = (loss,) + out
            return out

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )
