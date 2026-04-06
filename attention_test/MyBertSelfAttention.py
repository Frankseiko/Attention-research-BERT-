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

class MyBertSelfAttention(BertSelfAttention):
    def __init__(self,config,domain_token_ids=None, boost_init=2.0, learnable_boost=True):
        super().__init__(config)
        self.domain_token_ids = domain_token_ids or []
    
        if learnable_boost:
            inv_sp = math.log(math.exp(boost_init)-1.0) if boost_init > 0 else -5.0
            self.boost_param = nn.Parameter(torch.tensor(inv_sp,dtype= torch.float32))
        else:
            self.register_buffer("boost_const",torch.tensor(float(boost_init),dtype= torch.float32))
            self.boost_param = None
    def _current_boost(self, dtype, device):
        if self.boost_param is not None:
            boost = F.softplus(self.boost_param)
        else:
            boost = self.boost_const
        return boost.to(dtype=dtype, device=device)

    def forward(self, *args, **kwargs):
        # —— 安全地从 args/kwargs 里取出关心的字段 —— #
        # args[0]  必定是 hidden_states
        hidden_states          = args[0]
        attention_mask         = kwargs.get("attention_mask", None)
        head_mask              = kwargs.get("head_mask", None)
        encoder_hidden_states  = kwargs.get("encoder_hidden_states", None)
        encoder_attention_mask = kwargs.get("encoder_attention_mask", None)
        past_key_values        = kwargs.get("past_key_values", None)
        output_attentions      = kwargs.get("output_attentions", False)
        input_ids              = kwargs.get("input_ids", None)

        #注意力计算 qkv
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer   = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer   = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2)
        ) / math.sqrt(self.attention_head_size)



        #权重提升的（domain-token）
        if input_ids is not None and len(self.domain_token_ids) > 0:
            # 创建 mask: shape = [batch, seq_len]
            #is_domain_token = torch.zeros_like(input_ids,dtype=torch.bool)
            #for token_id in self.domain_token_ids:
            #    is_domain_token = is_domain_token | (input_ids == token_id)
            
            ####自学习参数
            domain_ids = torch.tensor(self.domain_token_ids,device=input_ids.device,dtype = input_ids.dtype)
            is_domain_token = torch.isin(input_ids,domain_ids)
            domain_mask = is_domain_token.unsqueeze(1).unsqueeze(2).to(attention_scores.dtype)
            boost = self._current_boost(attention_scores.dtype, attention_scores.device)
            attention_scores = attention_scores + domain_mask*boost
            
            
            ####手动设置domain_factor参数
            ###加法
            # 扩展成 attention shape: [batch, 1, 1, seq_len]
            #domain_boost = is_domain_token.unsqueeze(1).unsqueeze(2).float() * self.boost_factor
            # 加入到 attention_scores（加法形式）
            #attention_scores = attention_scores + domain_boost
            ###乘法
            # mask = is_domain_token.unsqueeze(1).unsqueeze(2).float()
            # domain_factor =  1.0 + mask*(self.boost_factor-1.0)
            # attention_scores = attention_scores * domain_factor
        
            
            
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(attention_scores.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask
             
        attention_probs = nn.Softmax(dim = -1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask 
        context_layer = torch.matmul(attention_probs,value_layer)
        context_layer = context_layer.permute(0,2,1,3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # 计算完 context_layer 之后
        outputs = (context_layer, )
        if output_attentions:
            outputs = outputs + (attention_probs,)
        return outputs
    
    
    
from torch import nn
from transformers.models.bert.modeling_bert import BertLayer

class MyBertLayer(BertLayer):
    def __init__(self, config, domain_token_ids, boost_init = 1.0, learnable_boost=True):
        super().__init__(config)
        # SelfAttention
        self.attention.self = MyBertSelfAttention(
            config, domain_token_ids=domain_token_ids, boost_init=boost_init, learnable_boost=learnable_boost
        )

 
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        input_ids=None,
        **kwargs,                # 捕获所有未显式声明的额外参数
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
    def __init__(self, config, domain_token_ids, boost_init = 1.0,learnable_boost = True):
        super().__init__(config)
        # 用自定义的 MyBertLayer 完全替换所有层
        self.layer = nn.ModuleList(
            [MyBertLayer(config, domain_token_ids, boost_init=boost_init,learnable_boost=learnable_boost)
             for _ in range(config.num_hidden_layers)]
        )

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
    def __init__(self, config, domain_token_ids, boost_init = 1.0,learnable_boost= True):
        super().__init__(config)
        self.encoder = MyBertEncoder(config, domain_token_ids, boost_init=boost_init,learnable_boost=learnable_boost)

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
            embeddings = self.embeddings(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids
            )
        elif inputs_embeds is not None:
            embeddings = inputs_embeds
        else:
            raise ValueError("必须提供 input_ids 或 inputs_embeds")

        # 2) 调用自定义 encoder，并显式透传 input_ids
        encoder_outputs = self.encoder(
            embeddings,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            input_ids=input_ids,  # ★ 关键：透传 input_ids
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

        
        
class MyBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, domain_token_ids, boost_init = 1.0,learnable_boost= True):
        super().__init__(config)
        self.bert = MyBertModel(config, domain_token_ids, boost_init=boost_init,learnable_boost=True)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, output_attentions=True, **kwargs):
        # 关键：传入 input_ids 到 Attention 层
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions= output_attentions,
            **kwargs
        )
        pooled_output = outputs[1]  # [CLS]

        logits = self.classifier(self.dropout(pooled_output))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, 
                "logits": logits,
                } if loss is not None else {"logits": logits}