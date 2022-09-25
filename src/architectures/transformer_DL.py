# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Optional, Any, Union, Callable
from torch import nn
import torch
from torch import Tensor
from torch.nn  import functional as F
from torch.nn.modules import Module
from torch.nn import MultiheadAttention
from torch.nn.modules import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules import Dropout
from torch.nn.modules import Linear
from torch.nn.modules import LayerNorm

import math

#from torch.nn import Transformer
from utilities import *

from quantize import *

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))



###transformer model
class transformer_model(nn.Module):
    def __init__(self,
                 custom_decoder,
                 device,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 ):
        super(transformer_model, self).__init__()

        #####when src= None ,Transformer_TIM will handle the forward pass differently
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       custom_decoder=custom_decoder,
                                       device=device)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)
        self.emb_size=emb_size



    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        ####tgt has shape(T, bz), starting with token BOS

        ###convert to embedding
        if src!=None:
            src_emb = self.positional_encoding(self.src_tok_emb(src)).to(DEVICE)
        else:
            src_emb=None
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))

        if tgt_mask!=None:
            tgt_mask=tgt_mask.to(DEVICE)

        if tgt_padding_mask!=None:
            tgt_padding_mask=tgt_padding_mask.to(DEVICE)
        if src_padding_mask!=None:
            src_padding_mask=src_padding_mask.to(DEVICE)

        outs = self.transformer(src=src_emb, tgt=tgt_emb.to(DEVICE),
                                src_mask=src_mask,tgt_mask=tgt_mask,
                                tgt_key_padding_mask=tgt_padding_mask,
                                memory_key_padding_mask=src_padding_mask)

        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):

       
        ###convert to embedding
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))

        outs=self.transformer.decoder(tgt_emb, memory,
                            tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                            tgt_key_padding_mask= tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask)



        return outs




class transformerEncoderOnly_model(nn.Module):
    def __init__(self,
                 custom_encoder,
                 custom_decoder,
                 device,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 ):
        super(transformerEncoderOnly_model, self).__init__()

        #####when src= None ,Transformer_TIM will handle the forward pass differently
        self.transformer = TransformerEncoderOnly(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       custom_decoder=custom_decoder,
                                       custom_encoder=custom_encoder,
                                       device=device)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        #self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        ####tgt has shape(T, bz), starting with token BOS

        ###convert to embedding
        if src!=None:
            src_emb = self.positional_encoding(self.src_tok_emb(src)).to(DEVICE)
        else:
            src_emb=None
        #tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))

   
        outs = self.transformer(src=src_emb, tgt=None,
                                src_mask=src_mask,tgt_mask=tgt_mask,
                                tgt_key_padding_mask=tgt_padding_mask,
                                memory_key_padding_mask=src_padding_mask)

        return self.generator(outs.mean(0))

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):

       
        ###convert to embedding
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))

        outs=self.transformer.decoder(tgt_emb, memory,
                            tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                            tgt_key_padding_mask= tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask)



        return outs





###transformer model with factor embedding
class transformerFactor_model(nn.Module):
    def __init__(self,
                 custom_decoder,
                 device,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 tgt_vocab_size: int,
                 n_mechanisms:int,
                 StartPos:int,
                 EndPos:int,
                 src_vocab_size: int = 0,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 quantization=False
                 ):
        super(transformerFactor_model, self).__init__()

        #####when src= None ,Transformer_TIM will handle the forward pass differently
        self.quantization=quantization

        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       custom_decoder=custom_decoder,
                                       device=device,
                                       quantization=quantization)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        
        if src_vocab_size>0:
            self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size//2)##half the size,  because the other half will be factor embedding
        
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size//2)
        


        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

        self.n_mechanisms=n_mechanisms

        self.StartPos=StartPos

        self.EndPos=EndPos

        ###factor emebdding
        self.factor_emb = TokenEmbedding(n_mechanisms+1, emb_size//2) ###+1 because one embedding will be used for "not a factor"

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        ####tgt has shape(T, bz), starting with token BOS

        ###convert to embedding
        if src!=None:
            src_emb = self.positional_encoding(self.src_tok_emb(src)).to(DEVICE)
        else:
            src_emb=None

        tgt_factor_tokens=calFactorTokens(trg,StartPos=self.StartPos,EndPos=self.EndPos,n_mechanisms=self.n_mechanisms).to(DEVICE)

        tgt_token_emb=self.tgt_tok_emb(trg).to(DEVICE)
        factor_token_emb=self.factor_emb(tgt_factor_tokens).to(DEVICE)
        tgt_emb=torch.cat((tgt_token_emb,factor_token_emb),2).to(DEVICE)#concatenate the token embeddding and factor embedding

        tgt_emb = self.positional_encoding(tgt_emb).to(DEVICE)

        if self.quantization:
            outs,codebookloss = self.transformer(src=src_emb, tgt=tgt_emb.to(DEVICE),
                                    src_mask=src_mask,tgt_mask=tgt_mask.to(DEVICE),
                                    tgt_key_padding_mask=tgt_padding_mask.to(DEVICE),
                                    memory_key_padding_mask=src_padding_mask)

        else:
            outs = self.transformer(src=src_emb, tgt=tgt_emb.to(DEVICE),
                                    src_mask=src_mask,tgt_mask=tgt_mask.to(DEVICE),
                                    tgt_key_padding_mask=tgt_padding_mask.to(DEVICE),
                                    memory_key_padding_mask=src_padding_mask)

        
        if self.quantization:
            return self.generator(outs),codebookloss
        else:
            return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):

       
        ###convert to embedding
        tgt_factor_tokens=calFactorTokens(tgt,StartPos=self.StartPos,EndPos=self.EndPos,n_mechanisms=self.n_mechanisms).to(DEVICE)

        tgt_token_emb=self.tgt_tok_emb(tgt).to(DEVICE)
        factor_token_emb=self.factor_emb(tgt_factor_tokens).to(DEVICE)
        tgt_emb=torch.cat((tgt_token_emb,factor_token_emb),2).to(DEVICE)#concatenate the token embeddding and factor embedding

        tgt_emb = self.positional_encoding(tgt_emb).to(DEVICE)

        if self.quantization:
            outs,codebookloss = self.transformer(src=memory, tgt=tgt_emb.to(DEVICE),
                                    src_mask= memory_mask,tgt_mask=tgt_mask.to(DEVICE),
                                    tgt_key_padding_mask=tgt_key_padding_mask.to(DEVICE),
                                    memory_key_padding_mask= memory_key_padding_mask)

        else:
            outs = self.transformer(src=memory, tgt=tgt_emb.to(DEVICE),
                                    src_mask=memory_mask,tgt_mask=tgt_mask.to(DEVICE),
                                    tgt_key_padding_mask=tgt_key_padding_mask.to(DEVICE),
                                    memory_key_padding_mask= memory_key_padding_mask)



        return outs




########function to add factor embeding onto the model


def calFactorTokens(InputTokens,StartPos,EndPos,n_mechanisms):
    ####convert toekns to factor index for using the factor emebedding 
    FactorTokens=torch.zeros(InputTokens.size())
    # print("FactorTokens")
    # print(FactorTokens.shape)
    ######Inputtoekn size (T, Bz)
    if FactorTokens.shape[0]>=(StartPos+1):##during decoding the seq into may be very short
        for i in range(StartPos,min(EndPos,FactorTokens.shape[0])):
            FactorTokens[i,:]=((i-StartPos)%n_mechanisms)+1
    return FactorTokens
















####modified version of transformer by DL


class Transformer(Module):
    r"""A transformer model. User is able to modify the attributes as needed. The architecture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
    model with corresponding parameters.
    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, encoder and decoder layers will perform LayerNorms before
            other attention and feedforward operations, otherwise after. Default: ``False`` (after).
    Examples::
        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)
    Note: A full example to apply nn.Transformer module for the word language model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None,quantization=False) -> None:
    
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Transformer, self).__init__()

        self.quantization=quantization

        if num_encoder_layers!=0:###0 when decoder only
            if custom_encoder is not None:
                self.encoder = custom_encoder
            else:
                encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                        activation, layer_norm_eps, batch_first, norm_first,
                                                        **factory_kwargs)
                encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
                self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first,
                                                    **factory_kwargs)
            decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Take in and process masked source/target sequences.
        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).
        Shape:
            - src: :math:`(S, N, E)`, `(N, S, E)` if batch_first.
            - tgt: :math:`(T, N, E)`, `(N, T, E)` if batch_first.
            - src_mask: :math:`(S, S)`.
            - tgt_mask: :math:`(T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.
            Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
            [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a ByteTensor is provided, the non-zero positions will be ignored while the zero
            positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - output: :math:`(T, N, E)`, `(N, T, E)` if batch_first.
            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.
            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number
        Examples:
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """
        if src!=None:

    

            if not self.batch_first and src.size(1) != tgt.size(1):
                raise RuntimeError("the batch number of src and tgt must be equal")
            elif self.batch_first and src.size(0) != tgt.size(0):
                raise RuntimeError("the batch number of src and tgt must be equal")

            if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
                raise RuntimeError("the feature number of src and tgt must be equal to d_model")

            memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        else:
            memory=None

        if self.quantization==True:
            output,codebookloss = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)           

        
            return output, codebookloss
        else:   
            output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
            return output

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)






class TransformerEncoderOnly(Module):
    r"""
    transformer that only has encoder part plus a classider part for downstream task
    A transformer model. User is able to modify the attributes as needed. The architecture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
    model with corresponding parameters.
    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, encoder and decoder layers will perform LayerNorms before
            other attention and feedforward operations, otherwise after. Default: ``False`` (after).
    Examples::
        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)
    Note: A full example to apply nn.Transformer module for the word language model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
    
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderOnly, self).__init__()

        if num_encoder_layers!=0:###0 when decoder only
            if custom_encoder is not None:
                self.encoder = custom_encoder
            else:
                encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                        activation, layer_norm_eps, batch_first, norm_first,
                                                        **factory_kwargs)
                encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
                self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # if custom_decoder is not None:
        #     self.decoder = custom_decoder
        # else:
        #     decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
        #                                             activation, layer_norm_eps, batch_first, norm_first,
        #                                             **factory_kwargs)
        #     decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        #     self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Take in and process masked source/target sequences.
        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).
        Shape:
            - src: :math:`(S, N, E)`, `(N, S, E)` if batch_first.
            - tgt: :math:`(T, N, E)`, `(N, T, E)` if batch_first.
            - src_mask: :math:`(S, S)`.
            - tgt_mask: :math:`(T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.
            Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
            [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a ByteTensor is provided, the non-zero positions will be ignored while the zero
            positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - output: :math:`(T, N, E)`, `(N, T, E)` if batch_first.
            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.
            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number
        Examples:
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """
        if src!=None :

    
            if tgt!=None:
                if not self.batch_first and src.size(1) != tgt.size(1):
                    raise RuntimeError("the batch number of src and tgt must be equal")
                elif self.batch_first and src.size(0) != tgt.size(0):
                    raise RuntimeError("the batch number of src and tgt must be equal")

                if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
                    raise RuntimeError("the feature number of src and tgt must be equal to d_model")

            memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        else:
            memory=None

        # output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
        #                       tgt_key_padding_mask=tgt_key_padding_mask,
        #                       memory_key_padding_mask=memory_key_padding_mask)
        
        return memory

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)















####orgiinal encoder and encoder layer 


class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


##########decoder layer 

class TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectivaly. Otherwise it's done after.
            Default: ``False`` (after).
    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None ,quantization=False,n_quan_head=4,
                 transformer_quant_codebook_size=512) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

        
        self.quantization=quantization
        if quantization==True:
            print("conducting quantization")
            self.quantize = Quantize(num_hiddens=d_model,
                         n_embed=transformer_quant_codebook_size, 
                          groups=n_quan_head).to(device)



    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            if self.quantization:
                q_attention,codebookloss,_=self.quantize(self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask))
                x = x + q_attention
            else: 
                x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)          
            
            if memory!=None:
                x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            
            if self.quantization:
                q_attention,codebookloss,_=self.quantize(self._sa_block(x, tgt_mask, tgt_key_padding_mask))
                x = self.norm1(x + q_attention)
            else: 
                x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            
            if memory!=None:
                x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        if self.quantization:
            return x,codebookloss
        else:
            return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]

        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

###transformer decoder


class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers
    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None,quantization=False):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.quantization=quantization

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.
        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = tgt
        codebookloss=0
        for i in range(len(self.layers)):
            mod=self.layers[i]
            
            if self.quantization and i>1 and i<(len(self.layers)-1): #the first 2 and last layers are not quantized
                mod.quantization=True
                output,cbloss = mod(output, memory, tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask)
                codebookloss+=cbloss

            else:
                mod.quantization=False
                output = mod(output, memory, tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)


        if self.quantization:
            # print("codebookloss")
            # print(codebookloss)
            return output, codebookloss
        else:
            return output





# ################modified version of transformer to accomodate slot-specific TIM


# class TIMDecoder(nn.Module):
#     """TransformerDecoder is a stack of N decoder layers
#     Args:
#         decoder_layer: an instance of the TransformerDecoderLayer() class (required).
#         num_layers: the number of sub-decoder-layers in the decoder (required).
#         norm: the layer normalization component (optional).
#     Examples::
#         >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
#         >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
#         >>> memory = torch.rand(10, 32, 512)
#         >>> tgt = torch.rand(20, 32, 512)
#         >>> out = transformer_decoder(tgt, memory)
#     """
#     __constants__ = ['norm']

#     def __init__(self, decoder_layer, num_layers, norm=None):
#         super(TIMDecoder, self).__init__()
        
#         #self.layers = _get_clones(decoder_layer, num_layers)

#         layer_lst = []

#         for j in range(0,num_layers):
#             layer_lst.append(decoder_layer)

#         self.layers = nn.ModuleList(layer_lst)

#         self.num_layers = num_layers
#         self.norm = norm

#     def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
#                 memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
#                 memory_key_padding_mask: Optional[Tensor] = None,
#                 tgt_mechanism_mask: Optional[Tensor] = None,
#                 tgt_mechanism_key_padding_masks: Optional[Tensor] = None) -> Tensor:
#         """Pass the inputs (and mask) through the decoder layer in turn.
#         Args:
#             tgt: the sequence to the decoder (required).
#             memory: the sequence from the last layer of the encoder (required).
#             tgt_mask: the mask for the tgt sequence (optional).
#             memory_mask: the mask for the memory sequence (optional).
#             tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
#             memory_key_padding_mask: the mask for the memory keys per batch (optional).
#         Shape:
#             see the docs in Transformer class.
#         """
#         output = tgt

#         for mod in self.layers:
#             output = mod(output, memory, tgt_mask=tgt_mask,
#                          memory_mask=memory_mask,
#                          tgt_key_padding_mask=tgt_key_padding_mask,
#                          memory_key_padding_mask=memory_key_padding_mask,
#                          tgt_mechanism_mask = tgt_mechanism_mask,
#                          tgt_mechanism_key_padding_masks=tgt_mechanism_key_padding_masks)

#         if self.norm is not None:
#             output = self.norm(output)

#         return output
