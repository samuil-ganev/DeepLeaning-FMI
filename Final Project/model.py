#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2023/2024
#############################################################################
###
### Невронен машинен превод
###
#############################################################################

import torch
import bpe
from parameters import *

### Positional Encoding ###

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, max_len = 5000):
        super().__init__()

        pe = torch.zeros(1, max_len, d_model)

        position = torch.arange(max_len).unsqueeze(0).unsqueeze(2)
        div_term = (10000.0 ** (torch.arange(0, d_model, 2) / d_model)).unsqueeze(0).unsqueeze(0)
        pe[0, :, 0::2] = torch.sin(position / div_term)
        pe[0, :, 1::2] = torch.cos(position / div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]  # x.shape = (batch_size, seq_len, embedding_dim)
        return x

### Multi Head Attention Cell ###

class MultiHeadAttn(torch.nn.Module):

    def __init__(self, n_head, d_model, d_keys, d_values):
        super().__init__()

        self.n_head, self.d_model, self.d_keys, self.d_values = n_head, d_model, d_keys, d_values
        self.scale = 1 / (d_keys ** 0.5)

        self.Wq_net = torch.nn.Linear(d_model, n_head * d_keys, bias=False)
        self.Wk_net = torch.nn.Linear(d_model, n_head * d_keys, bias=False)
        self.Wv_net = torch.nn.Linear(d_model, n_head * d_values, bias=False)
        self.Wo_net = torch.nn.Linear(n_head * d_values, d_model, bias=False)

        torch.nn.init.xavier_uniform_(self.Wq_net.weight)
        torch.nn.init.xavier_uniform_(self.Wk_net.weight)
        torch.nn.init.xavier_uniform_(self.Wv_net.weight)
        torch.nn.init.xavier_uniform_(self.Wo_net.weight)

    def forward(self, input, output = None, padding_mask = None, mask = None):

        # input.shape = (batch_size, seq_len_inp, d_model)
        # output.shape = (batch_size, seq_len_out, d_model)
        batch_size = input.shape[0]
        seq_len_inp = input.shape[1]
        seq_len_out = input.shape[1]

        if output is not None: seq_len_out = output.shape[1]

        if output is None: output = input

        head_q = self.Wq_net(output)
        # head_q.shape = (batch_size, seq_len_out, n_head * d_keys)
        head_k = self.Wk_net(input)
        # head_k.shape = (batch_size, seq_len_inp, n_head * d_keys)
        head_v = self.Wv_net(input)
        # head_v.shape = (batch_size, seq_len_inp, n_head * d_values)

        q = head_q.view(batch_size, seq_len_out, self.n_head, self.d_keys).transpose(1, 2)
        # q.shape = (batch_size, n_head, seq_len_out, d_keys)
        k = head_k.view(batch_size, seq_len_inp, self.n_head, self.d_keys).permute(0, 2, 3, 1)
        # k.shape = (batch_size, n_head, d_keys, seq_len_inp)
        v = head_v.view(batch_size, seq_len_inp, self.n_head, self.d_values).transpose(1, 2)
        # v.shape = (batch_size, n_head, seq_len_inp, d_values)

        attn_mat = torch.matmul(q, k)
        # attn_mat.shape = (batch_size, n_head, seq_len_out, seq_len_inp)

        if mask is not None:
            attn_mat = attn_mat.masked_fill(mask.unsqueeze(0).unsqueeze(1), -float('inf'))

        if padding_mask is not None:
            attn_mat = (attn_mat + padding_mask + padding_mask.transpose(2, 3)) * self.scale
        else:
            attn_mat *= self.scale

        attn_mat = torch.nn.functional.softmax(attn_mat, dim=-1)
        attn_mat = torch.matmul(attn_mat, v)
        # attn_mat.shape = (batch_size, n_head, seq_len_out, d_values)

        attn_mat = attn_mat.transpose(1, 2).flatten(2, 3)
        # attn_mat.shape = (batch_size, seq_len_out, n_head * d_values)

        attn_mat = self.Wo_net(attn_mat)
        # attn_mat.shape = (batch_size, seq_len_out, d_model)

        return attn_mat

### Transformer Encoder Cell ###

class TransformerEncoderCell(torch.nn.Module):

    def __init__(self, n_head, d_model, d_keys, d_values, d_ff, dropout):
        super().__init__()

        self.MHA = MultiHeadAttn(n_head, d_model, d_keys, d_values)
        self.layer_norm_1 = torch.nn.LayerNorm(d_model)
        self.dropout_1 = torch.nn.Dropout(encoder_dropout)
        self.W1 = torch.nn.Linear(d_model, d_ff)
        self.W2 = torch.nn.Linear(d_ff, d_model)
        self.layer_norm_2 = torch.nn.LayerNorm(d_model)
        self.dropout_2 = torch.nn.Dropout(encoder_dropout)

        torch.nn.init.xavier_uniform_(self.W1.weight)
        torch.nn.init.xavier_uniform_(self.W2.weight)

    def forward(self, input_eng, padding_mask = None):

        # input_eng.shape = (batch_size, seq_len_inp, d_model)

        if norm_first:
            y = input_eng + self.MHA(self.layer_norm_1(input_eng), padding_mask = padding_mask)
            y = y + self.W2(torch.nn.functional.relu(self.W1(self.layer_norm_2(y))))
        else:
            z1 = self.MHA(input_eng, padding_mask = padding_mask)
            # z1.shape = (batch_size, seq_len_inp, d_model)
            z2 = self.layer_norm_1(input_eng + self.dropout_1(z1))
            # z2.shape = (batch_size, seq_len_inp, d_model)
            z3 = self.W2(torch.nn.functional.relu(self.W1(z2)))
            # z3.shape = (batch_size, seq_len_inp, d_model)
            y = self.layer_norm_2(z2 + self.dropout_2(z3))
            # y.shape = (batch_size, seq_len_inp, d_model)

        return y

### Transformer Decoder Cell ###

class TransformerDecoderCell(torch.nn.Module):

    def __init__(self, n_head, d_model, d_keys, d_values, d_ff, dropout):
        super().__init__()

        self.MHA1 = MultiHeadAttn(n_head, d_model, d_keys, d_values)
        self.layer_norm_1 = torch.nn.LayerNorm(d_model)
        self.dropout_1 = torch.nn.Dropout(decoder_dropout)

        self.MHA2 = MultiHeadAttn(n_head, d_model, d_keys, d_values)
        self.layer_norm_2 = torch.nn.LayerNorm(d_model)
        self.dropout_2 = torch.nn.Dropout(decoder_dropout)
        self.W1 = torch.nn.Linear(d_model, d_ff)
        self.W2 = torch.nn.Linear(d_ff, d_model)
        self.layer_norm_3 = torch.nn.LayerNorm(d_model)
        self.dropout_3 = torch.nn.Dropout(decoder_dropout)

        torch.nn.init.xavier_uniform_(self.W1.weight)
        torch.nn.init.xavier_uniform_(self.W2.weight)

    def forward(self, encoder_output_eng, input_bg, src_padding_mask = None, trt_padding_mask = None, mask = None):

        # input_eng.shape = (batch_size, seq_len_inp, d_model)
        # input_bg.shape = (batch_size, seq_len_out, d_model)

        if norm_first:
            y = input_bg + self.MHA1(self.layer_norm_1(input_bg), padding_mask = trt_padding_mask, mask = mask)
            y = y + self.MHA2(encoder_output_eng, output = self.layer_norm_2(y))
            y = y + self.W2(torch.nn.functional.relu(self.W1(self.layer_norm_3(y))))
        else:
            z1 = self.MHA1(input_bg, padding_mask = trt_padding_mask, mask = mask)
            # z1.shape = (batch_size, seq_len_out, d_model)
            z2 = self.layer_norm_1(input_bg + self.dropout_1(z1))
            # z2.shape = (batch_size, seq_len_out, d_model)
            # if src_padding_mask is not None and trt_padding_mask is not None:
            #     encoder_output_eng = encoder_output_eng * ~src_padding_mask
            #     z2 = z2 * ~trt_padding_mask
            z3 = self.MHA2(encoder_output_eng, output = z2)
            # z3.shape = (batch_size, seq_len_out, d_model)
            z4 = self.layer_norm_2(z2 + self.dropout_2(z3))
            # z4.shape = (batch_size, seq_len_out, d_model)
            z5 = self.W2(torch.nn.functional.relu(self.W1(z4)))
            # z5.shape = (batch_size, seq_len_out, d_model)
            y = self.layer_norm_3(z4 + self.dropout_3(z5))
            # y.shape = (batch_size, seq_len_out, d_model)

        return y


class NMTmodel(torch.nn.Module):
    def preparePaddedBatch(self, source, pair2ind, lang = 'eng'):
        if lang == 'eng':
            unkTokenIdx = self.unkTokenIdx_eng
            padTokenIdx = self.padTokenIdx_eng
        else:
            unkTokenIdx = self.unkTokenIdx_bg
            padTokenIdx = self.padTokenIdx_bg
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents = [[pair2ind.get(pair, unkTokenIdx) for pair in s] for s in source]
        sents_padded = [ s+(m-len(s))*[padTokenIdx] for s in sents]
        return torch.tensor(sents_padded, dtype=torch.long, device=device)

    def save(self,fileName):
        torch.save(self.state_dict(), fileName)

    def load(self,fileName):
        self.load_state_dict(torch.load(fileName))

    def __init__(self, sourcePair2ind, targetPair2ind, sourcePath, targetPath, unkToken, padToken, num_layers, n_head, d_model, d_keys, d_values, d_ff, dropout, temperature, maxlen = 3000):
        super(NMTmodel, self).__init__()

        self.temperature = temperature

        self.targetInd2pair = {i : w for w, i in targetPair2ind.items()}

        self.pair2ind_eng = sourcePair2ind
        self.pair2ind_bg = targetPair2ind

        self.sourcePath = sourcePath
        self.targetPath = targetPath

        self.unkTokenIdx_eng = sourcePair2ind[unkToken]
        self.padTokenIdx_eng = sourcePair2ind[padToken]
        self.unkTokenIdx_bg = targetPair2ind[unkToken]
        self.padTokenIdx_bg = targetPair2ind[padToken]

        ### Model Parameters ###

        self.num_layers = num_layers

        self.embed_1 = torch.nn.Embedding(len(sourcePair2ind), d_model)
        self.embed_2 = torch.nn.Embedding(len(targetPair2ind), d_model)

        self.pos_embed_1 = PositionalEncoding(d_model)
        self.dropout_1 = torch.nn.Dropout(source_dropout)

        self.pos_embed_2 = PositionalEncoding(d_model)
        self.dropout_2 = torch.nn.Dropout(target_dropout)

        self.encoders = torch.nn.ModuleList([TransformerEncoderCell(n_head, d_model, d_keys, d_values, d_ff, dropout) for _ in range(num_layers)])
        self.decoders = torch.nn.ModuleList([TransformerDecoderCell(n_head, d_model, d_keys, d_values, d_ff, dropout) for _ in range(num_layers)])

        pos = torch.arange(maxlen)
        mask = pos.unsqueeze(0) > pos.unsqueeze(1)
        self.register_buffer('mask', mask)

        self.projection = torch.nn.Linear(d_model, len(targetPair2ind))

        torch.nn.init.xavier_uniform_(self.embed_1.weight)
        torch.nn.init.xavier_uniform_(self.embed_2.weight)
        torch.nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, source, target):

        source = self.preparePaddedBatch(source, self.pair2ind_eng)  # source.shape=(batch_size, seq_len_inp)
        seq_len_eng = source.shape[1]
        E1 = self.embed_1(source)  # E1.shape = (batch_size, seq_len_inp, d_model)
        input = self.dropout_1(self.pos_embed_1(E1))

        target = self.preparePaddedBatch(target, self.pair2ind_bg, lang = 'bg')  # target.shape=(batch_size, seq_len_out)
        seq_len_out = target.shape[1]
        E2 = self.embed_2(target[:, :-1])  # E2.shape = (batch_size, seq_len_out-1, d_model)
        output = self.dropout_2(self.pos_embed_2(E2))

        src_padding_mask = (source == self.padTokenIdx_eng).unsqueeze(1).unsqueeze(2).float() * (-1e9)
        trt_padding_mask = (target == self.padTokenIdx_bg).unsqueeze(1).unsqueeze(2)[:, :, :, :-1].float() * (-1e9)
        # scr_padding_mask.shape = (batch_size, 1, 1, seq_len_inp)
        # trt_padding_mask.shape = (batch_size, 1, 1, seq_len_out-1)

        batch_mask = self.mask[:seq_len_out-1, :seq_len_out-1]

        encoder_output = input # 0-level encoder

        for layer in range(self.num_layers):
            encoder_output = self.encoders[layer](encoder_output, src_padding_mask)

        decoder_output = output  # 0-level decoder

        for layer in range(self.num_layers):
            decoder_output = self.decoders[layer](encoder_output, decoder_output, src_padding_mask, trt_padding_mask, batch_mask)

        Z = self.projection(decoder_output.flatten(0, 1))
        # Z.shape = (batch_size * (seq_len_out-1), len(targetPair2ind))
        Y_bar = target[:, 1:].flatten(0, 1)
        # Y_bar.shape = (batch_size * (seq_len_out-1))
        H = torch.nn.functional.cross_entropy(Z, Y_bar, ignore_index=self.padTokenIdx_bg, label_smoothing=0.1)

        return H

    def translateSentence(self, sentence, limit=300):
        self.eval()

        startToken = '<S>'
        endToken = '</S>'

        device = next(self.parameters()).device

        output = torch.tensor([[self.pair2ind_bg[startToken]]], dtype=torch.long, device=device)
        sentence_translated = []

        with torch.no_grad():
            sentence = bpe.encode_corpus([sentence], self.sourcePath, 'eng', 0)
            source = self.preparePaddedBatch(sentence, self.pair2ind_eng)
            # source.shape = (1, seq_len_inp)
            E1 = self.embed_1(source)
            # E.shape = (1, seq_len_inp, d_model)
            input = self.pos_embed_1(E1)

            encoder_output = input  # 0-level encoder

            for layer in range(self.num_layers):
                encoder_output = self.encoders[layer](encoder_output)

            for _ in range(limit):

                if len(sentence_translated) != 0:
                    output = torch.tensor([self.pair2ind_bg[startToken]] + [self.pair2ind_bg[pair] for pair in sentence_translated], dtype=torch.long, device=device).unsqueeze(0)

                E2 = self.embed_2(output)
                output = self.dropout_2(self.pos_embed_2(E2))

                decoder_output = output  # 0-level decoder

                for layer in range(self.num_layers):
                    decoder_output = self.decoders[layer](encoder_output, decoder_output)

                Z = self.projection(decoder_output.flatten(0, 1))
                # Z.shape = (seq_len_out, len(targetPair2ind))
                prob = torch.nn.functional.softmax(Z / self.temperature, dim=1)[-1, :]
                # prob.shape = (len(targetPair2ind))
                next_pair_ind = torch.argmax(prob).item()

                if next_pair_ind == self.pair2ind_bg[endToken]:
                    break
                sentence_translated.append(self.targetInd2pair[next_pair_ind])

        sentence_translated = bpe.decode_corpus([sentence_translated], 'bg')
        print(sentence_translated[0])

        self.train()

        return sentence_translated[0]
