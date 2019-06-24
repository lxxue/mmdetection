import torch
from torch import nn
import torchvision
from torch.nn.utils.rnn import pack_padded_sequence

import os
import json
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CapDecoder(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim, dropout=0.5, feats_size=13, alpha_c=1):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        # lixin
        :param feats_size: size of encoded images
        """
        super(CapDecoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution
        # lixin
        self.avgpool = nn.AdaptiveAvgPool2d((feats_size, feats_size))
        self.feats_size = feats_size
        self.loss_cap = nn.CrossEntropyLoss()
        self.alpha_c = alpha_c

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        # lixin
        encoder_out = self.avgpool(encoder_out)
        encoder_out = encoder_out.permute(0, 2, 3, 1)

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        # caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()


        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

    def loss(self, scores, caps_sorted, decode_lengths, alphas):
        losses = {}
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        # decode_lengths = decode_lengths.tolist()
        targets = caps_sorted[:, 1:]


        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        # print(scores.data.size())
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        # print(targets.data.size())



        losses['loss_cap'] = 0.001 * self.loss_cap(scores.data, targets.data) + self.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        return losses


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha



if __name__ == "__main__":
    # Data parameters
    data_folder = '/data/home/v-lixxue/a-PyTorch-Tutorial-to-Image-Captioning/data'  # folder with data files saved by create_input_files.py
    data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

    # Model parameters
    embed_dim = 512  # dimension of word embeddings
    attention_dim = 512  # dimension of attention linear layers
    decoder_dim = 512  # dimension of decoder RNN
    dropout = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

    # Training parameters
    start_epoch = 0
    epochs = 120  # number of epochs to train for (if early stopping is not triggered)
    epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
    batch_size = 2
    workers = 1  # for data-loading; right now, only 1 works with h5py
    encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
    decoder_lr = 4e-4  # learning rate for decoder
    grad_clip = 5.  # clip gradients at an absolute value of
    alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
    best_bleu4 = 0.  # BLEU-4 score right now
    print_freq = 100  # print training/validation stats every __ batches
    fine_tune_encoder = False  # fine-tune encoder?
    checkpoint = None  # path to checkpoint, None if none

    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    cap_decoder = CapDecoder(attention_dim=attention_dim,
                             embed_dim=embed_dim,
                             decoder_dim=decoder_dim,
                             vocab_size=len(word_map))

    with open(os.path.join(data_folder, 'TRAIN' + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
        captions = json.load(j)

    with open(os.path.join(data_folder, 'TRAIN' + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            caplens = json.load(j)

    feats = torch.from_numpy(np.random.randn(2, 512, 40, 40).astype(np.float32))
    scores, caps_sorted, decode_lengths, alphas, sort_ind = cap_decoder(feats,
                                                                        torch.LongTensor(captions[0:2]),
                                                                        torch.LongTensor([[caplen] for caplen in caplens[0:2]]))

    print(scores, caps_sorted, decode_lengths, alphas, sort_ind)