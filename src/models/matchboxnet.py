"""Класс TCSConv:
Реализует раздельную временную свертку (Temporal Convolution Separable, TCS). Это включает в себя глубинную свертку, за которой следует точечная свертка, что позволяет эффективно обучать модель, сокращая количество параметров по сравнению со стандартной сверткой.
Класс SubBlock:
Представляет субблок, используемый в модели MatchboxNet. Включает в себя TCS-сверточный слой, пакетную нормализацию и отсев для регуляризации. При наличии предоставленной остаточной связи добавляется ReLU активация.
Класс MainBlock:
Строит основной блок модели MatchboxNet, содержащий несколько экземпляров SubBlock. Включает остаточную связь, которая согласовывает размерности входных и выходных данных через точечную свертку и пакетную нормализацию.
Класс MatchboxNet:
Полная настройка модели MatchboxNet для классификации аудио. Состоит из начального сверточного слоя (пролог), нескольких модулей MainBlock (основа модели), за которыми следуют дополнительные сверточные слои и пакетные нормализации (эпилог). Завершается адаптивным усредняющим пулингом для сжатия карт признаков в одно значение на класс, за которым следует softmax для классификации.
Класс MFCC_MatchboxNet:
Расширенная версия MatchboxNet, которая сначала преобразует входные волновые формы в коэффициенты мел-частотных кепстральных коэффициентов (MFCC), обычное представление признаков для аудио. Затем эти признаки обрабатываются через модель MatchboxNet."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchaudio.transforms import MFCC

class TCSConv(nn.Module):
    """
    A module implementing a Temporal Convolution Separable (TCS) convolution.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The size of the convolutional kernel.

    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(TCSConv, self).__init__()

        self.depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size, groups=in_channels,
                                        padding='same')  # effectively performing a depthwise convolution
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels,
                                        kernel_size=1)  # effectively performing a pointwise convolution

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        return x


class SubBlock(nn.Module):
    """
    A sub-block module used in the MatchboxNet model.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The size of the convolutional kernel.

    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SubBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.tcs_conv = TCSConv(self.in_channels, self.out_channels, self.kernel_size)
        self.bnorm = nn.BatchNorm1d(self.out_channels)
        self.dropout = nn.Dropout()

    def forward(self, x, residual=None):
        x = self.tcs_conv(x)
        x = self.bnorm(x)

        # apply the residual if passed
        if residual is not None:
            x = x + residual

        x = F.relu(x)
        x = self.dropout(x)

        return x


class MainBlock(nn.Module):
    """
    A module implementing the main block of MatchboxNet.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The size of the convolutional kernel.
        R (int): The number of sub-blocks. Defaults to 1.
    """
    def __init__(self, in_channels, out_channels, kernel_size, R=1):
        super(MainBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.residual_pointwise = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1)
        self.residual_batchnorm = nn.BatchNorm1d(self.out_channels)

        self.sub_blocks = nn.ModuleList()

        # Initial sub-block. If this is MainBlock 1, our input will be 128 channels which may not necessarily == out_channels
        self.sub_blocks.append(
            SubBlock(self.in_channels, self.out_channels, self.kernel_size)
        )

        # Other sub-blocks. Output of all of these blocks will be the same
        for i in range(R - 1):
            self.sub_blocks.append(
                SubBlock(self.out_channels, self.out_channels, self.kernel_size)
            )

    def forward(self, x):
        residual = self.residual_pointwise(x)
        residual = self.residual_batchnorm(residual)

        for i, layer in enumerate(self.sub_blocks):
            if (i + 1) == len(self.sub_blocks):  # compute the residual in the final sub-block
                x = layer(x, residual)
            else:
                x = layer(x)

        return x


class MatchboxNet(nn.Module):
    """
    A MatchboxNet model implementation for audio classification.

    Args:
        B (int): The number of convolutional blocks in the model.
        R (int): The number of sub-blocks per block in the model.
        C (int): The number of channels in the input feature map.
        bins (int): The number of Mel frequency bins in the input spectrogram.
        kernel_sizes (list): A list of kernel sizes to use for the sub-blocks in the model.
        NUM_CLASSES (int): The number of output classes for the model.

    """
    def __init__(self, B, R, C, bins=64, kernel_sizes=None, NUM_CLASSES=30):
        super(MatchboxNet, self).__init__()
        if not kernel_sizes:
            kernel_sizes = [k * 2 + 11 for k in range(1, 5 + 1)]  # incrementing kernel size by 2 starting at 13

        # the prologue layers
        self.prologue_conv1 = nn.Conv1d(bins, 128, kernel_size=11, stride=2)
        self.prologue_bnorm1 = nn.BatchNorm1d(128)

        # the intermediate blocks
        self.blocks = nn.ModuleList()

        self.blocks.append(
            MainBlock(128, C, kernel_sizes[0], R=R)
        )

        for i in range(1, B):
            self.blocks.append(
                MainBlock(C, C, kernel_size=kernel_sizes[i], R=R)
            )

        # the epilogue layers
        self.epilogue_conv1 = nn.Conv1d(C, 128, kernel_size=29, dilation=2)
        self.epilogue_bnorm1 = nn.BatchNorm1d(128)

        self.epilogue_conv2 = nn.Conv1d(128, 128, kernel_size=1)
        self.epilogue_bnorm2 = nn.BatchNorm1d(128)

        self.epilogue_conv3 = nn.Conv1d(128, NUM_CLASSES, kernel_size=1)

        # Pool the timesteps into a single dimension using simple average pooling
        self.epilogue_adaptivepool = nn.AdaptiveAvgPool1d(1)

    def padding(self, batch, seq_len):
        if len(batch[0][0]) < seq_len:
            m = torch.nn.ConstantPad1d((0, seq_len - len(batch[0][0])), 0)
            batch = m(batch)
        return batch

    def emphasis(self, audio, pre_emphasis = 0.97):
        audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        return audio

    def forward(self, x):
        # prologue block
        x = self.prologue_conv1(x)
        x = self.prologue_bnorm1(x)
        x = F.relu(x)

        # intermediate blocks
        for layer in self.blocks:
            x = layer(x)

        # epilogue blocks
        x = self.epilogue_conv1(x)
        x = self.epilogue_bnorm1(x)

        x = self.epilogue_conv2(x)
        x = self.epilogue_bnorm2(x)

        x = self.epilogue_conv3(x)
        x = self.epilogue_adaptivepool(x)
        x = x.squeeze(2)  # (N, 30, 1) > (N, 30)
        x = F.softmax(x, dim=1)  # softmax across classes and not batch

        return x


class MFCC_MatchboxNet(nn.Module):
    """
    A MatchboxNet model implementation for audio classification using MFCC features.

    Args:
        bins (int): The number of Mel frequency bins in the input spectrogram.
        B (int): The number of convolutional blocks in the model.
        R (int): The number of sub-blocks per block in the model.
        n_channels (int): The number of channels in the input feature map.
        kernel_sizes (list): A list of kernel sizes to use for the sub-blocks in the model.
        num_classes (int): The number of output classes for the model.

    """
    def __init__(self, bins: int, B: int, R: int, n_channels, kernel_sizes=None, num_classes=12):
        super(MFCC_MatchboxNet, self).__init__()
        self.sampling_rate = 16000
        self.bins = bins
        self.num_classes = num_classes
        self.mfcc_layer = MFCC(sample_rate=self.sampling_rate, n_mfcc=self.bins, log_mels=True)
        self.matchboxnet = MatchboxNet(B, R, n_channels, bins=self.bins, kernel_sizes=kernel_sizes,
                                       NUM_CLASSES=num_classes)

    def forward(self, waveform):
        mel_sepctogram = self.mfcc_layer(waveform)
        mel_sepctogram = mel_sepctogram.squeeze(1)
        mel_sepctogram = self.matchboxnet.padding(mel_sepctogram, 128)
        logits = self.matchboxnet(mel_sepctogram)
        return logits