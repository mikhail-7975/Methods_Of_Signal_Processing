import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import random
from plotly.subplots import make_subplots


def draw(y, x, yf, xf, title=''):
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(
        go.Scatter(y=y, x=x, name='f'), row=1, col=1)

    fig.add_trace(go.Scatter(y=yf, x=xf, name='fourier'), row=1, col=2)
    fig.update_layout(height=600, width=800, title_text=title)
    fig.show()


def lff(x, x0=3):
    return np.array([complex(1., abs(i) * 2 * np.pi / x0) if -(x0 + 0.01) < i < (x0 + 0.01) else complex(0., 0.) for i in x])


def draw_single(x, y, title='', t=5):
    yf = np.abs(np.fft.fft(y))
    N = len(y)
    xf = np.fft.fftfreq(N) * N * 1 / t
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Scatter(x=x, y=y), row=1, col=1)
    fig.add_trace(go.Scatter(x=xf[:500], y=yf[:500]), row=1, col=2)
    fig.update_layout(height=400, width=800, title_text=title)
    fig.show()


def draw_wo(y, yf, title=''):
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(
        go.Scatter(y=y, name='f'), row=1, col=1)

    fig.add_trace(go.Scatter(y=yf, name='fourier'), row=1, col=2)
    fig.update_layout(height=600, width=800, title_text=title)
    fig.show()


import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# butter_lowpass_filter(y_over, f_resample, current_freq*P)

def low_freq_filter(signal, freq, t):
    df = 1. / t
    N = len(signal)
    xf = np.fft.fftfreq(N) * N * df
    x_con = np.linspace(-5, 5, N)
    ideal_lff_F = lff(xf, freq)
    ideal_lff_F_c = ideal_lff_F

    g = gaus(np.concatenate((xf[int(N / 2):], xf[:int(N / 2)])), freq)

    ideal_lff = np.fft.ifft(ideal_lff_F_c);
    ideal_lff_real = ideal_lff
    ideal_lff_real = np.concatenate((ideal_lff_real[int(N / 2):], ideal_lff_real[:int(N / 2)]))

    ideal_lff_real_w = ideal_lff_real[int(N / 2 - 25):int(N / 2) + 25]
    graph_ideal_lff_real_w = np.concatenate(
        (np.zeros((int(N / 2) - 25)), ideal_lff_real_w, np.zeros((int(N / 2) - 25))))
    ideal_lff_real_wg = graph_ideal_lff_real_w * g
    low_freq_filter = ideal_lff_real_wg[int(N / 2) - 25:int(N / 2) + 25].real
    #     draw(ideal_lff_real, x_con, np.abs(np.fft.fft(ideal_lff_real)), xf)
    #     draw(graph_ideal_lff_real_w, x_con, np.abs(np.fft.fft(graph_ideal_lff_real_w)), xf)
    #     draw(ideal_lff_real_wg, x_con, np.abs(np.fft.fft(ideal_lff_real_wg)), xf)
    return np.convolve(signal, ideal_lff_real_wg[int(N / 2) - 25:int(N / 2) + 25])[25:N + 25].real


def gaus(x, sigma=1, mu=0):
    return np.array([np.exp(-(i - mu) ** 2 / (sigma ** 2)) for i in x])

Uc = 1
fc = 10

Um = 0.2
fm = 3
m = 0.8
t = np.linspace(0,5,1000)
carrier = Uc*np.cos(2*np.pi*fc*t)
modulator = Um*np.cos(2*np.pi*fm*t)
#product = Uc*(1 + m * np.cos(2 * np.pi * fm * t)) * np.cos(2*np.pi*fc*t)
Wr = 2 * np.pi * fc
reference_sig = 0.5 * np.cos(Wr * t)

def quad_demod(carrier, modulator, reference_sig, t, m=0.8):
    product = carrier*(1 + m * modulator / max(modulator))
    draw_single(t, carrier,'Несущий')
    draw_single(t, modulator, 'Модулируемый')
    draw_single(t, product, 'Произведение')
    draw_single(t, reference_sig, 'R')
    res = reference_sig*product
    draw_single(t, res, 'Произведение с R')
    y = low_freq_filter(res,fc,5)
    draw_single(t, y, 'То, что перенесли')
    return y

###########################################################################
Uc = 1 #амплитуда несущего сингнала
fc = 10 #несущая частота

Um = 0.2 #амплитуда модулируемого сигнала
fm = 3 #частота модулируемого сигнала
m = 0.8
t = np.linspace(0,5,1000)
carrier = Uc*np.cos(2*np.pi*fc*t)
modulator = Um*np.cos(2*np.pi*fm*t)
#product = Uc*(1 + m * np.cos(2 * np.pi * fm * t)) * np.cos(2*np.pi*fc*t)
Wr = 2 * np.pi * fc
reference_sig = 0.5 * np.cos(Wr * t)
y = quad_demod(carrier,modulator,reference_sig,t)
####################################################################################################

Uc = 1
fc = 10

Um = 0.2
fm = 3
m = 0.8
t = np.linspace(0,5,1000)
carrier = Uc*np.cos(2*np.pi*fc*t)
modulator = np.cos(2*np.pi*fm*t) * np.cos(0.5*np.pi*fm*t)
#product = Uc*(1 + m * np.cos(2 * np.pi * fm * t)) * np.cos(2*np.pi*fc*t)
Wr = 2 * np.pi * fc
reference_sig = 0.5 * np.cos(Wr * t)

y = quad_demod(carrier,modulator,reference_sig,t)

####################################################################################################

Uc = 1

fc = 10

Um = 0.2
fm = 3
m = 0.8
t = np.linspace(0,5,1000)
carrier = Uc*np.cos(2*np.pi*fc*t)
modulator = np.cos(2*np.pi*fm*t) * np.cos(0.5*np.pi*fm*t)
#product = Uc*(1 + m * np.cos(2 * np.pi * fm * t)) * np.cos(2*np.pi*fc*t)
Wr = 2 * np.pi * fc
reference_sig = 0.5 * np.cos(Wr * t)

y = quad_demod(carrier, modulator, reference_sig, t)

####################################################################################################

import random
t = np.linspace(-100,100,1000000)
y = np.sin(2*np.pi*t)+2*np.sin(4*2*np.pi*t)+1.5*np.sin(8*2*np.pi*t)
y1 = [0.1*random.random() + i for i in y]
draw_single(t, y, t=200)
draw_single(t, y1, t=200)