import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def draw(y, x, yf, xf, title=''):
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(
        go.Scatter(y=y, x=x, name='f'), row=1, col=1)

    fig.add_trace(go.Scatter(y=yf, x=xf, name='fourier'), row=1, col=2)
    fig.update_layout(height=600, width=800, title_text=title)
    fig.show()

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

def lff(x, x0=3):
    return np.array(
        [complex(1., abs(i) * 2 * np.pi / x0) if -(x0 + 0.01) < i < (x0 + 0.01) else complex(0., 0.) for i in x])

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
    return np.convolve(signal, ideal_lff_real_wg[int(N / 2) - 25:int(N / 2) + 25])[25:N + 25].real


def gaus(x, sigma=1, mu=0):
    return np.array([np.exp(-(i - mu) ** 2 / (sigma ** 2)) for i in x])


def amplitude_demod(carrier, modulator, reference_sig, product, t, m=0.8, L=5):
    draw_single(t, carrier, 'carrier')
    draw_single(t, modulator, 'modulator')
    draw_single(t, product, 'multiplication')
    draw_single(t, reference_sig, 'R ')
    res = reference_sig * product
    draw_single(t, res, 'multiply with R')
    y = low_freq_filter(res, fc, L)
    draw_single(t, y, 'Ys')
    return y


def quad_demod(reference_sig, product, t, m=0.8, L=5):
    draw_single(t, reference_sig, 'R_shift_phase')
    res = reference_sig * product
    draw_single(t, res, 'multiplication with R_shift_phase')
    y = low_freq_filter(res, fc, L)
    draw_single(t, y, 'Yc')
    return y

def demod(Uc, fc, Um, fm, m, fp, L):
    t = np.linspace(0, L, 1000)
    carrier = Uc * np.cos(2 * np.pi * fc * t)
    modulator = Um * np.cos(2 * np.pi * fm * t)
    Wr = 2 * np.pi * fc
    reference_sig = 0.5 * np.cos(Wr * t)
    reference_sig_shift_phase = 0.5 * np.cos(Wr * t + np.pi / 2)
    product = 3 * np.cos(2 * np.pi * fp * t)

    y_s = amplitude_demod(carrier, modulator, reference_sig, product, t, L)

    y_c = quad_demod(reference_sig_shift_phase, product, t, L)

    y = np.array([complex(y_s[i], y_c[i]) for i in range(len(y_s))])
    draw_single(t, abs(y), 'Y = ys + i * yc')
    return y

Uc = 1 #амплитуда несущего сигнала
fc = 10 #частота несущего сигнала

Um = 0.8 #амплитуда модулируемого сигнала
fm = 3 #частота модулируемого сигнала
m = 0.2
L = 5
y = demod(Uc, fc, Um, fm, m, 9, L)
t = 5
N = len(y)
yf = np.abs(np.fft.fft(y))
xf = np.fft.fftfreq(N) * N * 1 / t
fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Scatter(x=xf, y=yf), row=1, col=1)
fig.update_layout(height=400, width=800, title_text="frequency moving on fc-fp")
fig.show()