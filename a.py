
import matplotlib.pylab as plt  # 绘制图形
import numpy as np
from numpy.fft import fft, fftfreq, fftshift
from numpy.fft import ifftshift, ifft
from matplotlib.widgets import Slider

T = 1 / 4  # cos信号周期 T
Ts = T / 2  # 抽样间隔（过采样）
t_length1 = 5  # 信号的实际时间长度
sample_freq = 1024  # 模拟连续信号的“密度”
sample_interval = 1 / sample_freq  # 模拟连续信号的采样间隔
fcc = int(1 / T) + 2  # 解调LPF的带宽

titlestr = r"单频率信号时域抽样（$fs= %.*ffm$）"%(1,T/Ts)
namestr = r"单频率信号时域抽样（fs=%.*ffm）"%(1,T/Ts)

t = np.linspace(-t_length1, t_length1, 2 * t_length1 * sample_freq)  # 原信号的定义区间
f = fftshift(fftfreq(len(t), sample_interval))
sig1 = np.cos(2 * np.pi / T * t + 0.125 * np.pi)# + noise_amp * np.cos(15 * np.pi / T * t)

pulse_1 = np.hstack([[1], np.zeros(int(Ts * sample_freq) - 1)])
def return_pulse(t, Ts, sample_freq,pulse_1):
    pulse = np.array([])
    while len(pulse) < len(t):
        pulse = np.hstack([pulse, pulse_1])
    pulse = sample_freq * pulse[:len(t)]
    return pulse

pulse=return_pulse(t, Ts, sample_freq,pulse_1)
sample_sig = sig1 * pulse

fft_data1 = fft(sig1)
fft_amp1 = np.abs(fft_data1) * sample_interval  # 双边幅度谱
fft_amp1 = fftshift(fft_amp1)  # 修改坐标范围，方便画图

fft_data2 = fft(pulse)
fft_amp2 = np.abs(fft_data2) * sample_interval  # 双边幅度谱
fft_amp2 = fftshift(fft_amp2)  # 修改坐标范围，方便画图

fft_data3 = fft(sample_sig)
fft_amp3 = np.abs(fft_data3) * sample_interval  # 双边幅度谱
fft_amp3 = fftshift(fft_amp3)  # 修改坐标范围，方便画图
fontsize = 12



g,ax=plt.subplots(3,2,figsize=(16, 9))
axcolor = 'lightgoldenrodyellow'
ax_k = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
slider_k = Slider(ax_k, 'K', 0.1, 10.0, valinit=2)
plt.subplots_adjust(left=0.25, bottom=0.25)
axcolor = 'lightgoldenrodyellow'
ax_k = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
slider_k = Slider(ax_k, 'K', 0.1, 10.0, valinit=2)
line1,=ax[0,0].plot(t, sig1, 'r')
def update(res):
    k = slider_k.val
    Ts_t = T/k
    #重新定义抽样脉冲
    pulse_1 = np.hstack([[1], np.zeros(int(Ts_t * sample_freq) - 1)])
    pulse = np.array([])
    while len(pulse) < len(t):
        pulse = np.hstack([pulse, pulse_1])
    pulse = sample_freq * pulse[:len(t)]
    sample_sig = sig1 * pulse
    titlestr = r"单频率信号时域抽样（$fs= %.*ffm$）"%(1,T/Ts_t)
    namestr = r"单频率信号时域抽样（fs=%.*ffm）"%(1,T/Ts_t)

    line1.set_ydata(pulse)
slider_k.on_changed(update)
plt.show()