import numpy as np
import pandas as pd

def cal_s11(std_Feed, dut_Feed):
    delta_t = dut_Feed[1][1]-dut_Feed[1][0] # 時間分解能
    freq_list = np.fft.fftfreq(len(dut_Feed[1]), delta_t) # 周波数リスト

    standard = np.fft.fft(std_Feed[5]) # 標準のFFT
    reflect = np.fft.fft(dut_Feed[5]-std_Feed[5]) # 反射のFFT
    s11 = reflect/standard # S11を計算
    target_freq_mask = freq_list >= 0 # とりあえず正だけ考える
    s11 = s11[target_freq_mask]
    
    return s11

def cal_s21(std_Spe, dut_Spe):
    s21  = dut_Spe[2] / std_Spe[2]
    return s21

def _fitness(s21, std_Spe, alpha=0.5):
    pass_band0 = ((std_Spe.index >= 2.4e9) & (std_Spe.index <= 2.5e9))
    pass_band1 = ((std_Spe.index >= 5.725e9) & (std_Spe.index <= 5.875e9))
    s21_pass_band0 = s21[pass_band0]
    s21_pass_band1 = s21[pass_band1]
    stop_band = (std_Spe.index >= 2.6e9) & (std_Spe.index <= 5.6e9)
    s21_stop_band = s21[stop_band]

    assert alpha >= 0 and alpha <= 1, "alpha must be between 0 and 1"
    fitness_s21_pass0 = np.max(np.abs(20 * np.log10(np.abs(s21_pass_band0))))
    fitness_s21_pass1 = np.max(np.abs(20 * np.log10(np.abs(s21_pass_band1))))
    fitness_s21_stop = 20 - np.min(np.abs(20 * np.log10(np.abs(s21_stop_band))))

    # 絶対の制約
    if fitness_s21_pass0 > 3:
        fitness_s21_pass0 = 50
    if fitness_s21_pass1 > 3:
        fitness_s21_pass1 = 50

    fitness = alpha * (fitness_s21_pass0 + fitness_s21_pass1) + (1 - alpha) * fitness_s21_stop
    return fitness

# def _fitness(s21, std_Spe, w_11=1, w_21=1):
# # def _fitness(s11, s21, std_Spe, w_11=1, w_21=1):
#     target_range = ((std_Spe.index >= 2.4e9) & (std_Spe.index <= 2.5e9)) | ((std_Spe.index >= 5.725e9) & (std_Spe.index <= 5.875e9))
#     # target_range = ((std_Spe.index >= 902e6) & (std_Spe.index <= 928e6)) | ((std_Spe.index >= 2.4e9) & (std_Spe.index <= 2.5e9)) | ((std_Spe.index >= 5.725e9) & (std_Spe.index <= 5.875e9))
#     s21_target_range = s21[target_range]
#     s11_non_target_range = 1 - s21[~target_range & ((std_Spe.index >= 5e8) & (std_Spe.index <= 12e9))]
#     # s11_non_target_range = s11[~target_range & ((std_Spe.index >= 5e8) & (std_Spe.index <= 12e9))]
#     fitness_s11 = w_11 * np.mean(np.abs(20*np.log10(np.abs(s11_non_target_range))))
#     fitness_s21 = w_21 * np.mean(np.abs(20*np.log10(np.abs(s21_target_range))))
#     fitness = np.sqrt(fitness_s11**2 + fitness_s21**2)
#     return fitness

def cal_fitness(i):
    # 結果の読み込み
    # std_Feed = pd.read_csv('/home/takayama/workspace/20250121_filter_optimization/std/std.Feed', header=None, sep='\s+', index_col=0)
    std_Spe = pd.read_csv('/home/takayama/20250121_filter_optimization/std/std.Spectrum', header=None, sep='\s+', index_col=0)
    # dut_Feed = pd.read_csv(f'{i}.Feed', header=None, sep='\s+', index_col=0)
    dut_Spe = pd.read_csv(f'{i}.Spectrum', header=None, sep='\s+', index_col=0)

    # S11, S21の計算
    # s11 = cal_s11(std_Feed, dut_Feed)
    s21 = cal_s21(std_Spe, dut_Spe)

    # 適応度の計算
    fitness = _fitness(s21, std_Spe, alpha=0.5)
    # fitness = _fitness(s11, s21, std_Spe, w_11=1, w_21=2)
    return fitness
