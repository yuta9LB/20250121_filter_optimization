import numpy as np
import pandas as pd


def cal_spara(std_Feed, dut_Feed):
    delta_t = dut_Feed[1][1]-dut_Feed[1][0] # 時間分解能
    freq_list = np.fft.fftfreq(len(dut_Feed[1]), delta_t) # 周波数リスト
    target_freq_mask = freq_list >= 0 # とりあえず正だけ考える
    freq_list = freq_list[target_freq_mask] # 正の周波数のみを抽出

    standard = np.fft.fft(std_Feed[5]) # 標準のFFT
    reflect = np.fft.fft(dut_Feed[5]-std_Feed[5]) # 反射のFFT
    trancmission  = np.fft.fft(dut_Feed[6]) # 通過のFFT

    s11 = np.abs(reflect / standard) # S11を計算
    s11 = s11[target_freq_mask] # 正の周波数のみを抽出
    s21 = np.abs(trancmission / standard) # S21を計算
    s21 = s21[target_freq_mask] # 正の周波数のみを抽出

    return s11, s21, freq_list

def calculate_objective_function(
    s11,
    s21,
    freq_list, 
    pass_band_list,
    stop_band_list,
    w_p=1.0,
    w_s=1.0
    ):
    pass_band = np.zeros_like(freq_list, dtype=bool)
    stop_band = np.zeros_like(freq_list, dtype=bool)

    for band in pass_band_list:
        pass_band |= (freq_list >= band[0]) & (freq_list < band[1])
    for band in stop_band_list:
        stop_band |= (freq_list >= band[0]) & (freq_list < band[1])

    s21_pass = s21[pass_band]
    s11_stop = s11[stop_band]

    s21_pass_db = 20 * np.log10(s21_pass)  # dBに変換
    e_pass = np.sum(s21_pass_db**2)  # 平方和を計算
    for band in pass_band_list:
        e_pass -= (1 / 2) * 20 * np.log10(np.max((s21[(freq_list >= band[0]) & (freq_list < band[1])])) - np.min((s21[(freq_list >= band[0]) & (freq_list < band[1])])))
    # e_pass = np.sum(1 - np.abs(s21_pass)**2)  # 平方和を計算

    s11_stop_db = 20 * np.log10(s11_stop)  # dBに変換
    e_stop = np.sum(s11_stop_db**2)  # 平方和を計算
    for band in stop_band_list:
        e_stop -= (1 / 5) * 20 * np.log10(np.max((s11[(freq_list >= band[0]) & (freq_list < band[1])])) - np.min((s11[(freq_list >= band[0]) & (freq_list < band[1])])))
    # e_stop = np.sum(1 - np.abs(s11_stop)**2)  # 平方和を計算

    objective_value = (w_p * e_pass) + (w_s * e_stop)

    return objective_value

# def calculate_objective_function(
#     s11,
#     s21,
#     pass_band0,
#     pass_band1,
#     stop_band0,
#     stop_band1,
#     stop_band2,
#     w_p=1.0,
#     w_s=1.0
# ):
#     """
#     マイクロ波マルチバンドパスフィルタの目的関数を計算します。

#     Args:
#         s11 (pd.Series): S11パラメータ (dB)。インデックスは周波数。
#         s21 (pd.Series): S21パラメータ (dB)。インデックスは周波数。
#         pass_band0 (pd.Series): 1番目の通過帯域を示すbooleanマスク。
#         pass_band1 (pd.Series): 2番目の通過帯域を示すbooleanマスク。
#         stop_band0 (pd.Series): 1番目の阻止帯域を示すbooleanマスク。
#         stop_band1 (pd.Series): 2番目の阻止帯域を示すbooleanマスク。
#         stop_band2 (pd.Series): 3番目の阻止帯域を示すbooleanマスク。
#         r_target (float): リターンロスの目標値 [dB]。この値以上を目指します。
#         i_target (float): 挿入損失の目標値 [dB]。この値以下（0dBに近い）を目指します。
#         a_target (float): 減衰量の目標値 [dB]。この値以下を目指します。
#         w_p (float): 通過帯域の誤差に対する重み係数。
#         w_s (float): 阻止帯域の誤差に対する重み係数。

#     Returns:
#         float: 計算された目的関数の値 (スコア)。
#     """

#     pass_bands_all = pass_band0 | pass_band1
#     s11_pass = s11[pass_bands_all]
#     s21_pass = s21[pass_bands_all]

#     error_s11 = np.sum(np.abs(s11_pass)**2)
#     error_s21_pass = np.sum(1 - np.abs(s21_pass)**2)
#     e_pass = error_s11 + error_s21_pass

#     stop_bands_all = stop_band0 | stop_band1 | stop_band2
#     s21_stop = s21[stop_bands_all]

#     e_stop = np.sum(np.abs(s21_stop)**2)

#     objective_value = (w_p * e_pass) + (w_s * e_stop)

#     return objective_value

def cal_fitness(i):
    # 結果の読み込み
    std_Feed = pd.read_csv('/home/takayama/20250121_filter_optimization/std_2G/std.Feed', header=None, sep='\s+', index_col=0)
    # std_Feed = pd.read_csv('/home/takayama/20250121_filter_optimization/std/std.Feed', header=None, sep='\s+', index_col=0)
    dut_Feed = pd.read_csv(f'{i}.Feed', header=None, sep='\s+', index_col=0)

    # S11, S21の計算
    s11, s21, freq_list = cal_spara(std_Feed, dut_Feed)

    fitness = calculate_objective_function(
        s11=s11,
        s21=s21,
        freq_list=freq_list, 
        pass_band_list=[(2.3e9, 2.6e9), (5.6e9, 6.0e9)],
        stop_band_list=[(1.0e9, 2.3e9),(2.6e9, 5.6e9), (6.0e9, 8.0e9)],
        w_p=9.0*7/10,
        w_s=64/103
    )

    # # 適応度の計算
    # fitness = calculate_objective_function(
    #     s11=s11,
    #     s21=s21,
    #     pass_band0=((freq_list >= 2.4e9) & (freq_list <= 2.5e9)),
    #     pass_band1=((freq_list >= 5.725e9) & (freq_list <= 5.875e9)),
    #     stop_band0=(freq_list >= 2.0e9) & (freq_list <= 2.3e9),
    #     stop_band1=(freq_list >= 2.6e9) & (freq_list <= 5.6e9),
    #     stop_band2=(freq_list >= 6.0e9) & (freq_list <= 7.0e9),
    #     w_p=15.0,
    #     w_s=1.0
    # )
    return fitness
