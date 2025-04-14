import os
import numpy as np


TEMPLATE = open('StaticFiles/static.inc').readlines()
TEMPLATE0 = open('StaticFiles/static0.inc').readlines()


def make_conf(particles):
    for i in range(particles.N):
        # 設定ファイルの作成
        template = TEMPLATE.copy()

        for [x0, y0], [x1, y1] in particles.x[i]:
            template.append(f"Copper Plane ({y0}, {x0}, 3) ({y1}, {x1}, 3)\n")
        
        # 設定ファイルに書き込み
        with open(f'{i}.conf', 'w') as f:
            f.writelines(template)

def make_jobscript(t, N, save_dir):
    # ジョブスクリプトの作成
    jobscript = open('job.sh', 'w')
    jobscript.write(f'''#!/bin/bash
#PBS -V
#PBS -j oe
#PBS -l nodes=1:gpus=1:exclusive_process
#PBS -t 0-{N-1}

cd {save_dir}/{t}
''')
    jobscript.write('''fdtd_gpu ${PBS_ARRAYID}.conf ${PBS_ARRAYID}\n''')
    jobscript.close()

def make_circuit_image(particles, t ,N, DB_dir):
    for i in range(N):
        template = TEMPLATE0.copy()

        for [x0, y0], [x1, y1] in particles.x[i]:
            template.append(f"Copper Plane ({y0}, {x0}, 3) ({y1}, {x1}, 3)\n")
        
        with open(f'tmp.conf', 'w') as f:
            f.writelines(template)
        
        os.system(f'fdtd_gpu tmp.conf tmp')
        os.system(f'gerbv --border=0 --dpi=287.547169811 tmp.ger -x png -o {DB_dir}/{t*N+i}.png')
    # os.remove('rm tmp*')

def cal_sparams(std_Feed, dut_Feed, std_Spe, dut_Spe):
    delta_t = dut_Feed[1][1]-dut_Feed[1][0] # 時間分解能
    freq_list = np.fft.fftfreq(len(dut_Feed[1]), delta_t) # 周波数リスト
    standard = np.fft.fft(std_Feed[5]) # 標準のFFT
    reflect = np.fft.fft(dut_Feed[5]-std_Feed[5]) # 反射のFFT
    s11 = reflect/standard # S11を計算
    target_freq_mask = freq_list >= 0 # とりあえず正だけ考える
    s11 = s11[target_freq_mask]
    s21  = dut_Spe[2] / std_Spe[2]
    return s11, s21

def fitness(s11, s21, std_Spe, w_11=1, w_21=1):
    target_range = ((std_Spe.index >= 902e6) & (std_Spe.index <= 928e6)) | ((std_Spe.index >= 2.4e9) & (std_Spe.index <= 2.5e9)) | ((std_Spe.index >= 5.725e9) & (std_Spe.index <= 5.875e9))
    s21_target_range = s21[target_range]
    s11_non_target_range = s11[~target_range & ((std_Spe.index >= 5e8) & (std_Spe.index <= 12e9))]
    fitness_s11 = w_11 * np.mean(np.abs(20*np.log10(np.abs(s11_non_target_range))))
    fitness_s21 = w_21 * np.mean(np.abs(20*np.log10(np.abs(s21_target_range))))
    fitness = np.sqrt(fitness_s11**2 + fitness_s21**2)
    return fitness
