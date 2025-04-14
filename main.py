import os
import yaml
import time
import subprocess
from cal_fitness import cal_fitness
from particles import Particles
from utils import make_conf, make_jobscript, make_circuit_image


def main(config):
    save_dir = config['save_dir']
    save_name = config['save_name']
    N = config['N']
    T = config['T']
    patch_num = config['patch_num']
    height = tuple(config['height']) # (-109, 109)
    width = tuple(config['width']) # (-165, 201)

    DB_dir = f'{save_dir}/DB'

    os.makedirs(DB_dir, exist_ok=True)

    # DBファイル作成
    with open(f'{DB_dir}/DB.csv', 'w') as f:
        f.write(f'Iter\tID\tSeries_ID\tfitness\n')
        f.close()

    # 初期設定定義
    t = 0
    particles = Particles(N=N, patch_num=patch_num, height=height, width=width)

    while (t < T) or (particles.cnt < 30):
        print(f'Iteration: {t}')

        # ステップごとのディレクトリを作成し、移動
        os.makedirs(f'{save_dir}/{t}', exist_ok=True)
        os.chdir(f'{save_dir}/{t}')

        # configファイルの作成
        make_conf(particles)

        # 回路画像の作成
        make_circuit_image(particles, t, N, DB_dir)

        # ジョブスクリプトの作成
        make_jobscript(t, N, save_dir)

        # FDTDの実行
        os.system('qsub job.sh >& PBS.log')
        # qstatで何も出力されなくなるまで待つ.
        # 但し出力はしないようにする
        while True:
            result = subprocess.run(["qstat"], capture_output=True, text=True)
            output = result.stdout.strip()
            if not output or "Job ID" not in output:  # qstatが空 or ヘッダーのみなら終了
                break
            time.sleep(10)

        print(f'Queue insertion was completed.')

        # 評価
        particles.evaluate(cal_fitness)

        # 各粒子のIDとSeries ID、適応度を保存
        with open(f'{DB_dir}/DB.csv', 'a') as f:
            for i in range(N):
                f.write(f'{t}\t{i}\t{t*N+i}\t{particles.fitness[i]}\n')
    
        # 更新
        t += 1
        particles.update()

        # Global Best保存
        if not os.path.exists(f'{save_dir}/{save_name}.csv'):
            with open(f'{save_dir}/{save_name}.csv', 'w') as f:
                f.write(f'iter,fitness\n')
                f.close()
    
        with open(f'{save_dir}/{save_name}.csv', 'a') as f:
            f.write(f'{t},{particles.gbest_fitness}\n')
            f.close()

if __name__ == '__main__':
    config = yaml.safe_load(open('./yaml/config.yaml'))
    main(config)
