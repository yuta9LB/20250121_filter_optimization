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
    max_area = config['max_area']
    range_len = tuple(config['range_len']) if config['range_len'] else None
    stop_steps = config['stop_steps']

    DB_dir = f'{save_dir}/DB'

    os.makedirs(DB_dir, exist_ok=True)
    
    # configfileのコピーを保存
    with open(f'{save_dir}/config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=True)
        f.close()

    # DBファイル作成
    with open(f'{DB_dir}/DB.csv', 'w') as f:
        f.write(f'Iter\tID\tSeries_ID\tfitness\n')
        f.close()

    # 初期設定定義
    t = 0
    particles = Particles(N=N, patch_num=patch_num, height=height, width=width, max_area=max_area, range_len=range_len)

    while (t < T) and (particles.cnt < stop_steps):
        print(f'Iteration: {t} Count: {particles.cnt} Global Best Fitness: {particles.gbest_fitness}')

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
            if not output:  # qstatが空 or ヘッダーのみなら終了
                break
            time.sleep(10)

        print(f'Queue insertion was completed.')

        # job.sh.oから始まるファイルを削除
        for file in os.listdir('.'):
            if file.startswith('job.sh.o'):
                os.remove(file)

        # 評価
        particles.evaluate(cal_fitness)

        # 各粒子のIDとSeries ID、適応度を保存
        with open(f'{DB_dir}/DB.csv', 'a') as f:
            for i in range(N):
                f.write(f'{t}\t{i}\t{t*N+i}\t{particles.fitness[i]}\n')
    
        # 更新
        particles.update()

        # Global Best保存
        if not os.path.exists(f'{save_dir}/{save_name}.csv'):
            with open(f'{save_dir}/{save_name}.csv', 'w') as f:
                f.write(f'iter,fitness\n')
                f.close()
    
        with open(f'{save_dir}/{save_name}.csv', 'a') as f:
            f.write(f'{t},{particles.gbest_fitness}\n')
            f.close()

        t += 1

    print(f'End of the program. The best fitness is {particles.gbest_fitness}.')
    

if __name__ == '__main__':
    config = yaml.safe_load(open('./yaml/config_2G.yaml'))
    main(config)
