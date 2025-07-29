import numpy as np

class Particles:
    def __init__(self, w=0.9, c1=1.4, c2=0.7, N=100, patch_num=20, height=(-100, 100), width=(-100, 100), max_area=None, range_len=None):
        self.x = np.random.randint([height[0], width[0]], [height[1], width[1]], (N, patch_num, 2, 2))
        self.v = np.random.rand(N, patch_num, 2, 2)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.N = N
        self.patch_num = patch_num
        self.height = height
        self.width = width
        # max_areaかrange_lenのいずれかのみであることを確認
        assert (max_area is not None) ^ (range_len is not None), "Either max_area or range_len must be specified, but not both."
        if max_area is not None:
            self.max_area = max_area
        else:
            self.range_len = range_len
        self.pbest = np.zeros((N, patch_num, 2, 2)) # 自己ベスト
        self.pbest_fitness = np.inf * np.ones(N)
        self.gbest = np.zeros((patch_num, 2, 2)) # グローバルベスト
        self.gbest_fitness = np.inf
        self.fitness = np.zeros(N) # 適応度
        self.cnt = 0 # 停滞カウント

        if max_area is not None:
            # 初期パッチの面積を制限
            for i in range(self.N):
                for j in range(self.patch_num):
                    patch = self.x[i, j]
                    area = np.abs((patch[1, 0] - patch[0, 0]) * (patch[1, 1] - patch[0, 1]))
                    if area > self.max_area:
                        scale_factor = (self.max_area / area) ** 0.25
                        patch_center = np.mean(patch, axis=0)
                        self.x[i, j] = (patch_center + (patch - patch_center) * scale_factor).astype(int)
        if range_len is not None:
            # 初期パッチの長さを制限
            # 縦横の長さの和がrange_len[1]未満かつ各辺の長さがrange_len[0]以上になるように調整
            for i in range(self.N):
                for j in range(self.patch_num):
                    # 条件を満たすまでパッチを調整
                    while True:
                        patch = self.x[i, j]
                        height_length = np.abs(patch[1, 0] - patch[0, 0])
                        width_length = np.abs(patch[1, 1] - patch[0, 1])
                        total_length = height_length + width_length
                        if total_length > self.range_len[1]:
                            scale_factor = self.range_len[1] / total_length
                            patch_center = np.mean(patch, axis=0)
                            self.x[i, j] = (patch_center + (patch - patch_center) * scale_factor).astype(int)
                        elif height_length < self.range_len[0]:
                            patch[:, 0] = np.random.randint(self.height[0], self.height[1], (2,))
                        elif width_length < self.range_len[0]:
                            patch[:, 1] = np.random.randint(self.width[0], self.width[1], (2,))
                        else:
                            break
                        
                    self.x[i, j] = patch.astype(int)

    def update(self):
        # 粒子の位置と速度を更新
        for i in range(self.N):
            r1 = np.random.rand()
            r2 = np.random.rand()
            self.v[i] = self.w * self.v[i] + self.c1 * r1 * (self.pbest[i] - self.x[i]) + self.c2 * r2 * (self.gbest - self.x[i])
            if self.v[i].sum() == 0:
                self.v[i] = np.random.uniform(-2.0, 2.0, (self.patch_num, 2, 2))
            self.x[i] = (self.x[i].astype(float) + self.v[i]).astype(int)

            # パッチの位置を制限
            for j in range(self.patch_num):
                patch = self.x[i, j]
                # 高さの制限
                patch[0, 0] = max(self.height[0], min(self.height[1], patch[0, 0]))
                patch[1, 0] = max(self.height[0], min(self.height[1], patch[1, 0]))
                # 幅の制限
                patch[0, 1] = max(self.width[0], min(self.width[1], patch[0, 1]))
                patch[1, 1] = max(self.width[0], min(self.width[1], patch[1, 1]))
                
                # パッチの面積制限
                if hasattr(self, 'max_area'):
                    area = np.abs((patch[1, 0] - patch[0, 0]) * (patch[1, 1] - patch[0, 1]))
                    if area > self.max_area:
                        scale_factor = (self.max_area / area) ** 0.25
                        patch_center = np.mean(patch, axis=0)
                        self.x[i, j] = (patch_center + (patch - patch_center) * scale_factor).astype(int)
                
                # パッチの長さ制限
                if hasattr(self, 'range_len'):
                    height_length = np.abs(patch[1, 0] - patch[0, 0])
                    width_length = np.abs(patch[1, 1] - patch[0, 1])
                    total_length = height_length + width_length
                    if total_length > self.range_len[1]:
                        scale_factor = self.range_len[1] / total_length
                        patch_center = np.mean(patch, axis=0)
                        self.x[i, j] = (patch_center + (patch - patch_center) * scale_factor).astype(int)
                    elif height_length < self.range_len[0]:
                        patch[:, 0] = np.random.randint(self.height[0], self.height[1], (2,))
                    elif width_length < self.range_len[0]:
                        patch[:, 1] = np.random.randint(self.width[0], self.width[1], (2,))
                
                self.x[i, j] = patch.astype(int)

    def evaluate(self, fitness_func):
        updated = False

        # 適応度を評価
        for i in range(self.N):
            self.fitness[i] = fitness_func(i)
        
        # 自己ベストの更新
        for i in range(self.N):
            if self.fitness[i] < self.pbest_fitness[i]:
                assert self.pbest[i].shape == self.x[i].shape
                self.pbest[i] = self.x[i]
                self.pbest_fitness[i] = self.fitness[i]

            # グローバルベストの更新
            if self.fitness[i] < self.gbest_fitness:
                assert self.gbest.shape == self.x[i].shape
                self.gbest = self.x[i]
                self.gbest_fitness = self.fitness[i]
                updated = True

        # 停滞カウントの更新
        if updated:
            self.cnt = 0
        else:
            self.cnt += 1
