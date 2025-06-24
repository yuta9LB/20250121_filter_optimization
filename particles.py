import numpy as np

class Particles:
    def __init__(self, w=0.729, c1=1.4, c2=1.4, N=100, patch_num=20, height=(-100, 100), width=(-100, 100), max_area=2000):
        self.x = np.random.randint([height[0], width[0]], [height[1], width[1]], (N, patch_num, 2, 2))
        self.v = np.random.rand(N, patch_num, 2, 2)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.N = N
        self.patch_num = patch_num
        self.height = height
        self.width = width
        self.max_area = max_area
        self.pbest = np.zeros((N, patch_num, 2, 2)) # 自己ベスト
        self.pbest_fitness = np.inf * np.ones(N)
        self.gbest = np.zeros((patch_num, 2, 2)) # グローバルベスト
        self.gbest_fitness = np.inf
        self.fitness = np.zeros(N) # 適応度
        self.cnt = 0 # 停滞カウント

        # 初期パッチの面積を制限
        for i in range(self.N):
            for j in range(self.patch_num):
                patch = self.x[i, j]
                area = np.abs((patch[1, 0] - patch[0, 0]) * (patch[1, 1] - patch[0, 1]))
                if area > self.max_area:
                    scale_factor = (self.max_area / area) ** 0.25
                    patch_center = np.mean(patch, axis=0)
                    self.x[i, j] = (patch_center + (patch - patch_center) * scale_factor).astype(int)

    def update(self):
        # 粒子の位置と速度を更新
        for i in range(self.N):
            r1 = np.random.rand()
            r2 = np.random.rand()
            self.v[i] = self.w * self.v[i] + self.c1 * r1 * (self.pbest[i] - self.x[i]) + self.c2 * r2 * (self.gbest - self.x[i])
            if self.v[i].sum() == 0:
                self.v[i] = np.random.uniform(-2.0, 2.0, (2, self.patch_num, 2, 2))
            self.x[i] = (self.x[i].astype(float) + self.v[i]).astype(int)

            # パッチの面積を制限
            for j in range(self.patch_num):
                patch = self.x[i, j]
                area = np.abs((patch[1, 0] - patch[0, 0]) * (patch[1, 1] - patch[0, 1]))
                if area > self.max_area:
                    scale_factor = (self.max_area / area) ** 0.25
                    patch_center = np.mean(patch, axis=0)
                    self.x[i, j] = (patch_center + (patch - patch_center) * scale_factor).astype(int)

                # パッチの位置を制限
                self.x[i, j, :, 0] = np.clip(self.x[i, j, :, 0], self.height[0], self.height[1])
                self.x[i, j, :, 1] = np.clip(self.x[i, j, :, 1], self.width[0], self.width[1])

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
