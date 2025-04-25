import numpy as np

class Particles:
    def __init__(self, w=0.729, c1=1.4, c2=1.4, N=100, patch_num=20, height=(-100, 100), width=(-100, 100), max_length=50):
        self.x = np.random.randint([height[0], width[0]], [height[1], width[1]], (N, patch_num, 2, 2))
        self.v = np.random.rand(N, 1, 2, 2)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.N = N
        self.patch_num = patch_num
        self.height = height
        self.width = width
        self.max_length = max_length
        self.pbest = np.zeros((N, patch_num, 2, 2)) # 自己ベスト
        self.pbest_fitness = np.inf * np.ones(N)
        self.gbest = np.zeros((patch_num, 2, 2)) # グローバルベスト
        self.gbest_fitness = np.inf
        self.fitness = np.zeros(N) # 適応度
        self.cnt = 0

        # 初期パッチの一片の長さを制限
        for i in range(self.N):
            for j in range(self.patch_num):
                patch = self.x[i, j]
                lengths = np.linalg.norm(patch[1] - patch[0])
                if lengths > self.max_length:
                    scale_factor = self.max_length / lengths
                    patch_center = np.mean(patch, axis=0)
                    self.x[i, j] = patch_center + (patch - patch_center) * scale_factor

    def update(self):
        # 粒子の位置を更新
        self.x += self.v.astype(int)

        # パッチの一片の長さを制限
        for i in range(self.N):
            for j in range(self.patch_num):
                patch = self.x[i, j]
                lengths = np.linalg.norm(patch[1] - patch[0])
                if lengths > self.max_length:
                    scale_factor = self.max_length / lengths
                    patch_center = np.mean(patch, axis=0)
                    self.x[i, j] = patch_center + (patch - patch_center) * scale_factor

        # パッチのサイズを制限
        self.x = np.clip(self.x, [self.height[0], self.width[0]], [self.height[1], self.width[1]])

        # 粒子の速度を更新
        self.v = self.w * self.v + self.c1 * np.random.rand() * (self.pbest - self.x) + self.c2 * np.random.rand() * (self.gbest - self.x)
        for n in range(self.N):
            if np.all(self.v[n] == 0):
                print("v is zero")
                self.v[n] = np.random.rand(1, 2, 2)

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

        if updated:
            self.cnt = 0
        else:
            self.cnt += 1
