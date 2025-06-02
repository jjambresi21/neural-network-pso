import numpy as np

class PSO:
    def __init__(self, nn, X, y, num_particles=50, max_iter=100, w=0.7, c1=1.5, c2=1.5, verbose=True):
        self.nn = nn
        self.X = X
        self.y = y
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.verbose = verbose
        
        self.swarm = np.random.uniform(-1, 1, (num_particles, nn.total_params))
        self.velocities = np.zeros_like(self.swarm)
        self.pbest_positions = self.swarm.copy()
        self.pbest_scores = np.full(num_particles, np.inf)
        self.gbest_position = None
        self.gbest_score = np.inf
    
    def optimize(self):
        for iteration in range(self.max_iter):
            for i in range(self.num_particles):
                current_score = self.nn.calculate_mse_loss(self.y, self.nn.forwardpropagation(self.X, self.swarm[i]))
                
                if current_score < self.pbest_scores[i]:
                    self.pbest_positions[i] = self.swarm[i]
                    self.pbest_scores[i] = current_score
                
                if current_score < self.gbest_score:
                    self.gbest_position = self.swarm[i]
                    self.gbest_score = current_score
            
            for i in range(self.num_particles):
                r1, r2 = np.random.random(2)
                cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.swarm[i])
                social = self.c2 * r2 * (self.gbest_position - self.swarm[i])
                
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                self.swarm[i] += self.velocities[i]
            
            # Ispis najboljeg rezultata svakih 10 iteracija
            # if self.verbose and (iteration % 10 == 0 or iteration == self.max_iter - 1):
            #     print(f"Iteracija {iteration+1}/{self.max_iter}, Najbolji MSE: {self.gbest_score:.6f}")
        
        return self.gbest_position, self.gbest_score