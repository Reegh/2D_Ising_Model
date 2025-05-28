import numpy as np
from numba import jit  # Para aceleración JIT

class IsingModel2D:
    def __init__(self, L=50, J=1.0, h=0.0):
        """
        Inicializa el modelo de Ising 2D.
        
        Args:
            L (int): Tamaño de la red (LxL)
            J (float): Constante de acoplamiento (J>0: ferromagnético)
            h (float): Campo magnético externo
        """
        self.L = L
        self.J = J
        self.h = h
        # Inicialización aleatoria de espines (+1 o -1)
        self.spins = np.random.choice([-1, 1], size=(L, L)).astype(np.int8)  # int8 para ahorrar memoria
        self.initial_spins = self.spins.copy()
        self.temperature = None

    @staticmethod
    @jit(nopython=True)
    def _metropolis_step(spins, L, J, h, temperature):
        """Paso de Metropolis optimizado con Numba."""
        for _ in range(L * L):
            i, j = np.random.randint(0, L), np.random.randint(0, L)
            nn_sum = spins[(i+1)%L, j] + spins[(i-1)%L, j] + spins[i, (j+1)%L] + spins[i, (j-1)%L]
            delta_E = 2 * J * spins[i,j] * nn_sum + 2 * h * spins[i,j]
            
            if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / temperature):
                spins[i,j] *= -1
        return spins

    def metropolis_step(self):
        """Wrapper para el paso de Metropolis con Numba."""
        self.spins = self._metropolis_step(
            self.spins, self.L, self.J, self.h, self.temperature
        )

    def calculate_energy(self):
        """Calcula la energía del sistema (vectorizado)."""
        # Uso de np.roll para condiciones de frontera periódicas
        nn_sum = (np.roll(self.spins, 1, axis=0) + np.roll(self.spins, -1, axis=0) +
                 np.roll(self.spins, 1, axis=1) + np.roll(self.spins, -1, axis=1))
        return (-self.J * np.sum(self.spins * nn_sum) - self.h * np.sum(self.spins)) / 2

    def calculate_magnetization(self):
        """Calcula la magnetización promedio por espín."""
        return np.mean(self.spins)

    def simulate(self, temperature, n_steps=10000, eq_steps=2000, capture_frames=False, frame_interval=50):
        """
        Ejecuta la simulación para una temperatura dada.
        
        Args:
            temperature (float): Temperatura en unidades de J/kB
            n_steps (int): Pasos totales de Monte Carlo
            eq_steps (int): Pasos de equilibración (se descartan)
        
        Returns:
            dict: Resultados de la simulación
        """
        self.temperature = temperature
        frames = [self.spins.copy()] if capture_frames else []
        
        # Fase de equilibración
        for _ in range(eq_steps):
            self.metropolis_step()
        
        # Fase de medición
        energies = np.zeros(n_steps - eq_steps)
        magnetizations = np.zeros(n_steps - eq_steps)
        
        for step in range(n_steps - eq_steps):
            self.metropolis_step()
            energies[step] = self.calculate_energy()
            magnetizations[step] = self.calculate_magnetization()

            if capture_frames and step % frame_interval == 0:
                frames.append(self.spins.copy())
        
        # Cálculo de observables termodinámicos
        avg_E = np.mean(energies)
        var_E = np.var(energies)
        
        if capture_frames:
                frames.append(self.spins.copy())

        return {
            'temperature': temperature,
            'avg_energy': avg_E / (self.L**2),
            'abs_magnetization': np.mean(np.abs(magnetizations)),
            'heat_capacity': var_E / (temperature**2 * self.L**2),
            'initial_spins': self.initial_spins,
            'final_spins': self.spins.copy(),
            'frames': frames if capture_frames else None  # << Solo cuando se pide
        }