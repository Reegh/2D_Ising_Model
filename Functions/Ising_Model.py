import numpy as np
from numba import jit  # For JIT acceleration
import yaml
from pathlib import Path

def cargar_config():
    ruta_config = Path(__file__).parent.parent / "config.yaml"
    
    with open(ruta_config, 'r') as archivo:
        return yaml.safe_load(archivo) 
config = cargar_config()

class IsingModel2D:
    def __init__(self, L=config["Model"]["matrix"], J=config["Model"]["coupling"], h=config["Model"]["magnetic"]):
        """
        Initializes the 2D Ising-Model
        
        Args:
            L (int): Size of the square matrix (LxL)
            J (float): Coupling constant (J>0: ferromagnetic)
            h (float): External magnetic field
        """
        self.L = L
        self.J = J
        self.h = h
        # Random spin initialization (+1 or -1)
        self.spins = np.random.choice([-1, 1], size=(L, L)).astype(np.int8)  # int8 to save memory
        self.initial_spins = self.spins.copy()
        self.temperature = None

    @staticmethod
    @jit(nopython=True)
    def _metropolis_step(spins, L, J, h, temperature):
        """Metropolis algorithm optimized with Numba."""
        for _ in range(L * L):
            i, j = np.random.randint(0, L), np.random.randint(0, L)
            nn_sum = spins[(i+1)%L, j] + spins[(i-1)%L, j] + spins[i, (j+1)%L] + spins[i, (j-1)%L]
            delta_E = 2 * J * spins[i,j] * nn_sum + 2 * h * spins[i,j]
            
            if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / temperature):
                spins[i,j] *= -1
        return spins

    def metropolis_step(self):
        """Wrapper for Metropolis algorithm with Numba."""
        self.spins = self._metropolis_step(
            self.spins, self.L, self.J, self.h, self.temperature
        )

    def calculate_energy(self):
        """Calculates the energy of the system (vectorized)."""
        # Using np.roll for periodic boundary conditions
        nn_sum = (np.roll(self.spins, 1, axis=0) + np.roll(self.spins, -1, axis=0) +
                 np.roll(self.spins, 1, axis=1) + np.roll(self.spins, -1, axis=1))
        return (-self.J * np.sum(self.spins * nn_sum) - self.h * np.sum(self.spins)) / 2

    def calculate_magnetization(self):
        """Calculates the average magnetización per spin."""
        return np.mean(self.spins)

    def simulate(self, temperature, n_steps=config["Carlo"]["steps"], eq_steps=config["Carlo"]["equilibrium"], capture_frames=False, frame_interval=config["Video"]["frames"]):
        """
        Runs the simulation for a given temperature.
        
        Args:
            temperature (float): Temperature in units of J/kB
            n_steps (int): Total Monte Carlo steps
            eq_steps (int): Balancing steps (discarded)
            capture_frames (boolean): False
            frame_interval (int): Number of frames
        
        Returns:
            dict: Simulations results
        """
        self.temperature = temperature
        frames = [self.spins.copy()] if capture_frames else []
        
        # Balancing phase
        for _ in range(eq_steps):
            self.metropolis_step()
        
        # Measurment phase
        energies = np.zeros(n_steps - eq_steps)
        magnetizations = np.zeros(n_steps - eq_steps)
        
        for step in range(n_steps - eq_steps):
            self.metropolis_step()
            energies[step] = self.calculate_energy()
            magnetizations[step] = self.calculate_magnetization()

            if capture_frames and step % frame_interval == 0:
                frames.append(self.spins.copy())
        
        # Calculation of thermodynamic observables
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
            'frames': frames if capture_frames else None  # << Only when requested
        }