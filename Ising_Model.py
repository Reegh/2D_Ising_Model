import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm  # Para barra de progreso
from numba import jit  # Para aceleración JIT

# Configuración de estilo para los gráficos
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12

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

    def simulate(self, temperature, n_steps=10000, eq_steps=2000):
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
        
        # Cálculo de observables termodinámicos
        avg_E = np.mean(energies)
        var_E = np.var(energies)
        
        return {
            'temperature': temperature,
            'avg_energy': avg_E / (self.L**2),
            'abs_magnetization': np.mean(np.abs(magnetizations)),
            'heat_capacity': var_E / (temperature**2 * self.L**2),
            'final_spins': self.spins.copy()
        }

def simulate_at_temperature(args):
    """
    Función wrapper para multiprocesamiento.
    Args:
        args: Tupla con (T, L, n_steps, seed)
    """
    T, L, n_steps, seed = args
    np.random.seed(seed)  # Semilla única por proceso
    model = IsingModel2D(L=L)
    initial_spins = model.spins.copy()  # Guardamos la configuración inicial
    result = model.simulate(T, n_steps=n_steps)
    result['initial_spins'] = initial_spins  # Añadimos los espines iniciales al resultado
    return result

def run_temperature_sweep(L=50, T_min=1.0, T_max=3.5, n_T=20, n_steps=10000):
    """
    Barrido de temperaturas en paralelo.
    
    Args:
        L (int): Tamaño de la red
        T_min/max (float): Rango de temperaturas
        n_T (int): Número de puntos de temperatura
        n_steps (int): Pasos de Monte Carlo por simulación
    """
    temperatures = np.linspace(T_min, T_max, n_T)
    seeds = np.random.randint(0, 1e6, size=n_T)  # Semillas aleatorias
    
    # Usamos solo 4 procesos para reducir overhead
    with Pool(processes=min(4, cpu_count())) as pool:
        args = [(T, L, n_steps, seed) for T, seed in zip(temperatures, seeds)]
        results = list(tqdm(
            pool.imap(simulate_at_temperature, args),
            total=n_T,
            desc="Running temperature sweep"
        ))
    
    return sorted(results, key=lambda x: x['temperature'])

def plot_spin_configurations(initial_spins, final_spins, temperature):
    """Visualiza las configuraciones inicial y final de los espines."""
    # Creamos la figura con un diseño manual
    fig = plt.figure(figsize=(12, 6))
    
    # Añadimos los subplots con posiciones manuales
    ax1 = fig.add_axes([0.05, 0.1, 0.4, 0.8])  # [left, bottom, width, height]
    ax2 = fig.add_axes([0.5, 0.1, 0.4, 0.8])
    
    # Configuración inicial
    im1 = ax1.imshow(initial_spins, cmap='coolwarm', vmin=-1, vmax=1)
    ax1.set_title(f'Initial Configuration (T = {temperature:.3f})')
    ax1.axis('off')
    
    # Configuración final
    im2 = ax2.imshow(final_spins, cmap='coolwarm', vmin=-1, vmax=1)
    ax2.set_title(f'Final Configuration (T = {temperature:.3f})')
    ax2.axis('off')
    
    # Barra de color compartida
    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # Posición de la barra de color
    fig.colorbar(im2, cax=cax, label='Spin value')
    cax.set_yticks([-1, 0, 1])
    
    plt.savefig(f'Results/spin_configs_T_{temperature:.3f}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_results(results):
    """Visualización profesional de los resultados."""
    Ts = np.array([r['temperature'] for r in results])
    Es = np.array([r['avg_energy'] for r in results])
    Ms = np.array([r['abs_magnetization'] for r in results])
    Cs = np.array([r['heat_capacity'] for r in results])
    
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))
    
    # Gráfico de energía
    plt.figure(figsize=(8, 5))
    plt.plot(Ts, Es, 'o-', color='royalblue', label='Simulation')
    plt.axvline(x=2.269, color='red', linestyle='--', label='Critical T')
    plt.xlabel(r'Temperature ($k_BT/J$)')
    plt.ylabel(r'Energy per spin ($E/J$)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('Results/ising_energy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfico de magnetización
    plt.figure(figsize=(8, 5))
    plt.plot(Ts, Ms, 'o-', color='darkorange', label='Simulation')
    plt.axvline(x=2.269, color='red', linestyle='--')
    plt.xlabel(r'Temperature ($k_BT/J$)')
    plt.ylabel('Magnetization per spin')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('Results/ising_magnetization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfico de capacidad calorífica
    plt.figure(figsize=(8, 5))
    plt.plot(Ts, Cs, 'o-', color='forestgreen', label='Simulation')
    plt.axvline(x=2.269, color='red', linestyle='--')
    plt.xlabel(r'Temperature ($k_BT/J$)')
    plt.ylabel(r'Heat Capacity ($C/k_B$)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('Results/ising_heat_capacity.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Graficar configuraciones de espines para temperaturas seleccionadas
    for result in results:
        if result['temperature'] in [min(Ts), 2.0, 2.269, max(Ts)]:  # Temperaturas clave
            plot_spin_configurations(
                initial_spins=result['initial_spins'],
                final_spins=result['final_spins'],
                temperature=result['temperature']
            )

if __name__ == "__main__":
    # Parámetros optimizados para buen balance velocidad-calidad
    results = run_temperature_sweep(
        L=50,              # Tamaño balanceado
        T_min=1.0,         # Rango térmico
        T_max=3.5,
        n_T=25,            # Puntos de temperatura
        n_steps=10000       # Pasos totales (500 de equilibración)
    )
    
    plot_results(results)
    print("¡Simulación completada! Resultados guardados en 'ising_results.png'")