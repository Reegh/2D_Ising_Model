import numpy as np
from tqdm import tqdm  # Para barra de progreso
from multiprocessing import Pool, cpu_count
from Functions.Ising_Model import IsingModel2D

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
    
    capture_frames = T in [1.0, 2.0, 2.269, 3.5]
    result = model.simulate(
        T,
        n_steps=n_steps,
        capture_frames=capture_frames,
        frame_interval=30
    )
    result['initial_spins'] = initial_spins
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