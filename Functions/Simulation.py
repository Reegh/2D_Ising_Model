import numpy as np
from tqdm import tqdm  # Progress bar
from multiprocessing import Pool, cpu_count
from Functions.Ising_Model import IsingModel2D
import yaml
from pathlib import Path

def cargar_config():
    ruta_config = Path(__file__).parent.parent / "config.yaml"
    
    with open(ruta_config, 'r') as archivo:
        return yaml.safe_load(archivo) 
config = cargar_config()

def simulate_at_temperature(args):
    """
    Función wrapper para multiprocesamiento.
    Args:
        args: Tupla con (T, L, n_steps, seed)
    """
    T, L, n_steps, seed = args
    np.random.seed(seed)  # Single seed per process
    model = IsingModel2D(L=L)
    initial_spins = model.spins.copy()  # Saves the initial configuration
    
    capture_frames = T in [config["Model"]["initial.temp"], 2.0, 2.269, config["Model"]["final.temp"]]
    result = model.simulate(
        T,
        n_steps=n_steps,
        capture_frames=capture_frames,
        frame_interval=config["Video"]["frames"]
    )
    result['initial_spins'] = initial_spins
    return result

def run_temperature_sweep(L=config["Model"]["matrix"], T_min=config["Model"]["initial.temp"], T_max=config["Model"]["final.temp"], n_T=config["Model"]["temp.points"], n_steps=config["Carlo"]["steps"]):
    """
    Parallel temperature sweep.
    
    Args:
        L (int): Size of the square matrix (LxL)
        T_min/max (float): Temperatures range
        n_T (int): Number of temperature points
        n_steps (int): Monte Carlo steps per simulation
    """
    temperatures = np.linspace(T_min, T_max, n_T)
    seeds = np.random.randint(0, 1e6, size=n_T)  # Random seeds
    
    # Uses only 4 processors to reduce overhead
    with Pool(processes=min(4, cpu_count())) as pool:
        args = [(T, L, n_steps, seed) for T, seed in zip(temperatures, seeds)]
        results = list(tqdm(
            pool.imap(simulate_at_temperature, args),
            total=n_T,
            desc="Running temperature sweep"
        ))
    
    return sorted(results, key=lambda x: x['temperature'])