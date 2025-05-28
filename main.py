from Functions.Simulation import run_temperature_sweep
from Functions.Graphics import plot_results
import yaml

def load_config(ruta="config.yaml"):
    with open(ruta, 'r') as archivo:
        return yaml.safe_load(archivo)

config = load_config()

if __name__ == "__main__":
    # Optimized parameters for a good speed-quality balance
    results = run_temperature_sweep(
        L=config["Model"]["matrix"],              # Balanced size
        T_min=config["Model"]["initial.temp"],         # Heat rating
        T_max=config["Model"]["final.temp"],
        n_T=config["Model"]["temp.points"],            # Temperature points
        n_steps=config["Carlo"]["steps"]      # Total steps
    )
    
    plot_results(results)
    print("Simulation completed! Results saved in Results")