import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Configuración de estilo para los gráficos
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12

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
            if result.get('frames') is not None:
                animate_spin_frames(result['frames'], result['temperature'])

def animate_spin_frames(frames, temperature):

    os.makedirs('Results', exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(frames[0], cmap='coolwarm', vmin=-1, vmax=1)
    ax.axis('off')
    ax.set_title(f'Spin Evolution (T = {temperature:.3f})')

    def update(frame):
        im.set_data(frame)
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=frames, blit=True, interval=100
    )

    # Guardar como GIF
    #ani.save(f"Results/spin_animation_T_{temperature:.3f}.gif", writer='pillow')

    # Guardar como MP4 (requiere ffmpeg)
    ani.save(f"Results/spin_animation_T_{temperature:.3f}.mp4", writer='ffmpeg', fps=10)

    plt.close()


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