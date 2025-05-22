import numpy as np
import vispy
from vispy import app, visuals
from vispy.scene import SceneCanvas
from vispy.color import Colormap
from vispy import scene
import imageio
import os
import time

def setup_vispy_scatter_plot():
    """Inicjalizacja canvas i obiektów VisPy (offscreen)."""
    # Canvas z białym tłem (offscreen)
    canvas = scene.SceneCanvas(keys='interactive', title='Scatter Plot', 
                             show=False, bgcolor='white', size=(800, 600))
    
    # Widok
    view = canvas.central_widget.add_view()
    view.camera = 'panzoom'  # Umożliwia automatyczne skalowanie
    view.bgcolor = '#f0f0f0'  # Jasnoszare tło dla lepszej widoczności siatki
    
    # Siatka (grid) - bardziej widoczna
    grid = scene.visuals.GridLines(color='black', scale=(1, 1))
    grid.set_gl_state('translucent', line_width=1)
    view.add(grid)
    
    # Osie z opisami
    
    # Scatter plot (pusty na początku)
    scatter = scene.visuals.Markers(parent=view.scene)
    scatter.set_gl_state('translucent', depth_test=False)
    
    # Mapa kolorów "hot"
    cmap = Colormap(['black', 'red', 'orange', 'yellow'])
    
    return canvas, view, scatter, cmap

def save_vispy_scatter(canvas, view, scatter, cmap,
                      rrr, vector_f, zzz, v_scatt, xlim, ylim, output_file):
    """Aktualizuje i zapisuje wykres offscreen."""
    # Aktualizacja danych
    scatter.set_data(
        np.column_stack([rrr, vector_f]),
        size=v_scatt,
        face_color=cmap.map(zzz),
        edge_color=None,
        symbol='o'
    )
    
    # Ustawienie zakresu osi (z marginesem dla lepszej widoczności)
    margin_x = (xlim[1] - xlim[0]) * 0.05
    margin_y = (ylim[1] - ylim[0]) * 0.05

    view.camera.set_range(x=(xlim[0]-margin_x, xlim[1]+margin_x),
                         y=(ylim[0]-margin_y, ylim[1]+margin_y))
    
    '''
    axis_x = scene.visuals.Axis(pos=[[0, ylim[0]], [1, ylim[0]]],
                              tick_direction=(0, -1),
                              font_size=10,
                              axis_label='r [μm]',
                              axis_font_size=12,
                              tick_color='black',
                              axis_color='black',
                              text_color='black')'''

    x_label = scene.visuals.Text(
        text='r [μm]',
        pos=(np.mean(xlim), ylim[0] - 0.05 * (ylim[1] - ylim[0])),
        color='black',
        font_size=12,
        parent=view.scene
    )
    
    y_label = scene.visuals.Text(
        text='Density',
        pos=(xlim[0] - 0.05 * (xlim[1] - xlim[0]), np.mean(ylim)),
        color='black',
        font_size=12,
        rotation=90,
        parent=view.scene
    )


    # Wymuszenie renderowania (offscreen)
    canvas.update()
    canvas.app.process_events()
    
    # Zapis do pliku (zwiększona DPI dla lepszej jakości)
    img = canvas.render(alpha=False)
    imageio.imwrite(output_file, img)

# Przykład użycia w pętli:
if __name__ == "__main__":
    # Inicjalizacja (tylko raz)
    canvas, view, scatter, cmap = setup_vispy_scatter_plot()
    
    # Parametry wspólne dla wszystkich wykresów
    zfliml, zflimh = 0.0, 1.0
    v_scatt = 10
    xlim = (0, 10)
    ylim = (0, 5)
    
    # Symulacja danych (tutaj użyj swoich danych)
    num_plots = 5
    os.makedirs("plots_vispy", exist_ok=True)  # folder na wykresy
    
    for ii in range(num_plots):
        # Generowanie losowych danych (zastąp swoimi)
        n_points = 1000
        rrr = np.random.rand(n_points) * 10
        vector_f = np.random.rand(n_points) * 5
        zzz = np.random.rand(n_points)
        
        # Zapis wykresu
        output_file = f"plots_vispy/scatt90nr_{ii}.png"
        t=time.time()
        save_vispy_scatter(
            canvas, view, scatter, cmap,
            rrr, vector_f, zzz, v_scatt,
            xlim, ylim, output_file
        )
        print(f"Zapisano {output_file} zajeło {time.time()-t:.5f} s")
    
    # Zamknięcie canvas (opcjonalne)
    canvas.close()