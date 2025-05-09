import time
import numpy as np
from vispy import scene, io
from vispy.scene import visuals
from vispy.color import Colormap
import sys
#from PyQt6.QtOpenGL import QOpenGLWidget
from PyQt6.QtWidgets import QApplication

def render_vispy_3d(xxx, yyy, zzz, vvv, 
                   xfliml, xflimh, yfliml, yflimh, zfliml, zflimh, 
                   output_path='./', size=(900.0, 600.0), show = False):
    app = QApplication.instance() or QApplication(sys.argv)
    
    # Konfiguracja canvas (off-screen)
    canvas = scene.SceneCanvas(keys='interactive', size=size, show=show, bgcolor='gray')
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'  # lub 'arcball'
    
    vmin = np.min(vvv)
    vmax = np.max(vvv)
    # Punkty 3D
    scatter = visuals.Markers()
    scatter.set_data(np.column_stack([xxx, yyy, zzz]), 
                    edge_color=None, 
                    face_color=_map_colors(vvv, vmin, vmax),
                    size=5)
    view.add(scatter)
    
    # Osie
    axes = visuals.XYZAxis(parent=view.scene)
    
    # Ustawienia kamery
    view.camera.set_range(x=(xfliml, xflimh), y=(yfliml, yflimh), z=(zfliml, zflimh))
    
    # Colorbar (wymaga dodatkowej implementacji)
    _add_colorbar(canvas, vmin, vmax, show)
    
    # Renderowanie do pliku
    img = canvas.render()
    io.write_png(output_path, img)
    canvas.close()
    app.quit()

def _map_colors(values, vmin, vmax):
    """Mapuje wartości na kolory używając palety 'hot'"""

    normalized = (values - vmin) / (vmax - vmin)
    # VisPy's hot colormap
    cmap = Colormap(['black', 'red', 'yellow', 'white'])
    return cmap.map(normalized)

def _add_colorbar(canvas, vmin, vmax, show):
    """Dodaje prostą skalę kolorów (poglądowo)"""
    # W VisPy nie ma wbudowanego colorbar, trzeba go ręcznie narysować
    # Tutaj uproszczona wersja - w praktyce lepiej dodać go w post-processingu
    from vispy import scene
    from vispy.scene import widgets
    

    cb_canvas = scene.SceneCanvas(keys='interactive', size=(100, 500), 
                                 show=show, bgcolor='white')
    cb_view = cb_canvas.central_widget.add_view()
    
    # Stwórz gradient
    gradient = np.linspace(0, 1, 256).reshape(256, 1)
    gradient = np.tile(gradient, (1, 50))
    gradient_img = scene.visuals.Image(gradient, parent=cb_view.scene, 
                                      cmap='hot', clim=(vmin, vmax))
    
    # Etykiety
    label_min = scene.visuals.Text(f'{vmin:.1f}', pos=(10, 10), 
                                  color='black', parent=cb_view.scene)
    label_max = scene.visuals.Text(f'{vmax:.1f}', pos=(10, 490), 
                                  color='black', parent=cb_view.scene)
    
    return cb_canvas

# Przykład użycia:
if __name__ == "__main__":
    # Generowanie przykładowych danych
    n = 1000
    xxx = np.random.rand(n) * 10
    yyy = np.random.rand(n) * 10
    zzz = np.random.rand(n) * 10
    vvv = np.random.rand(n) * 25 + 340  # zakres ~340-365
    t = time.time()
    render_vispy_3d(xxx, yyy, zzz, vvv, 
                   0, 10,     # xxx limits
                   0, 10,     # yyy limits
                   0, 10,     # zzz limits
                   'vispy_plot.png')
    
    print(f"Zapisano wykres w {time.time() - t:.7f} sekundy.")