from PIL import Image

import pycuda.autoinit  # Inicjalizacja CUDA
from pycuda import gpuarray  # Przenoszenie danych na GPU
import OpenGL.GL as gl  # Renderowanie OpenGL
from OpenGL.GLUT import *  # Zarządzanie oknem
import numpy as np


def render():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glPointSize(2.0)  # Rozmiar punktów
    gl.glBegin(gl.GL_POINTS)
    
    for i in range(len(xxx_gpu)):
        gl.glColor3f(vvv[i], 0, 0)  # Kolor zależny od vvv_gpu
        gl.glVertex3f(xxx[i], yyy[i], zzz[i])

    
    gl.glEnd()
    save_frame()
    glutSwapBuffers()
    


def save_frame():
    """Zapisuje aktualną klatkę do pliku PNG."""
    print("Zapisuję klatkę...")
    global frame_count
    width, height = 800, 600
    gl.glReadBuffer(gl.GL_FRONT)
    pixels = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGB", (width, height), pixels)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)  # OpenGL ma odwróconą oś Y
    image.save(f"./plots/frame_{frame_count:04d}.png")
    print(f"Zapisano frame_{frame_count:04d}.png")
    frame_count += 1
    glutDestroyWindow(glutGetWindow())

if __name__ == "__main__":
    # Przykładowe dane
    n = 1000
    xxx = np.random.rand(n) * 10
    yyy = np.random.rand(n) * 10
    zzz = np.random.rand(n) * 10
    vvv = np.random.rand(n) * 25 + 340  # zakres ~340-365

    # Przeniesienie danych na GPU
    xxx_gpu = gpuarray.to_gpu(xxx.astype(np.float32))
    yyy_gpu = gpuarray.to_gpu(yyy.astype(np.float32))
    zzz_gpu = gpuarray.to_gpu(zzz.astype(np.float32))
    vvv_gpu = gpuarray.to_gpu(vvv.astype(np.float32))

    # Inicjalizacja OpenGL i GLUT
    glutInit()
    glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGB | glut.GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow("3D Plot")
    
    # Ustawienia kamery
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gluPerspective(45.0, 1.0, 0.1, 100.0)
    
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    
    # Ustawienia oświetlenia (opcjonalne)
    gl.glEnable(gl.GL_LIGHTING)
    gl.glEnable(gl.GL_LIGHT0)
    
    # Ustawienia renderowania
    frame_count = 0
    glutDisplayFunc(render)
    
    glutInit()  # Inicjalizacja GLUT
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)  # Tryb wyświetlania
    glutInitWindowSize(800, 600)
    glutCreateWindow(b"GPU Scatter Plot")  # Tworzenie okna

    glutDisplayFunc(render)