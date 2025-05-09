import time
import numpy as np
import moderngl
from PIL import Image, ImageDraw, ImageFont
from pyrr import Matrix44
from numba import cuda, jit
import imageio.v3 as iio 


def render_moderngl_3d(xxx, yyy, zzz, vvv, 
                      xfliml, xflimh, yfliml, yflimh, zfliml, zflimh, 
                      output_path, size=(1920, 1080), point_size=3.0):
    # 1. Inicjalizacja kontekstu OpenGL (off-screen)
    ctx = moderngl.create_standalone_context(require=330)
    ctx.enable(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE)
    
    # 2. Przygotowanie danych
    vmin, vmax = np.min(vvv), np.max(vvv)
    normalized_colors = (vvv - vmin) / (vmax - vmin)
    
    # Konwersja do float32
    t = time.time()
    vertices = np.column_stack([xxx, yyy, zzz]).astype('f4')
    colors = normalized_colors.astype('f4')
    print(f'Przygotowanie danych: {time.time()-t:.7f} s')
    # 3. Shadery GLSL
    vertex_shader = """
    #version 330
    uniform mat4 mvp;
    in vec3 in_position;
    in float in_color;
    out float v_color;
    void main() {
        gl_Position = mvp * vec4(in_position, 1.0);
        v_color = in_color;
        gl_PointSize = %f;
    }
    """ % point_size
    
    fragment_shader = """
    #version 330
    uniform float vmin;
    uniform float vmax;
    in float v_color;
    out vec4 fragColor;
    
    vec3 hot_colormap(float t) {
        return vec3(
            min(1.0, max(0.0, t * 4.0 - 1.5)), // R
            min(1.0, max(0.0, t * 2.0 - 0.5)), // G
            min(1.0, max(0.0, t * 2.0))       // B
        );
    }
    
    void main() {
        vec3 color = hot_colormap(v_color);
        fragColor = vec4(color, 1.0);
    }
    """
    
    # 4. Kompilacja programu
    prog = ctx.program(
        vertex_shader=vertex_shader,
        fragment_shader=fragment_shader
    )
    
    # 5. Buffery wierzchołków
    vbo_verts = ctx.buffer(vertices.tobytes())
    vbo_colors = ctx.buffer(colors.tobytes())
    
    vao = ctx.vertex_array(
        prog,
        [
            (vbo_verts, '3f4', 'in_position'),
            (vbo_colors, '1f4', 'in_color')
        ]
    )
    print(f'Kompilacja shadera: {time.time()-t:.7f} s')
    # 6. Macierze transformacji
    center = np.array([np.mean([xfliml, xflimh]), 
                      np.mean([yfliml, yflimh]), 
                      np.mean([zfliml, zflimh])])
    
    radius = max(xflimh-xfliml, yflimh-yfliml, zflimh-zfliml) * 1.5
    
    proj = Matrix44.perspective_projection(45.0, size[0]/size[1], 0.1, radius*2)
    view = Matrix44.look_at(
        (center[0], center[1], center[2] + radius),
        center,
        (0, 1, 0)
    )
    mvp = proj * view
    
    # 7. Renderowanie
    t = time.time()
    fbo = ctx.framebuffer(
        color_attachments=[ctx.texture(size, 4)],
        depth_attachment=ctx.depth_texture(size)
    )
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 1.0)
    print(f'Inicjalizacja FBO: {time.time()-t:.7f} s')
    prog['mvp'].write(mvp.astype('f4').tobytes())
    #prog['vmin'].value = vmin
    #prog['vmax'].value = vmax
    
    vao.render(moderngl.POINTS)
    
    # 8. Pobranie obrazu
    t = time.time()
    img_data = fbo.read(components=3)
    img = Image.frombytes('RGB', size, img_data)
    print(f'Pobieranie obrazu: {time.time()-t:.7f} s')

    # 9. Dodanie colorbar
    img = add_colorbar(img, vmin, vmax)


    iio.imwrite(output_path, image = img,
               compression=6)
   # img.save(output_path) # wąskie gardło 0.1s

    vao.release()
    vbo_verts.release()
    vbo_colors.release()
    prog.release()
    fbo.release()
    ctx.release()
    print(f'Renderowanie: {time.time()-t:.7f} s')


def add_colorbar(img, vmin, vmax, width=100):
    """Dodaje colorbar do obrazu używając PIL"""
    # Utwórz gradient
    height = img.height
    gradient = np.linspace(1, 0, height).reshape(height, 1)
    gradient = np.tile(gradient, (1, width))
    
    # Mapowanie kolorów (hot)
    hot_colors = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        val = 1 - i/height
        r = min(255, max(0, int(val * 4 * 255 - 1.5 * 255)))
        g = min(255, max(0, int(val * 2 * 255 - 0.5 * 255)))
        b = min(255, max(0, int(val * 2 * 255)))
        hot_colors[i, :] = [r, g, b]
    
    # Stwórz finalny obraz
    colorbar_img = Image.fromarray(hot_colors, 'RGB')
    final_img = Image.new('RGB', (img.width + width, height))
    final_img.paste(img, (0, 0))
    final_img.paste(colorbar_img, (img.width, 0))
    
    # Dodaj etykiety
    draw = ImageDraw.Draw(final_img)

    
    draw.text((img.width + 10, 10), str(vmax), fill=(255,0,0))
    draw.text((img.width + 10, height-30), str(vmin), fill=(255,0,0))
    
    return final_img

# Przykład użycia:
if __name__ == "__main__":
    # Generowanie danych testowych
    n = 3102  # 100k punktów
    xxx = np.random.rand(n) * 10
    yyy = np.random.rand(n) * 10
    zzz = np.random.rand(n) * 10
    vvv = np.random.rand(n) * 25 + 340  # zakres ~340-365
    t0 = time.time()
    # Renderowanie
    render_moderngl_3d(
        xxx, yyy, zzz, vvv,
        xfliml=0, xflimh=10,
        yfliml=0, yflimh=10,
        zfliml=0, zflimh=10,
        output_path="output_moderngl.png"
    )
    print(f'Czas renderowania: {time.time()-t0:.7f} s')
    t3 = time.time()
    render_moderngl_3d(
        xxx, yyy, zzz, vvv,
        xfliml=0, xflimh=10,
        yfliml=0, yflimh=10,
        zfliml=0, zflimh=10,
        output_path="output_moderngl.png"
    )
    print(f'Czas renderowania 2: {time.time()-t3:.7f} s')
    t3 = time.time()
    render_moderngl_3d(
        xxx, yyy, zzz, vvv,
        xfliml=0, xflimh=10,
        yfliml=0, yflimh=10,
        zfliml=0, zflimh=10,
        output_path="output_moderngl.png"
    )
    print(f'Czas renderowania 2: {time.time()-t3:.7f} s')
    t3 = time.time()
    render_moderngl_3d(
        zzz, xxx, yyy, vvv,
        xfliml=0, xflimh=10,
        yfliml=0, yflimh=10,
        zfliml=0, zflimh=10,
        output_path="output_moderngl.png"
    )
    print(f'Czas renderowania 2: {time.time()-t3:.7f} s')