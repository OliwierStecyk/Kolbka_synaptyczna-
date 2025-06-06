
'''
def plot_task1(rrr, NT_dens, z_int, output_path):
    fig = go.Figure(data=go.Scatter(
        x=rrr,
        y=NT_dens,
        mode='markers',
        marker=dict(
            color=z_int,
            colorscale='Hot',
            cmin=-1.6,
            cmax=1.6,
            size=8,
            line=dict(width=0)
        )
    ))
    fig.update_layout(
        xaxis=dict(range=[-0.00, 0.65], gridcolor='rgba(200,200,200,0.3)'),
        yaxis=dict(range=[280.0, 520.0], gridcolor='rgba(200,200,200,0.3)'),
        plot_bgcolor='white'
    )
    fig.write_image(output_path)'''
'''
def ScattWorker(q):
    while True:
        rrr, NT_dens, z_int, output_path = q.get()
        if rrr is None:
            break
        plot_task1(rrr, NT_dens, z_int, output_path)
'''

# plot_worker.py
import matplotlib
matplotlib.use('Agg')  # Must be before pyplot import
import matplotlib.pyplot as plt

def plot_task(rrr, NT_dens, z_int, output_path):
    plt.figure()
    cmap_hot = plt.get_cmap("hot")
    plt.grid(True)
    plt.scatter(rrr, NT_dens, cmap=cmap_hot, c=z_int, vmin=-1.6, vmax=+1.6, s=30.0, lw=0)
    plt.xlim(-0.00, 0.65)
    plt.ylim(280.0, 520.0)
    plt.savefig(output_path)
    plt.close()

def ScattWorker(q):
    while True:
        rrr, NT_dens, z_int, output_path = q.get()
        if rrr is None:
            break
        plot_task(rrr, NT_dens, z_int, output_path)




