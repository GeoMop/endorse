import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def el_boundary_line(nodes):
    l = [(x,y) for x,y,z in nodes]
    l.append(l[0])
    return l

def mesh_2d(ax, mesh:'GmshIO'):
    # plot points
    ax.set_aspect('equal')
    nodes = mesh.nodes
    segments = [
         el_boundary_line((nodes[i] for i in i_nodes))
         for type, tags, i_nodes in mesh.elements.values()
    ]
    ax.add_collection(LineCollection(segments, linewidths=0.5, zorder=0))
    X, Y = zip(*( (x,y) for x,y,z in nodes.values()))
    ax.scatter(X, Y, color='red', zorder=1, s=1)

def mesh_3d(mesh:'GmshIO'):
    segments = [
         el_boundary_line((nodes[i] for i in i_nodes))
         for type, tags, i_nodes in mesh.elements.values()
    ]

    trace1=go.Scatter3d(x=Xe, y=Ye, z=Ze, mode='lines', line=dict(color='rgb(125,125,125)', width=1),hoverinfo='none')

trace2=go.Scatter3d(x=Xn, y=Yn, z=Zn, mode='markers', name='actors',
                   marker=dict(symbol='circle', size=6, color=group, colorscale='Viridis',
                      line=dict(color='rgb(50,50,50)', width=0.5)), text=labels, hoverinfo='text')

axis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title='')

layout = go.Layout(
         title="Network of coappearances of characters in Victor Hugo's novel<br> Les Miserables (3D visualization)",
         width=1000,
         height=1000,
         showlegend=False,
         scene=dict(
             xaxis=dict(axis),
             yaxis=dict(axis),
             zaxis=dict(axis),
        ))

data=[trace1, trace2]

fig=go.Figure(data=data, layout=layout)

iplot(fig, filename='Les-Miserables')