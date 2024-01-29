import os
from endorse.Bukov2.l5_mesh import make_mesh
from endorse import common
script_dir = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    make_mesh(script_dir, "./Bukov2_mesh.yaml")