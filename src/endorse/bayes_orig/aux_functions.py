import os
import shutil


def force_mkdir(path, force=False):
    """
    Make directory 'path' with all parents,
    remove the leaf dir recursively if it already exists.
    :param path: path to directory
    :param force: if dir already exists then remove it and create new one
    :return: None
    """
    if force:
        if os.path.isdir(path):
            shutil.rmtree(path)
    os.makedirs(path, mode=0o775, exist_ok=True)


def substitute_placeholders(file_in, file_out, params):
    """
    Substitute for placeholders of format '<name>' from the dict 'params'.
    :param file_in: Template file.
    :param file_out: Values substituted.
    :param params: { 'name': value, ...}
    """
    used_params = []
    with open(file_in, 'r') as src:
        text = src.read()
    for name, value in params.items():
        placeholder = '<%s>' % name
        n_repl = text.count(placeholder)
        if n_repl > 0:
            used_params.append(name)
            text = text.replace(placeholder, str(value))
    with open(file_out, 'w') as dst:
        dst.write(text)
    return used_params


def check_conv_reasons(log_fname):
    with open(log_fname, "r") as f:
        # print(log_fname)
        for line in f:
            # check HM iterations:
            if "Nonlinear solver did not converge." in line:
                print(line)
                return -100

            # check linear solvers conv. reason:
            tokens = line.split(" ")
            try:
                i = tokens.index('convergence')
                if tokens[i + 1] == 'reason':
                    value = tokens[i + 2].rstrip(",")
                    conv_reason = int(value)
                    if conv_reason < 0:
                        print("Some Linear Solver failed to converge (conv_reason: {})".format(conv_reason))
                        return conv_reason
            except ValueError:
                continue
    return 0


def check_gmsh_log(lines):
    """
    Search for "No elements in volume" message -> could not mesh the volume -> empty mesh.
    # PLC Error:  A segment and a facet intersect at point (-119.217,65.5762,-40.8908).
    #   Segment: [70,2070] #-1 (243)
    #   Facet:   [3147,9829,13819] #482
    # Info    : failed to recover constrained lines/triangles
    # Info    : function failed
    # Info    : function failed
    # Error   : HXT 3D mesh failed
    # Error   : No elements in volume 1
    # Info    : Done meshing 3D (Wall 0.257168s, CPU 0.256s)
    # Info    : 13958 nodes 34061 elements
    # Error   : ------------------------------
    # Error   : Mesh generation error summary
    # Error   :     0 warnings
    # Error   :     2 errors
    # Error   : Check the full log for details
    # Error   : ------------------------------
    """
    empty_volume_error = "No elements in volume"
    res = [line for line in lines if empty_volume_error in line]
    if len(res) != 0:
        raise Exception("GMSH error - No elements in volume")
