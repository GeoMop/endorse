import os
import sys
# import shutil
import subprocess
# import matplotlib.pyplot as plt
import yaml

from endorse import common
from flow123d_simulation import generate_time_axis


def print_num(num, type, ndigits):
    # dynamic string format template
    # string = "{:{fill}{align}{width}}"
    # num = "{:{align}{width}.{precision}f}"
    form="{:." + str(ndigits) + type + "}"
    if isinstance(num, list):
        res = "["
        for v in num[:-1]:
            res = res + form.format(v) + ", "
        res = res + form.format(num[-1]) + "]"
        return res
    else:
        return form.format(num)

if __name__ == "__main__":

    import json
    import flow_wrapper
    script_dir = os.path.dirname(os.path.abspath(__file__))

    output_dir = None
    len_argv = len(sys.argv)
    assert len_argv > 1, "Specify output directory!"
    if len_argv > 1:
        output_dir = os.path.abspath(sys.argv[1])

    config_dict = flow_wrapper.setup_config(output_dir)

    report_dir = os.path.join(output_dir, "report")
    if not os.path.isdir(report_dir):
        os.mkdir(report_dir)

    tex_template = os.path.join(script_dir, "DOC/bayes_results_template.tex")
    tex_file = os.path.join(report_dir, "bayes_results_report.tex")
    # shutil.copyfile(tex_template, tex_file)

    bayes_output_yaml = os.path.join(output_dir, "saved_samples/config_mcmc_bayes/output.yaml")
    params = dict()
    with open(bayes_output_yaml, "r") as f:
        loaded_yaml = yaml.load(f, yaml.CSafeLoader)

        params["name"] = os.path.basename(output_dir).replace("_", " ")
        MH = loaded_yaml["samplers_list"][0]
        DAMHsmu = loaded_yaml["samplers_list"][1]

        params["MH_N"] = MH["N samples [a, r, pr, all]"]
        params["MH_acceptance"] = print_num(MH["acceptance ratio [a/r, a/all]"], 'f', 3)
        params["MH_proposal_std"] = MH["proposal_std"]

        params["DAMHsmu_N"] = DAMHsmu["N samples [a, r, pr, all]"]
        params["DAMHsmu_acceptance"] = print_num(DAMHsmu["acceptance ratio [a/r, a/all]"], 'f', 3)
        params["DAMHsmu_proposal_std"] = DAMHsmu["proposal_std"]
        best_fit = loaded_yaml["best_fit_L2"]["parameters"]


    common.substitute_placeholders(tex_template, tex_file, params)

    os.chdir(output_dir)
    arguments = ["pdflatex", "-interaction=nonstopmode", "-output-directory", report_dir, tex_file]
    stdout_path = os.path.join(report_dir, "tex_stdout")
    stderr_path = os.path.join(report_dir, "tex_stderr")
    print("Running LaTeX: ", " ".join(arguments))
    with open(stdout_path, "w") as stdout:
        with open(stderr_path, "w") as stderr:
            completed = subprocess.run(arguments, stdout=stdout, stderr=stderr)
    print("Exit status: ", completed.returncode)
    # os.system(" ".join(arguments))

    # is best fit accepted or rejected?
    num = print_num(best_fit[0], 'e', 14)[:14]
    # res = os.system("grep -rn saved_samples -e \"" + num +"\"")
    res = subprocess.check_output(["grep -rn saved_samples -e \"" + num +"\""], shell=True)
    res = str(res)
    # print(res)
    if "rejected" in res:
        print("BEST FIT L2 is REJECTED")
    elif "accepted" in res:
        print("BEST FIT L2 is ACCEPTED")







