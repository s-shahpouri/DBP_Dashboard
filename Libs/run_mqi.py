import os
import shutil
import subprocess

def run_mqi(root_mqi_binary, name_mqi_binary, input_parameters, delimeter = " ", nr_beams = 0, rerun = False):
    print('run')
    dir_output = input_parameters['OutputDir']
    filename_mqi_input_file = "input_file.in"
    path_mqi_input_file = os.path.join(dir_output, filename_mqi_input_file)
    if nr_beams:
        beam_nrs = range(1, nr_beams + 1)
    else:
        beam_nrs = [0]

    for beam_nr in beam_nrs:
        input_parameters['BeamNumbers'] = beam_nr

        with open(path_mqi_input_file, 'w') as f:
            for key, value in input_parameters.items():
                f.write(key + delimeter + str(value) + '\n')
        orig_working_dir = os.getcwd()
        os.chdir(root_mqi_binary)
#        subprocess.check_call("./" + name_mqi_binary + f" \"{path_mqi_input_file}\" >  +xxx.log", shell=True)
        if rerun:
            for i in range(3):
                try:
                    subprocess.check_call(f"./{name_mqi_binary} \"{path_mqi_input_file}\" > {os.path.dirname(path_mqi_input_file)}/log.txt", shell=True)
                    break
                except subprocess.CalledProcessError as err:
                    if i < 5:
                        print("Run failed. Trying again.")
                        shutil.move(f"{os.path.dirname(path_mqi_input_file)}/log.txt", f"{os.path.dirname(path_mqi_input_file)}/log_error.txt")
                        print(err)
                    else: 
                        print("Run failed")
                        os.chdir(orig_working_dir)
                        return "FAILED"
            
        else:
            subprocess.check_call(f"./{name_mqi_binary} \"{path_mqi_input_file}\" > {os.path.dirname(path_mqi_input_file)}/log.txt", shell=True)
        # os.system("./" + name_mqi_binary + f" \"{path_mqi_input_file}\"")
        os.chdir(orig_working_dir)











