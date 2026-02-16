echo Starting GPU Check > debug_gpu_check.log
"C:\Program Files\MATLAB\R2025a\bin\matlab.exe" -batch "cd('c:\Users\udoo_w2\Desktop\work_traffic\matlab_multigpus'); run('check_toolboxes.m');" >> debug_toolbox_check.log 2>&1
