echo Starting Network Check > debug_net_check.log
"C:\Program Files\MATLAB\R2025a\bin\matlab.exe" -batch "cd('c:\Users\udoo_w2\Desktop\work_traffic\matlab_multigpus'); run('check_networks.m');" >> debug_net_check.log 2>&1
