echo Starting Debug > debug_info_log.txt
"C:\Program Files\MATLAB\R2025a\bin\matlab.exe" -batch "cd('c:\Users\udoo_w2\Desktop\work_traffic\matlab_multigpus'); run('check_debug.m');" >> debug_info_log.txt 2>&1
