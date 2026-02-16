echo Starting CSV Debug > debug_csv.log
"C:\Program Files\MATLAB\R2025a\bin\matlab.exe" -batch "cd('c:\Users\udoo_w2\Desktop\work_traffic\matlab_multigpus'); run('test_csv_read.m');" >> debug_csv.log 2>&1
