cd c:\Users\udoo_w2\Desktop\work_traffic\matlab_multigpus
mkdir xFolder
if %errorlevel% neq 0 echo Failed to create xFolder >> creation_log.txt
mkdir cFolder
if %errorlevel% neq 0 echo Failed to create cFolder >> creation_log.txt
dir >> creation_log.txt
