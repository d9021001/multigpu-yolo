matlab -batch "disp('MATLAB is working'); quit;" > test_matlab.log 2>&1
if %errorlevel% neq 0 echo MATLAB command failed >> test_matlab.log
