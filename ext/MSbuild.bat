@echo off
REM for pytorch 1.0 amd64
REM Installed VS2017 Community
set "VS150COMNTOOLS=D:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\"
set DISTUTILS_USE_SDK=1
REM call "%VS140COMNTOOLS%\..\..\VC\vcvarsall.bat" x64
call "%VS150COMNTOOLS%\vcvarsall.bat" x64
REM D:\CONDA is root of anaconda
REM anaconda\Scripts should already be added into Path
call activate D:\CONDA
call activate pytorch
python setup.py install
pause