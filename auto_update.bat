@echo off
tar.exe -a -c -f kaggle/fairmot.zip src/*
kaggle datasets version -p kaggle -m "update"