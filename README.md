
Trevor

Jackson

Jain Iftesam

William Ilkanic

If 
$ ./run
-bash: ./run: cannot execute: required file not found

Then 
$ dos2unix run
dos2unix: converting file run to Unix format...


If 
$ ./run install doesn't work

Then
$ python 3 -m venv venv
$ cd venv
$ source bin/activate
$ cd ..
$ ./run install