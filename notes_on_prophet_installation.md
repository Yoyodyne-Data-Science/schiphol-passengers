### Not so straight forward getting fbprophet installed

First run:
`conda install libpython m2w64-toolchain -c msys2`

Then:
`conda install pystan -c conda-forge`

Then go get prophet:
`conda install -c conda-forge fbprophet`

(N.B. some people found that a simple `conda config --set ssl_verify no` solved it)

All comes down to installing pystan: https://pystan.readthedocs.io/en/latest/windows.html#windows

Problem with Easter?

https://github.com/dr-prodigy/python-holidays/issues/277