# https://stackoverflow.com/questions/46141631/running-matlab-using-python-gives-no-module-named-matlab-engine-error
# You need to install the Matlab Engine for Python,
# and it cannot be installed using pip. Try the instructions listed here. I have listed the instructions briefly below:
#
# Make sure you have Python in your PATH.
# Find the Matlab root folder. You can use the matlabroot command within Matlab to find it.
# Go to the Matlab root folder in the command line.
# cd "matlabroot\extern\engines\python" (In Windows)
# python setup.py install
import matlab.engine

eng = matlab.engine.start_matlab()
eng.calc_parts_obb(nargout=0)
eng.quit()
