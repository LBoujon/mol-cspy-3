Several different environments are available, depending on the user's needs.
The default/recommend environment is flexible and is named cspy.yml

# Python
Python 3.11 is now the recommend version of Python for mol-CSPy. 
All associated environments may be found inside the `Python3.11` directory.

# Environment Flexibility
We provide two types of environments, Flexible and Complete.
Complete environments list all dependencies and specific versions of those dependencies.
The filename lists the OS (e.g. RHEL7) that the environment was created on, and the date
that the environment was created. These environments should always work on the 
associated OS, but may not work on other OSs.
Flexible environments are more likely to work on different OSs and allow for the use of
newer versions of software as they are released.

# CSD Python API
All Python 3.11 environments support the CSD python api, unless specified in the name.

# MACE
MACE and it's dependencies are large, and complex to install. Therefore it is not included by default.
Where an environment includes MACE, the filename includes `-MACE-`.
Environments that do include MACE have inflexible MACE dependencies.
The user may be able to access a newer version of MACE and it's dependencies by installing
a flexible environment that doesn't include MACE, and then running:
pip install mace-torch --upgrade-strategy only-if-needed

# Clusters
Some clusters have particular dependency requirements.
For these clusters we provide non-flexible environments in the 'Clusters' directory.

# Installation
conda env create -f conda/Python3.11/Flexible/cspy.yml
