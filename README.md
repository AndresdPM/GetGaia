# GetGaia
GetGaia is a software tool that aids on downloading data from the Gaia archive and select member stars from stellar systems.

## License and Referencing
This code is released under a BSD 2-clause license.

If you find this code useful for your research, please mention it in your acknowledgements or cosider citing del Pino et al. 2021, Martínez-García et al. 2021. Part of the GetGaia membership selection is based on Multi-Gaussian Expansions (MGEs) modelling. Please consider also citing Cappellari (2002, MNRAS, 333, 400).

## Features

GetGaia includes lots of useful features:

* Search of objects based on names.
* Automatic screening out of poorly measured stars.
* Interactive selection of members.
* Voronoi tesellation with sigma clipping rejection of contaminants.
* Statistics about the systemic proper motions of the object.
* Automatic generation of plots.

that download data from the Gaia archive, screen out select member stars from stellar systems. It alos  to download Gaia data and select members stars, and screening out poorly measured stars through a series of nested criteria.

## Usage

GetGaia can be used in automatic or interactive form. For example, the user can interactively retrieve all the data available in the Sculptor dwarf spheroidal galaxy, and a selection of members with:

$ GetGaia --name "Sculptor dSph"

Or in the NGC5053 globular cluster:

$ GetGaia --name "NGC5053"

Adding the option "--silent True" forces GetGaia to adopt all the default parameters (not recommended).

Please read the help to know more about GetGaia options.

$ GetGaia --help

## Requirements

This code makes use of Astropy, Pandas, Numpy, Scipy, Matplotlib, and Sklearn among others. The code will require the installation of the gaiadr3_zeropoint package: https://pypi.org/project/gaiadr3-zeropoint/
