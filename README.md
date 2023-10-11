<h1 align = "center"> Random forest analysis (MAIO) </h1> <br>
Random forest analysis of air quality for the MAIO course of 2023-2024.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation
All required packages can be installed by running `pip install -r requirements.txt` in a Python-capable terminal window.

## Usage
Everything should work out of the box, when run in something like Spyder. This will generate all results and graphs used in the accompanying report (to be added here?). Some of the function (like `clean_data(...)` could in principle be used on other data, but I advise caution. This code has not been written to be reusable or easily adaptable, and will take a while to understand.

### Note
In line 678, a function is run which uses the `concurrent.futures` package to multithread, drastically increasing computation speed (on my machine - 16 cores). There is a good chance that it will take substantially longer on other devices, so feel free to comment this line. The results of this function are as shown here:

<p align="center">
    <img alt="R2 score" title="Trees vs R2 score" src="https://github.com/Renseck/MAIO-RandomForest/blob/main/results/o3_trees_vs_r2_100.jpg" width="450">
</p>

## License

[MIT](https://choosealicense.com/licenses/mit/) license, Copyright (c) 2023 Rens van Eck.
