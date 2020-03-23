#!/bin/bash
../../../build_NimbleSM-UQ/src/NimbleSM_Serial uq_wave_in_bar.in
scrape_1D.py uq_wave_in_bar.serial.e
analytic_solution.py x.dat t.dat > solution.log
paste solution.dat uq_wave_in_bar.dat > u.dat
