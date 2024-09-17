## Setup Instructions

Instructions below is to get markrouber's puzzle bot to run, in particular ryan's puzzle-solver, which is the library that does the actual puzzle matching work.  

#### Assumptions

- miniconda is installed on a Mac OS 14 

- Terminal is setup and working



#### Instructions:

Create environment

```bash
conda create -y -n puzzle python=3.10
conda activate puzzle
```

Clone repository:

```bash
git clone  https://github.com/markroberyoutube/puzzle_bot.git
cd puzzle_bot
git clone https://github.com/iancharnas/ryan-puzzle-solver.git
```

Install packages:

```bash
pip install exif
pip install argparse
conda install -c conda-forge opencv
pip install PyQt5
pip install scipy
pip install pyserial

```

This should allow you to run the **run_batch.py** command:

```bash
cd ryan-puzzle-solver/src
python run_batch.py
```








