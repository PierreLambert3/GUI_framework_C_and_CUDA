That's a minimalist loop and file structure for a GUI or game written in C, with possibility to launch CUDA kernels on the algorithmic side. This was built on my own free time.

Use the makefile for compilation: I suffered enough building it, let my sacrifice not be in vain. The compiler is set for max speed, including unsafe and approximate arithmetics: don't run your space program with this config.

Requirements: GCC, NVCC (cuda dev), SDL2 including TTF (for fonts) and gfx-dev (increases the rendering possibilities).

To have an actual algorithm running, just launch a pthread from GUI_logic.c and allow communications between GUI and the worker thread by creating your own data structure. Of course, don't forget to use mutexes (mutii?) to keep things safe.

Slava Ukraini, may invaders get what they deserve.
