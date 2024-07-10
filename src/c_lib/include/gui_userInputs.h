#ifndef GUI_USERINPUTS_H
#define GUI_USERINPUTS_H

#include <SDL.h>
#include <stdbool.h>
#include "system.h"

typedef struct {
    bool held;
    bool released;
} Keyboard_keys;

typedef struct {
    bool  left_held;    // button is held down
    bool  left_clicked; // click = on release
    bool  right_held;
    bool  right_clicked;
    bool  moved; // mouse moved since last check
    int   x;
    int   y;
    int   x_prev;
    int   y_prev;
    float dx;
    float dy;
} Mouse_inputs;

typedef struct {
    bool change_happened;
    Keyboard_keys ctrl;
    Keyboard_keys shift;
    Keyboard_keys space;
    Keyboard_keys enter;
    Keyboard_keys escape;
} Keyboard_inputs;

// allocated and initialise the keyboard and mouse inputs
void initialise_user_inputs(Keyboard_inputs** keyboard, Mouse_inputs** mouse);
// poll user inputs and fill the corresponding data structures
void update_user_inputs(Keyboard_inputs* keyboard, Mouse_inputs* mouse, SDL_Event* event);


#endif // GUI_USERINPUTS_H