#include "gui_userInputs.h"


void initialise_user_inputs(Keyboard_inputs** keyboard, Mouse_inputs** mouse){
    *mouse = (Mouse_inputs*)malloc(sizeof(Mouse_inputs));
    (*mouse)->left_held     = false;
    (*mouse)->left_clicked  = false;
    (*mouse)->right_held    = false;
    (*mouse)->right_clicked = false;
    (*mouse)->moved         = false;
    (*mouse)->x      = 0;
    (*mouse)->y      = 0;
    (*mouse)->x_prev = 0;
    (*mouse)->y_prev = 0;
    (*mouse)->dx     = 0.0f;
    (*mouse)->dy     = 0.0f;
   
    *keyboard = (Keyboard_inputs*)malloc(sizeof(Keyboard_inputs));
    memset(*keyboard, 0, sizeof(Keyboard_inputs));
}

void update_user_inputs(Keyboard_inputs* keyboard, Mouse_inputs* mouse, SDL_Event* event){
    // ~~~~~~~~~~ ready the keyboard and mouse data structures ~~~~~~~~~~
    // mouse
    mouse->x_prev         = mouse->x;
    mouse->y_prev         = mouse->y;
    Uint32 mouseState     = SDL_GetMouseState(&mouse->x, &mouse->y);
    mouse->left_held      = mouseState & SDL_BUTTON(SDL_BUTTON_LEFT);
    mouse->right_held     = mouseState & SDL_BUTTON(SDL_BUTTON_RIGHT);
    mouse->dx             = (float)(mouse->x - mouse->x_prev);
    mouse->dy             = (float)(mouse->y - mouse->y_prev);
    mouse->moved          = (mouse->dx != 0.0f || mouse->dy != 0.0f);
    mouse->left_clicked   = false;
    mouse->right_clicked  = false;
    //keyboard
    keyboard->change_happened = false;
    const Uint8 *state = SDL_GetKeyboardState(NULL);
    if (state[SDL_SCANCODE_LCTRL] || state[SDL_SCANCODE_RCTRL]) {
        if(!keyboard->ctrl.held){
            keyboard->ctrl.held       = true;
            keyboard->change_happened = true;
        }
    }
    if(state[SDL_SCANCODE_LSHIFT] || state[SDL_SCANCODE_RSHIFT]){
        if(!keyboard->shift.held){
            keyboard->shift.held      = true;
            keyboard->change_happened = true;
        }
    }
    if(state[SDL_SCANCODE_SPACE]){
        if(!keyboard->space.held){
            keyboard->space.held      = true;
            keyboard->change_happened = true;
        }
    }
    if(state[SDL_SCANCODE_RETURN]){
        if(!keyboard->enter.held){
            keyboard->enter.held      = true;
            keyboard->change_happened = true;
        }
    }

    // ~~~~~~~~~~ event polling ~~~~~~~~~~
    while (SDL_PollEvent(event)){
        switch (event->type){
            // mouse button released
            case SDL_MOUSEBUTTONUP:
                if (event->button.button == SDL_BUTTON_LEFT){
                    mouse->left_clicked = true;}
                if (event->button.button == SDL_BUTTON_RIGHT){
                    mouse->right_clicked = true;}
                break;
            // keyboard key released
            case SDL_KEYUP:
                // escape: quit
                if (event->key.keysym.sym == SDLK_ESCAPE){
                    keyboard->change_happened = true;
                    keyboard->escape.released = true;
                }
                // ctrl
                if (event->key.keysym.sym == SDLK_LCTRL || event->key.keysym.sym == SDLK_RCTRL){
                    keyboard->change_happened = true;
                    keyboard->ctrl.released   = true;
                }
                // shift
                if (event->key.keysym.sym == SDLK_LSHIFT || event->key.keysym.sym == SDLK_RSHIFT){
                    keyboard->change_happened = true;
                    keyboard->shift.released  = true;
                }
                // space
                if (event->key.keysym.sym == SDLK_SPACE){
                    keyboard->change_happened = true;
                    keyboard->space.released  = true;
                }
                // enter
                if (event->key.keysym.sym == SDLK_RETURN){
                    keyboard->change_happened = true;
                    keyboard->enter.released  = true;
                }

            default:
                break;
        }
    }
    return;
}

void update_mouse(Mouse_inputs* mouse, SDL_Event* event){
    return;
}

void update_keyboard(Keyboard_inputs* keyboard, SDL_Event* event){
    return;
}