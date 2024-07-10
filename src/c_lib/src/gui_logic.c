#include "gui_logic.h"

// --------- periodic timer ---------
void PeriodicTimer_init(PeriodicTimer** timer_ptrAdrr, uint32_t period){
    // allocate memory for PeriodicTimer
    *timer_ptrAdrr = (PeriodicTimer*) malloc(sizeof(PeriodicTimer));
    PeriodicTimer* timer = *timer_ptrAdrr;

    timer->timestamp = SDL_GetTicks(); // SDL_Init(SDL_INIT_VIDEO) must have been called before this
    timer->elapsed   = 0u;
    timer->period    = period;
}

bool PeriodicTimer_update(PeriodicTimer* timer){
    uint32_t now    = SDL_GetTicks();
    timer->elapsed += now - timer->timestamp;
    if(timer->elapsed >= timer->period){
        timer->elapsed   = 0u;
        timer->timestamp = now;
        return true;
    }
    return false;
}

// ambiant sound

// --------- gui logic ---------
void GuiLogic_init(GuiLogic** gui_logic_ptrAdrr, uint32_t rand_state){

    printf("obs to embedding mapping: using SOM (both at neuron and brain level)\n");

    // allocate memory for GuiLogic
    *gui_logic_ptrAdrr  = (GuiLogic*) malloc(sizeof(GuiLogic));
    GuiLogic* gui_logic = *gui_logic_ptrAdrr;

    // allocate memory & initialise user inputs
    initialise_user_inputs(&(gui_logic->keyboard), &(gui_logic->mouse));

    // allocate memory & initialise the drawer, which in term initialises SDL
    Drawer_init(&(gui_logic->drawer));

    // initialise GuiLogic internals
    gui_logic->isRunning      = true;
    gui_logic->rand_state     = rand_state;
    gui_logic->current_screen = SCREEN_MODEL_SELECTION;

    // initialise the periodic timer (SDL need to be initialised beforehand)
    PeriodicTimer_init(&(gui_logic->timer_0), 1000u); // 1000ms period
}

void GuiLogic_run(GuiLogic* gui_logic){
    uint32_t target_frame_time = 1000 / 16; // 16fps
    while (gui_logic->isRunning){
        uint32_t frame_start = SDL_GetTicks();

        // poll user inputs, and save state in the user inputs data structures
        update_user_inputs(gui_logic->keyboard, gui_logic->mouse, &(gui_logic->SDL_events));

        /* // react to user inputs
        GuiLogic_handle_events(GuiLogic* gui_logic);

        // render the screen (calls the appropriate rendering functions from drawer)
        GuiLogic_render_screen(GuiLogic* gui_logic);
        */
        
        // cap the frame rate
        uint32_t frame_duration = SDL_GetTicks() - frame_start;
        if (frame_duration < target_frame_time) {
            SDL_Delay(target_frame_time - frame_duration);}

        // check if the user has closed the window
        if(gui_logic->keyboard->escape.released){
            gui_logic->isRunning = false;
        }
    }

    //cleanup
    // Drawer_destroy(gui_logic->drawer);
}


