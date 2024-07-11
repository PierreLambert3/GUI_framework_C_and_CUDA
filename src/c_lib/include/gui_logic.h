#ifndef GUI_LOGIC_H
#define GUI_LOGIC_H

#include "gui_userInputs.h"
#include "gui_render.h"
#include "gui_screen_model_selection.h"

#define SCREEN_MODEL_SELECTION 0u
#define SCREEN_SANDBOX         1u
#define SCREEN_GA_HYPERPARAMS  2u
#define SCREEN_GA_RUNNING      3u

// --------- periodic timer  ---------
typedef struct { // (don't run this for more that 49days, if you do, you'll have to change the type of timestamp to uint64_t)
    uint32_t timestamp; // Timestamp at last iteration 
    uint32_t elapsed;   // Elapsed since period start
    uint32_t period;    // Period in milliseconds 
} PeriodicTimer;

void PeriodicTimer_init(PeriodicTimer** timer_ptrAdrr, uint32_t period);
bool PeriodicTimer_update(PeriodicTimer* timer, uint32_t frame_time_now); // returns true if the timer has elapsed

// --------- gui logic ---------
typedef struct {
    // user inputs
    SDL_Event        SDL_events;
    Keyboard_inputs* keyboard;
    Mouse_inputs*    mouse;

    // general drawing manager
    GenericDrawer* drawer;
    // screens
    ModelSelectionScreen* model_selection_screen;

    // internals
    uint32_t     rand_state;
    bool         isRunning;
    uint8_t      current_screen;

    // timers: used for periodic events
    PeriodicTimer* timer_0;
} GuiLogic;

// malloc and initialise the GuiLogic struct. GuiLogic** because we want to malloc the struct in the function
void GuiLogic_init(GuiLogic** gui_logic_ptrAdrr, uint32_t rand_state);
// launches the program proper
void GuiLogic_run(GuiLogic* gui_logic);
// handle the events themselves
void GuiLogic_handle_events(GuiLogic* gui_logic);
// render the screen, calls the appropriate rendering functions from drawer
void GuiLogic_render_screen(GuiLogic* gui_logic, float render_work_to_sleep_ratio);
// possibly sleep to limit the frame rate
void GuiLogic_cap_frame_rate(uint32_t frame_start, uint32_t target_frame_time, float* render_work_to_sleep_ratio);

// events for the model selection screen
void GuiLogic_handle_events_model_selection_screen(GuiLogic* gui_logic, ModelSelectionScreen* model_selection_screen);

#endif // GUI_LOGIC_H