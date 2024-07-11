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

bool PeriodicTimer_update(PeriodicTimer* timer, uint32_t frame_time_now){
    timer->elapsed += frame_time_now - timer->timestamp;
    if(timer->elapsed >= timer->period){
        timer->elapsed   = 0u;
        timer->timestamp = frame_time_now;
        return true;
    }
    return false;
}

// --------- gui logic ---------
void GuiLogic_cap_frame_rate(uint32_t frame_start, uint32_t target_frame_time, float* render_work_to_sleep_ratio){
    uint32_t frame_duration = SDL_GetTicks() - frame_start;
    if (frame_duration < target_frame_time) {
        uint32_t frame_sleep_time = target_frame_time - frame_duration;
        SDL_Delay(frame_sleep_time);
    }
    *render_work_to_sleep_ratio = 0.95f*(*render_work_to_sleep_ratio) + 0.05f*(float)frame_duration / target_frame_time;
}

void GuiLogic_init(GuiLogic** gui_logic_ptrAdrr, uint32_t rand_state){

    // allocate memory for GuiLogic
    *gui_logic_ptrAdrr  = (GuiLogic*) malloc(sizeof(GuiLogic));
    GuiLogic* gui_logic = *gui_logic_ptrAdrr;

    // allocate memory & initialise user inputs
    initialise_user_inputs(&(gui_logic->keyboard), &(gui_logic->mouse));

    // allocate memory & initialise the GenericDrawer, which in term initialises SDL
    GenericDrawer_init(&(gui_logic->drawer));
    // allocate memory & initialise the ModelSelectionScreen
    ModelSelectionScreen_init(&(gui_logic->model_selection_screen));

    // initialise GuiLogic internals
    gui_logic->isRunning      = true;
    gui_logic->rand_state     = rand_state;
    gui_logic->current_screen = SCREEN_MODEL_SELECTION;

    // initialise the periodic timer (SDL need to be initialised beforehand)
    PeriodicTimer_init(&(gui_logic->timer_0), 1000u); // 1000ms period

    printf("Render using ");
    set_console_colour_info();
    printf("%s", RENDER_USING_GPU ? "GPU\n" : "CPU\n");
    reset_console_colour();
}

void GuiLogic_handle_events_model_selection_screen(GuiLogic* gui_logic, ModelSelectionScreen* model_selection_screen){
    // hovering on the buttons
    model_selection_screen->button_1.hovered = TogglableButton_is_hovered(&(model_selection_screen->button_1), gui_logic->mouse->x, gui_logic->mouse->y);
    model_selection_screen->button_2.hovered = TogglableButton_is_hovered(&(model_selection_screen->button_2), gui_logic->mouse->x, gui_logic->mouse->y);
    model_selection_screen->button_3.hovered = TogglableButton_is_hovered(&(model_selection_screen->button_3), gui_logic->mouse->x, gui_logic->mouse->y);
    model_selection_screen->button_4.hovered = TogglableButton_is_hovered(&(model_selection_screen->button_4), gui_logic->mouse->x, gui_logic->mouse->y);

    model_selection_screen->button_1.clicking = model_selection_screen->button_1.hovered && gui_logic->mouse->left_held;
    model_selection_screen->button_2.clicking = model_selection_screen->button_2.hovered && gui_logic->mouse->left_held;
    model_selection_screen->button_3.clicking = model_selection_screen->button_3.hovered && gui_logic->mouse->left_held;
    model_selection_screen->button_4.clicking = model_selection_screen->button_4.hovered && gui_logic->mouse->left_held;
   
    if(gui_logic->mouse->left_clicked){
        if(model_selection_screen->button_1.hovered){
            die("don't click me!");
        }
        if(model_selection_screen->button_2.hovered){
            printf("button 2 clicked\n");
        }
        if(model_selection_screen->button_3.hovered){
            printf("button 3 clicked\n");
        }
        if(model_selection_screen->button_4.hovered){
            printf("button 4 clicked\n");
        }
    }

    /* // handle the events for the 4 buttons
    if(model_selection_screen->button_1.hovered){
        if(gui_logic->mouse->left_clicked){
            die("heeeh");
        }
    }
    if(model_selection_screen->button_2.hovered){
        if(gui_logic->mouse->left_clicked){
            printf("button 2 clicked\n");
        }
    }
    if(model_selection_screen->button_3.hovered){
        if(gui_logic->mouse->left_clicked){
            printf("button 3 clicked\n");
        }
    }
    if(model_selection_screen->button_4.hovered){
        if(gui_logic->mouse->left_clicked){
            printf("button 4 clicked\n");
        }
    } */
}

void GuiLogic_handle_events(GuiLogic* gui_logic){
    switch (gui_logic->current_screen){
        case SCREEN_MODEL_SELECTION:
            GuiLogic_handle_events_model_selection_screen(gui_logic, gui_logic->model_selection_screen);
            break;
        default:
            die("GuiLogic_render_screen: unknown screen");
            break;
    }
}

void GuiLogic_render_screen(GuiLogic* gui_logic, float render_work_to_sleep_ratio){
    GenericDrawer_get_ready_to_draw(gui_logic->drawer);
    switch (gui_logic->current_screen){
        case SCREEN_MODEL_SELECTION:
            renderModelSelectionScreen(gui_logic->model_selection_screen, gui_logic->drawer);
            break;
        default:
            die("GuiLogic_render_screen: unknown screen");
            break;
    }
    GenericDrawer_finalise_drawing(gui_logic->drawer, render_work_to_sleep_ratio);

}

void GuiLogic_run(GuiLogic* gui_logic){
    uint32_t target_frame_time          = 1000 / 16; // 16fps
    float    render_work_to_sleep_ratio = 1.0f;
    while (gui_logic->isRunning){
        uint32_t frame_start = SDL_GetTicks();

        // update the periodic timer
        bool period0_end = PeriodicTimer_update(gui_logic->timer_0, frame_start);

        // poll user inputs, and save state in the user inputs data structures
        update_user_inputs(gui_logic->keyboard, gui_logic->mouse, &(gui_logic->SDL_events));

        // react to user inputs
        GuiLogic_handle_events(gui_logic);

        // render the screen (calls the appropriate rendering functions from drawer)
        GuiLogic_render_screen(gui_logic, render_work_to_sleep_ratio);

        // cap the frame rate
        GuiLogic_cap_frame_rate(frame_start, target_frame_time, &render_work_to_sleep_ratio);

        // check if the user has closed the window
        if(gui_logic->keyboard->escape.released){
            gui_logic->isRunning = false;}
    }

    //cleanup
    // Drawer_destroy(gui_logic->drawer);
}


