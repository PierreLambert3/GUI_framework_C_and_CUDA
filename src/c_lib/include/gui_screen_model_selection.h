#ifndef GUI_SCREEN_MODEL_SELECTION_H
#define GUI_SCREEN_MODEL_SELECTION_H

// include the necessary headers
#include "gui_render.h"

typedef struct {
    TogglableButton button_1;
    TogglableButton button_2;
    TogglableButton button_3;
    TogglableButton button_4;
} ModelSelectionScreen;

void ModelSelectionScreen_init(ModelSelectionScreen** model_selection_screen_ptrAdrr);

void renderModelSelectionScreen(ModelSelectionScreen* screen, GenericDrawer* drawer);

#endif // GUI_SCREEN_MODEL_SELECTION_H