
#include "gui_screen_model_selection.h"


void ModelSelectionScreen_init(ModelSelectionScreen** model_selection_screen_ptrAdrr){
    // allocate memory for ModelSelectionScreen
    *model_selection_screen_ptrAdrr = (ModelSelectionScreen*) malloc(sizeof(ModelSelectionScreen));
    ModelSelectionScreen* model_selection_screen = *model_selection_screen_ptrAdrr;

    // calculate the x and y coordinates for centering the grid
    uint32_t gridWidth = 1000; // increase the grid width to 1000
    uint32_t gridHeight = 400;
    uint32_t gridX = (WINDOW_W - gridWidth) / 2;
    uint32_t gridY = (WINDOW_H - gridHeight) / 2;

    // initialise the 4 buttons in a 2x2 grid
    uint32_t buttonWidth = 500; // increase the button width to 500
    uint32_t buttonHeight = 200;
    uint32_t buttonGap = 2; // add a gap of 2 pixels between each button

    TogglableButton_init(&(model_selection_screen->button_1), gridX, gridY, buttonWidth, buttonHeight, MODEL_1_NAME);
    TogglableButton_init(&(model_selection_screen->button_2), gridX + buttonWidth + buttonGap, gridY, buttonWidth, buttonHeight, MODEL_2_NAME);
    TogglableButton_init(&(model_selection_screen->button_3), gridX, gridY + buttonHeight + buttonGap, buttonWidth, buttonHeight, MODEL_3_NAME);
    TogglableButton_init(&(model_selection_screen->button_4), gridX + buttonWidth + buttonGap, gridY + buttonHeight + buttonGap, buttonWidth, buttonHeight, MODEL_4_NAME);


}

void renderModelSelectionScreen(ModelSelectionScreen* screen, GenericDrawer* drawer){
    // draw a rectangle frame 
    set_colour(drawer, AMBER);
    GenericDrawer_draw_rect(drawer, 8, 1, WINDOW_W - 16, WINDOW_H - 2);

    // draw the title: " Model selection", centered
    GenericDrawer_draw_text(drawer, WINDOW_W/2 - 100, 250, "Model selection");

    // draw the 4 buttons
    TogglableButton_draw(&(screen->button_1), drawer);
    TogglableButton_draw(&(screen->button_2), drawer);
    TogglableButton_draw(&(screen->button_3), drawer);
    TogglableButton_draw(&(screen->button_4), drawer);

    return;
}