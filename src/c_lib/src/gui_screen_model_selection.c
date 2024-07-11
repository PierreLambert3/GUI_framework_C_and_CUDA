
#include "gui_screen_model_selection.h"

void renderModelSelectionScreen(DrawerThatDraws* drawer, float render_work_to_sleep_ratio){
    // draw in neon a rectangle frame
    set_colour(drawer, CYAN);
    Drawer_draw_neon_line(drawer, 10, 10, WINDOW_W - 10, 10);
    Drawer_draw_neon_line(drawer, WINDOW_W - 10, 10, WINDOW_W - 10, WINDOW_H - 10);
    Drawer_draw_neon_line(drawer, WINDOW_W - 10, WINDOW_H - 10, 10, WINDOW_H - 10);
    Drawer_draw_neon_line(drawer, 10, WINDOW_H - 10, 10, 10);

    return;
}