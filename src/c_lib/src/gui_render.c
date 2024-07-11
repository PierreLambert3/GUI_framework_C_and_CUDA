#include "gui_render.h"
#include "gui_screen_model_selection.h"

// ---------- initialisation ----------
void Drawer_init(DrawerThatDraws** drawer_ptrAdrr){
    printf("difference avec tSNE software: je faisais les init (sauf SDL_init()) à partir du SDL_thread\n");

    // allocate memory for DrawerThatDraws
    *drawer_ptrAdrr = (DrawerThatDraws*) malloc(sizeof(DrawerThatDraws));
    DrawerThatDraws* drawer = *drawer_ptrAdrr;

    // initialise SDL
    if(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0){
        die("SDL_Init failed!");}

    // initialise SDL_ttf  (ie: fonts)
    if(TTF_Init() != 0){
        die("TTF_Init failed!");}

    // create the window
    drawer->window = SDL_CreateWindow("Spiking neurons toolkit", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WINDOW_W, WINDOW_H, SDL_WINDOW_OPENGL);
    if(drawer->window == NULL){
        die("SDL_CreateWindow failed!");}

    // create the renderer
    if(RENDER_USING_GPU){
        drawer->renderer = SDL_CreateRenderer(drawer->window, -1, SDL_RENDERER_ACCELERATED);
    }   
    else{
        drawer->renderer = SDL_CreateRenderer(drawer->window, -1, SDL_RENDERER_SOFTWARE);
    }
    if(drawer->renderer == NULL){
        die("SDL_CreateRenderer failed!");}

    // load the font
    drawer->font = TTF_OpenFont("../assets/thefont.ttf", 15);
    if(drawer->font == NULL){
        die("TTF_OpenFont failed!");}
    
    // set the initial colour
    set_colour(drawer, AMBER);
}

void Drawer_cleanup(DrawerThatDraws* drawer){
    if (drawer != NULL) {
        if (drawer->font != NULL) {
            TTF_CloseFont(drawer->font);
            drawer->font = NULL;
        }
        // Destroy the renderer if it's created
        if (drawer->renderer != NULL) {
            SDL_DestroyRenderer(drawer->renderer);
            drawer->renderer = NULL;
        }
        // Destroy the window if it's created
        if (drawer->window != NULL) {
            SDL_DestroyWindow(drawer->window);
            drawer->window = NULL;
        }
        // Free the drawer structure itself
        free(drawer);
        drawer = NULL;
    }
    // Quit TTF and SDL subsystems
    TTF_Quit();
    SDL_Quit();
}

// ---------- colours ----------
inline void set_colour(DrawerThatDraws* drawer, uint32_t colour){
    switch(colour){
        case AMBER_0:
            SDL_SetRenderDrawColor(drawer->renderer, AMBER_0_R, AMBER_0_G, AMBER_0_B, 255);
            break;
        case AMBER_1:
            SDL_SetRenderDrawColor(drawer->renderer, AMBER_1_R, AMBER_1_G, AMBER_1_B, 255);
            break;
        case AMBER_2:
            SDL_SetRenderDrawColor(drawer->renderer, AMBER_2_R, AMBER_2_G, AMBER_2_B, 255);
            break;
        case CYAN:
            SDL_SetRenderDrawColor(drawer->renderer, CYAN_R, CYAN_G, CYAN_B, 255);
            break;
        case DANGER:
            SDL_SetRenderDrawColor(drawer->renderer, DANGER_R, DANGER_G, DANGER_B, 255);
            break;
        case BACKGROUND:
            SDL_SetRenderDrawColor(drawer->renderer, 0, 0, 0, 255);
            break;
        default:
            SDL_SetRenderDrawColor(drawer->renderer, 0, 0, 0, 255);
            break;
    }
    drawer->current_colour_SDL    = get_current_colour(drawer);
    drawer->current_colour_uint32 = SDLColorToUint32(drawer->current_colour_SDL);
}

SDL_Color get_current_colour(DrawerThatDraws* drawer){
    SDL_Color colour;
    SDL_GetRenderDrawColor(drawer->renderer, &colour.r, &colour.g, &colour.b, &colour.a);
    return colour;
}

uint32_t SDLColorToUint32(SDL_Color color) {
    return (color.r << 24) | (color.g << 16) | (color.b << 8) | color.a;
}


// ---------- generic drawing functions ----------
void Drawer_clear_screen(DrawerThatDraws* drawer){
    set_colour(drawer, BACKGROUND);
    SDL_RenderClear(drawer->renderer);
}

void Drawer_draw_text(DrawerThatDraws* drawer, uint32_t x, uint32_t y, const char* text){
    SDL_Surface* surfaceMessage = TTF_RenderText_Solid(drawer->font, text, drawer->current_colour_SDL);
    SDL_Texture* message = SDL_CreateTextureFromSurface(drawer->renderer, surfaceMessage);
    SDL_Rect Message_rect;
    Message_rect.x = x;
    Message_rect.y = y;
    Message_rect.w = strlen(text) * 10;
    Message_rect.h = 22;
    SDL_RenderCopy(drawer->renderer, message, NULL, &Message_rect);
    SDL_DestroyTexture(message);
    SDL_FreeSurface(surfaceMessage);
}

void Drawer_draw_rect(DrawerThatDraws* drawer, uint32_t x, uint32_t y, uint32_t w, uint32_t h){
    SDL_Rect rect = {x, y, w, h};
    SDL_RenderDrawRect(drawer->renderer, &rect);
}

void Drawer_draw_circle(DrawerThatDraws* drawer, uint32_t x, uint32_t y, uint32_t r){
    circleColor(drawer->renderer, x, y, r, drawer->current_colour_uint32);
}

void Drawer_draw_triangle(DrawerThatDraws* drawer, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, uint32_t x3, uint32_t y3){
    trigonColor(drawer->renderer, x1, y1, x2, y2, x3, y3, drawer->current_colour_uint32);
}

void Drawer_draw_line(DrawerThatDraws* drawer, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2){
    lineColor(drawer->renderer, x1, y1, x2, y2, drawer->current_colour_uint32);
}

void Drawer_fill_rect(DrawerThatDraws* drawer, uint32_t x, uint32_t y, uint32_t w, uint32_t h){
    SDL_Rect rect = {x, y, w, h};
    SDL_RenderFillRect(drawer->renderer, &rect);
}

void Drawer_fill_circle(DrawerThatDraws* drawer, uint32_t x, uint32_t y, uint32_t r){
    filledCircleColor(drawer->renderer, x, y, r, drawer->current_colour_uint32);
}

void Drawer_fill_triangle(DrawerThatDraws* drawer, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, uint32_t x3, uint32_t y3){
    filledTrigonColor(drawer->renderer, x1, y1, x2, y2, x3, y3, drawer->current_colour_uint32);
}

void Drawer_draw_neon_line(DrawerThatDraws* drawer, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2){
    // Enable OpenGL blending for neon effect
    glEnable(GL_BLEND); // state change in openGL: this is expensive
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // state change in openGL: this is expensive

    // Set the line width and draw the line
    glLineWidth(3.0f); // state change in openGL: this is expensive
    glBegin(GL_LINES);
    glVertex2f(x1, y1);
    glVertex2f(x2, y2);
    glEnd();

    // Disable blending and reset the color
    glDisable(GL_BLEND); // state change in openGL: this is expensive
}


// ---------- drawing the screens & overlay ----------
void Drawer_draw_overlay(DrawerThatDraws* drawer, float render_work_to_sleep_ratio){
    // at the bottom-left corner, display the render_work_to_sleep_ratio. Assume the colour is already set
    char ratio_string[20];
    sprintf(ratio_string, "RWSR: %.2f", render_work_to_sleep_ratio);
    Drawer_draw_text(drawer, 5, WINDOW_H - 30, ratio_string);
}


void Drawer_draw_model_selection_screen(DrawerThatDraws* drawer, float render_work_to_sleep_ratio){
    // clear the screen
    Drawer_clear_screen(drawer);
    // set colour to defaut
    set_colour(drawer, AMBER);
    // draw the screen
    renderModelSelectionScreen(drawer, render_work_to_sleep_ratio);
    // draw the overlay
    Drawer_draw_overlay(drawer, render_work_to_sleep_ratio);
    // now show the screen
    SDL_RenderPresent(drawer->renderer);
}