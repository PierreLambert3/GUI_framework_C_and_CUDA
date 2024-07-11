#include "gui_render.h"

// ---------- togglable button ----------
void TogglableButton_init(TogglableButton* button, uint32_t x, uint32_t y, uint32_t w, uint32_t h, const char* text){
    button->x        = x;
    button->y        = y;
    button->w        = w;
    button->h        = h;
    button->string_length = strlen(text);
    button->text     = text;
    button->active   = false;
    button->hovered  = false;
    button->clicking = false;
}

bool TogglableButton_is_hovered(TogglableButton* button, uint32_t mouse_x, uint32_t mouse_y){
    return (mouse_x >= button->x && mouse_x <= button->x + button->w && mouse_y >= button->y && mouse_y <= button->y + button->h);
}

void TogglableButton_draw(TogglableButton* button, GenericDrawer* drawer){
    if(button->active){
        set_colour(drawer, AMBER_2);
        GenericDrawer_fill_rect(drawer, button->x, button->y, button->w, button->h);
        set_colour(drawer, BACKGROUND);
    }
    else{
        set_colour(drawer, AMBER);
    }

    if(button->clicking){
        set_colour(drawer, AMBER_1);
        GenericDrawer_fill_rect(drawer, button->x, button->y, button->w, button->h);
        set_colour(drawer, BACKGROUND);
    }

    GenericDrawer_draw_text(drawer, button->x + button->w/2 - button->string_length*5, button->y + button->h/2 - 10, button->text);

    if(button->hovered){
        set_colour(drawer, AMBER_1);
        GenericDrawer_draw_rect(drawer, button->x-1, button->y-1, button->w+2, button->h+2);
        GenericDrawer_draw_rect(drawer, button->x, button->y, button->w, button->h);
        GenericDrawer_draw_rect(drawer, button->x+1, button->y+1, button->w-2, button->h-2);
    }
    else{
        set_colour(drawer, AMBER);
        GenericDrawer_draw_rect(drawer, button->x, button->y, button->w, button->h);
    }


}


// ---------- initialisation ----------
void GenericDrawer_init(GenericDrawer** drawer_ptrAdrr){
    printf("difference avec tSNE software: je faisais les init (sauf SDL_init()) à partir du SDL_thread\n");

    // allocate memory for GenericDrawer
    *drawer_ptrAdrr = (GenericDrawer*) malloc(sizeof(GenericDrawer));
    GenericDrawer* drawer = *drawer_ptrAdrr;

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
    drawer->font = TTF_OpenFont("../assets/thefont.ttf", 72);
    if(drawer->font == NULL){
        die("TTF_OpenFont failed!");}
	TTF_SetFontSize(drawer->font, 16);
    
    // set the initial colour
    set_colour(drawer, AMBER);
}

void GenericDrawer_cleanup(GenericDrawer* drawer){
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
extern inline void set_colour(GenericDrawer* drawer, uint32_t colour){
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

extern inline void set_colour_rgb(GenericDrawer* drawer, uint8_t r, uint8_t g, uint8_t b){
    SDL_SetRenderDrawColor(drawer->renderer, r, g, b, 255);
    drawer->current_colour_SDL    = get_current_colour(drawer);
    drawer->current_colour_uint32 = SDLColorToUint32(drawer->current_colour_SDL);
}

extern inline SDL_Color get_current_colour(GenericDrawer* drawer){
    SDL_Color colour;
    SDL_GetRenderDrawColor(drawer->renderer, &colour.r, &colour.g, &colour.b, &colour.a);
    return colour;
}

extern inline uint32_t SDLColorToUint32(SDL_Color color) {
    return (color.r << 24) | (color.g << 16) | (color.b << 8) | color.a;
}


// ---------- generic drawing functions ----------
extern inline void GenericDrawer_clear_screen(GenericDrawer* drawer){
    set_colour(drawer, BACKGROUND);
    SDL_RenderClear(drawer->renderer);
}

extern inline void GenericDrawer_draw_text(GenericDrawer* drawer, uint32_t x, uint32_t y, const char* text){
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

extern inline void GenericDrawer_draw_rect(GenericDrawer* drawer, uint32_t x, uint32_t y, uint32_t w, uint32_t h){
    SDL_Rect rect = {x, y, w, h};
    SDL_RenderDrawRect(drawer->renderer, &rect);
}

extern inline void GenericDrawer_draw_circle(GenericDrawer* drawer, uint32_t x, uint32_t y, uint32_t r){
    circleColor(drawer->renderer, x, y, r, drawer->current_colour_uint32);
}

extern inline void GenericDrawer_draw_triangle(GenericDrawer* drawer, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, uint32_t x3, uint32_t y3){
    trigonColor(drawer->renderer, x1, y1, x2, y2, x3, y3, drawer->current_colour_uint32);
}

extern inline void GenericDrawer_draw_line(GenericDrawer* drawer, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2){
    lineColor(drawer->renderer, x1, y1, x2, y2, drawer->current_colour_uint32);
}

extern inline void GenericDrawer_fill_rect(GenericDrawer* drawer, uint32_t x, uint32_t y, uint32_t w, uint32_t h){
    SDL_Rect rect = {x, y, w, h};
    SDL_RenderFillRect(drawer->renderer, &rect);
}

extern inline void GenericDrawer_fill_circle(GenericDrawer* drawer, uint32_t x, uint32_t y, uint32_t r){
    filledCircleColor(drawer->renderer, x, y, r, drawer->current_colour_uint32);
}

extern inline void GenericDrawer_fill_triangle(GenericDrawer* drawer, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, uint32_t x3, uint32_t y3){
    filledTrigonColor(drawer->renderer, x1, y1, x2, y2, x3, y3, drawer->current_colour_uint32);
}

// ---------- drawing the screens & overlay ----------
// reset the colour, clear the screen
void GenericDrawer_get_ready_to_draw(GenericDrawer* drawer){
    // clear the screen
    GenericDrawer_clear_screen(drawer);
    // reset colour to defaut
    set_colour(drawer, AMBER);
}

void GenericDrawer_draw_overlay(GenericDrawer* drawer, float render_work_to_sleep_ratio){
    // a small gauge indicates how much the rendering work is compared to the sleeping time, low is good
    float gauge_value = render_work_to_sleep_ratio > 1.0 ? 1.0 : render_work_to_sleep_ratio;
    uint8_t red   = (uint8_t) (DANGER_R * gauge_value + AMBER_2_R * (1 - gauge_value));
    uint8_t green = (uint8_t) (DANGER_G * gauge_value + AMBER_2_G * (1 - gauge_value));
    uint8_t blue  = (uint8_t) (DANGER_B * gauge_value + AMBER_2_B * (1 - gauge_value));
    set_colour_rgb(drawer, red, green, blue);
    GenericDrawer_fill_rect(drawer, 0, WINDOW_H - 31*gauge_value, 6, 30*gauge_value); 
    GenericDrawer_draw_rect(drawer, 0, WINDOW_H - 31, 6, 30);
}

// send the rendered image to the screen
void GenericDrawer_finalise_drawing(GenericDrawer* drawer, float render_work_to_sleep_ratio){
     // draw the overlay
    GenericDrawer_draw_overlay(drawer, render_work_to_sleep_ratio);
    // now show the screen
    SDL_RenderPresent(drawer->renderer);
}
