#ifndef GUI_RENDER_H
#define GUI_RENDER_H

#include <SDL.h>
#include <SDL2/SDL_ttf.h>
#include <GL/gl.h>

#include "SDL2/SDL2_gfxPrimitives.h"
#include "lib_constants.h"
#include "common_includes.h"
#include "const_gui.h"

#define RENDER_USING_GPU false

#define AMBER_0      0u
#define AMBER_1      1u
#define AMBER_2      2u
#define AMBER        AMBER_1
#define CYAN         3u
#define DANGER       4u
#define BACKGROUND   5u
#define WINDOW_W     2200u
#define WINDOW_H     1300u

typedef struct {
    // things proper to SDL
    SDL_Thread*    sdl_thread;
    SDL_Renderer*  renderer;
    SDL_Window*    window;
    TTF_Font*      font;

    // contains the current rendering colour
    SDL_Color current_colour_SDL;
    uint32_t  current_colour_uint32;

    // contains N_rand_colours colours (in a matrix)
    uint8_t     N_rand_colours;
    uint8_t**   rand_colours;
} GenericDrawer;

// ---------- togglable button ----------
typedef struct {
    uint32_t    x;
    uint32_t    y;
    uint32_t    w;
    uint32_t    h;
    uint32_t    string_length;
    const char* text;
    bool        active;
    bool        hovered;
    bool        clicking;
} TogglableButton;
// sets the button's properties (assumes that memory has been allocated for the button)
void TogglableButton_init(TogglableButton* button, uint32_t x, uint32_t y, uint32_t w, uint32_t h, const char* text);
bool TogglableButton_is_hovered(TogglableButton* button, uint32_t mouse_x, uint32_t mouse_y);
void TogglableButton_draw(TogglableButton* button, GenericDrawer* drawer);

// ---------- initialisation & cleanup ----------
void GenericDrawer_init(GenericDrawer** drawer_ptrAdrr); 
void GenericDrawer_cleanup(GenericDrawer* drawer); 

// ---------- colours ----------
void set_colour(GenericDrawer* drawer, uint32_t colour); // set the colour of the SDL renderer
void set_colour_rgb(GenericDrawer* drawer, uint8_t r, uint8_t g, uint8_t b); // set the colour of the SDL renderer
SDL_Color get_current_colour(GenericDrawer* drawer); // get the current colour of the SDL renderer
uint32_t SDLColorToUint32(SDL_Color color); // convert an SDL_Color to a uint32_t
// ---------- generic drawing functions ----------
void GenericDrawer_clear_screen(GenericDrawer* drawer); // clear the screen
void GenericDrawer_draw_text(GenericDrawer* drawer, uint32_t x, uint32_t y, const char* text); // draw text on the screen
void GenericDrawer_draw_line(GenericDrawer* drawer, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2); // draw a line on the screen
void GenericDrawer_draw_rect(GenericDrawer* drawer, uint32_t x, uint32_t y, uint32_t w, uint32_t h); // draw a rectangle on the screen
void GenericDrawer_draw_circle(GenericDrawer* drawer, uint32_t x, uint32_t y, uint32_t r); // draw a circle on the screen
void GenericDrawer_draw_triangle(GenericDrawer* drawer, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, uint32_t x3, uint32_t y3); // draw a triangle on the screen
void GenericDrawer_fill_rect(GenericDrawer* drawer, uint32_t x, uint32_t y, uint32_t w, uint32_t h); // fill a rectangle on the screen
void GenericDrawer_fill_circle(GenericDrawer* drawer, uint32_t x, uint32_t y, uint32_t r); // fill a circle on the screen
void GenericDrawer_fill_triangle(GenericDrawer* drawer, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, uint32_t x3, uint32_t y3); // fill a triangle on the screen

// ---------- drawing the screens & overlay ----------
// reset the colour, clear the screen
void GenericDrawer_get_ready_to_draw(GenericDrawer* drawer);
void GenericDrawer_draw_overlay(GenericDrawer* drawer, float render_work_to_sleep_ratio);
// draw overlay then send the rendered image to the screen
void GenericDrawer_finalise_drawing(GenericDrawer* drawer, float render_work_to_sleep_ratio);

#endif // GUI_RENDER_H