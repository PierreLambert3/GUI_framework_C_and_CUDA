#ifndef GUI_RENDER_H
#define GUI_RENDER_H

#include <SDL.h>
#include <SDL2/SDL_ttf.h>
#include "SDL2/SDL2_gfxPrimitives.h"
#include "lib_constants.h"
#include "common_includes.h"

#include <GL/gl.h>

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

} DrawerThatDraws;

// ---------- initialisation & cleanup ----------
void Drawer_init(DrawerThatDraws** drawer_ptrAdrr); 
void Drawer_cleanup(DrawerThatDraws* drawer); 

// ---------- colours ----------
void set_colour(DrawerThatDraws* drawer, uint32_t colour); // set the colour of the SDL renderer
SDL_Color get_current_colour(DrawerThatDraws* drawer); // get the current colour of the SDL renderer
uint32_t SDLColorToUint32(SDL_Color color); // convert an SDL_Color to a uint32_t
// ---------- drawing ----------
void Drawer_clear_screen(DrawerThatDraws* drawer); // clear the screen

void Drawer_draw_text(DrawerThatDraws* drawer, uint32_t x, uint32_t y, const char* text); // draw text on the screen

void Drawer_draw_line(DrawerThatDraws* drawer, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2); // draw a line on the screen
void Drawer_draw_rect(DrawerThatDraws* drawer, uint32_t x, uint32_t y, uint32_t w, uint32_t h); // draw a rectangle on the screen
void Drawer_draw_circle(DrawerThatDraws* drawer, uint32_t x, uint32_t y, uint32_t r); // draw a circle on the screen
void Drawer_draw_triangle(DrawerThatDraws* drawer, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, uint32_t x3, uint32_t y3); // draw a triangle on the screen

void Drawer_fill_rect(DrawerThatDraws* drawer, uint32_t x, uint32_t y, uint32_t w, uint32_t h); // fill a rectangle on the screen
void Drawer_fill_circle(DrawerThatDraws* drawer, uint32_t x, uint32_t y, uint32_t r); // fill a circle on the screen
void Drawer_fill_triangle(DrawerThatDraws* drawer, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, uint32_t x3, uint32_t y3); // fill a triangle on the screen

void Drawer_draw_neon_line(DrawerThatDraws* drawer, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2); // neon light effect

#endif // GUI_RENDER_H