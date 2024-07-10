#ifndef LIB_CONSTANTS_H
#define LIB_CONSTANTS_H

#include <stdint.h>

extern const float PI;
extern const float FLOAT_EPS;

// ---------- COLOURS ----------
extern const uint8_t AMBER_0_R; // darker colour
extern const uint8_t AMBER_0_G;
extern const uint8_t AMBER_0_B;
extern const uint8_t AMBER_1_R; // base colour
extern const uint8_t AMBER_1_G;
extern const uint8_t AMBER_1_B;
extern const uint8_t AMBER_2_R; // highlight colour
extern const uint8_t AMBER_2_G;
extern const uint8_t AMBER_2_B;
extern const uint8_t CYAN_R;
extern const uint8_t CYAN_G;
extern const uint8_t CYAN_B;
extern const uint8_t DANGER_R;
extern const uint8_t DANGER_G;
extern const uint8_t DANGER_B;

// ---------- TERMINAL COLOURS ----------
// a set of 3 uint8_t values representing base terminal text colour
extern const uint8_t TERMINAL_TEXT_COLOUR_R;
extern const uint8_t TERMINAL_TEXT_COLOUR_G;
extern const uint8_t TERMINAL_TEXT_COLOUR_B;
// a set of 3 uint8_t values representing error terminal text colour
extern const uint8_t TERMINAL_ERROR_COLOUR_R;
extern const uint8_t TERMINAL_ERROR_COLOUR_G;
extern const uint8_t TERMINAL_ERROR_COLOUR_B;
// a set of 3 uint8_t values representing success terminal text colour
extern const uint8_t TERMINAL_SUCCESS_COLOUR_R;
extern const uint8_t TERMINAL_SUCCESS_COLOUR_G ;
extern const uint8_t TERMINAL_SUCCESS_COLOUR_B;

#endif // LIB_CONSTANTS_H