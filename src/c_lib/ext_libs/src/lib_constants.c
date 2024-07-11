#include "lib_constants.h"

const float PI        = 3.14159f;
const float FLOAT_EPS = 1e-10f;

// ---------- COLOURS ----------
const uint8_t AMBER_0_R = 234;
const uint8_t AMBER_0_G = 116;
const uint8_t AMBER_0_B = 28;
const uint8_t AMBER_1_R = 234;
const uint8_t AMBER_1_G = 116;
const uint8_t AMBER_1_B = 28;
const uint8_t AMBER_2_R = 242;
const uint8_t AMBER_2_G = 200;
const uint8_t AMBER_2_B = 148;
const uint8_t CYAN_R    = 149;
const uint8_t CYAN_G    = 221;
const uint8_t CYAN_B    = 217;
const uint8_t DANGER_R  = 254;
const uint8_t DANGER_G  = 34;
const uint8_t DANGER_B  = 95;

// ---------- TERMINAL COLOURS ----------
// a set of 3 uint8_t values representing base terminal text colour
const uint8_t TERMINAL_TEXT_COLOUR_R = AMBER_1_R;
const uint8_t TERMINAL_TEXT_COLOUR_G = AMBER_1_G;
const uint8_t TERMINAL_TEXT_COLOUR_B = AMBER_1_B;
// a set of 3 uint8_t values representing error terminal text colour
const uint8_t TERMINAL_ERROR_COLOUR_R = DANGER_R;
const uint8_t TERMINAL_ERROR_COLOUR_G = DANGER_G;
const uint8_t TERMINAL_ERROR_COLOUR_B = DANGER_B;
// a set of 3 uint8_t values representing success terminal text colour
const uint8_t TERMINAL_SUCCESS_COLOUR_R = AMBER_2_R;
const uint8_t TERMINAL_SUCCESS_COLOUR_G = AMBER_2_G;
const uint8_t TERMINAL_SUCCESS_COLOUR_B = AMBER_2_B;
// a set of 3 uint8_t values representing misc info terminal text colour
const uint8_t TERMINAL_INFO_COLOUR_R = CYAN_R;
const uint8_t TERMINAL_INFO_COLOUR_G = CYAN_G;
const uint8_t TERMINAL_INFO_COLOUR_B = CYAN_B;
