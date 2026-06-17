#include <common.h>
#include <cpu/decode.h>

void init_pit(const char*, bool);
void pit(Decode*, uint64_t, bool);
void redirect_pit(uint64_t);
