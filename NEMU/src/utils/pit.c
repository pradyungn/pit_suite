#include <utils.h>
#include <malloc.h>
#include <pit.h>

#define PITBUFSZ 512
FILE *pit_fp = NULL;

pitPacket pitbuf[PITBUFSZ];
bool pitbufmem[PITBUFSZ];
size_t pitptr = 0;

int pitcount = 0;
int pitmemcount = 0;

void init_pit(const char *file) {
  if (file == NULL) return;
  pit_fp = fopen(file, "w");
  Assert(pit_fp, "Can not open '%s'", file);
}

void drain_pit() {
  for (int i = 0; i < pitptr; i++) {
    fwrite(&(pitbuf[i].instr), sizeof(uint32_t), 1, pit_fp);
    if (pitbufmem[i]) fwrite(&(pitbuf[i].memaddr), sizeof(uint64_t), 1, pit_fp);
  }

  pitptr = 0;
}

// CHATGPT test function
bool is_mem_access(uint32_t instr)
{
    /* Compressed? (bits [1:0] != 3) */
    if ((instr & 3) != 3) {
        unsigned quadrant = instr & 3;
        unsigned funct3   = (instr >> 13) & 7;

        /* Quadrant 0 (..00) */
        if (quadrant == 0) {
            if (funct3 == 2 ||  /* C.LW   */
                funct3 == 6 ||  /* C.SW   */
                funct3 == 1 ||  /* C.FLD  */
                funct3 == 3 ||  /* C.FLW  */
                funct3 == 7 ||  /* C.FSW  */
                funct3 == 5)    /* C.FSD  */
                return true;
        }

        /* Quadrant 2 (..10) */
        if (quadrant == 2) {
            if (funct3 == 2 ||  /* C.LWSP   */
                funct3 == 6 ||  /* C.SWSP   */
                funct3 == 1 ||  /* C.FLDSP  */
                funct3 == 5 ||   /* C.FSDSP  */
                funct3 == 3 ||   /* C.FLWSP  */
                funct3 == 7)
                return true;
        }

        return false;
    }

    /* Standard 32-bit opcode */
    unsigned opcode = instr & 0x7F;

    if (opcode == 3  ||  /* LOAD */
        opcode == 35 ||  /* STORE */
        opcode == 7  ||  /* LOAD_FP */
        opcode == 39 ||  /* STORE_FP */
        opcode == 47)    /* AMO */
        return true;

    return false;
}

void pit(pitPacket pkt, bool mem) {
  if (pit_fp) {
    pitbufmem[pitptr] = mem;
    pitbuf[pitptr] = pkt;

    pitptr++;
    pitcount++;
    pitmemcount += (int)mem;

    if (pitptr == PITBUFSZ) {
      drain_pit();
    }
  }
};

void end_pit() {
  if (pit_fp) {
    drain_pit();
    fclose(pit_fp);
    printf("[PIT] Drained %d instructions, %d memory instructions\n",
           pitcount, pitmemcount);
  }
}
