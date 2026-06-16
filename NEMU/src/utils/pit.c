#include <utils.h>
#include <malloc.h>
#include <pit.h>
#include <profiling/profiling_control.h>

#define PITBUFSZ 512
FILE *pit_fp = NULL;

pitPacket pitbuf[PITBUFSZ];
bool pitbuf_hasaddr[PITBUFSZ];
size_t pitptr = 0;

int pitcount = 0;
int pitmemcount = 0;
int pitctrlcount = 0;

void init_pit(const char *file) {
  if (file == NULL) return;
  pit_fp = fopen(file, "w");
  Assert(pit_fp, "Can not open '%s'", file);
}

void drain_pit() {
  for (int i = 0; i < pitptr; i++) {
    fwrite(&(pitbuf[i].instr), sizeof(uint32_t), 1, pit_fp);
    if (pitbuf_hasaddr[i]) fwrite(&(pitbuf[i].memaddr), sizeof(uint64_t), 1, pit_fp);
  }

  pitptr = 0;
}

void redirect_pit(uint64_t redir_pc) {
  if (pit_fp) {
    if (pitcount <= warmup_interval) return;

    drain_pit();
    uint32_t magic_inst = 0xb;
    fwrite(&magic_inst, sizeof(uint32_t), 1, pit_fp);
    fwrite(&redir_pc, sizeof(uint64_t), 1, pit_fp);
  }
}

void pit(Decode *pkt, uint64_t next_pc, bool is_ctrl) {
  if (pit_fp) {
    if (pitcount++ < warmup_interval) return;
    if (pitcount == warmup_interval + 1)
      fwrite(&(pkt->pc), sizeof(uint64_t), 1, pit_fp);

    pitbuf_hasaddr[pitptr] = pkt->is_mem | is_ctrl;
    pitbuf[pitptr] = (pitPacket) {
      .instr = pkt->isa.instr.val,
      .memaddr = is_ctrl ? next_pc : pkt->maddr
    };

    pitptr++;
    pitmemcount += (int)pkt->is_mem;
    pitctrlcount += (int)is_ctrl;

    if (pitptr == PITBUFSZ) {
      drain_pit();
    }
  }
};

void end_pit() {
  if (pit_fp) {
    drain_pit();
    fclose(pit_fp);
    printf("[PIT] Drained %d instructions: %d memory inst, %d control inst\n",
           pitcount-(int)warmup_interval, pitmemcount, pitctrlcount);
  }
}
