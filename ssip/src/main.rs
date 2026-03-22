use std::env;
use std::fs::File;
use std::str::FromStr;
use riscv_isa::{self, Instruction::*, Target};

use std::io::{BufReader, Read, ErrorKind};

#[derive(Clone, Copy)]
struct PitInst {
    inst: riscv_isa::Instruction,
    addr: Option<u64>
}

#[derive(Default)]
struct TracerState {
    // global
    insts: u64,
    compressed: u64,
    vector: u64,

    // mem_checker
    meminsts: u64,
    loadinsts: u64,
    storeinsts: u64,
    ctrlinsts: u64,

    // dfg_gen

    // fusionCheck
    lastinst: Option<PitInst>,
    fusions: u64,
    adjloads: u64,
    shiftfusions: u64,
    shiftadds: u64,
    farloads: u64,
    alubranch: u64,
    alujalr: u64,
    constbr: u64,

    alubranch_dist: u64,
    alubranch_dist_tot: u64,
}

fn inst_mix(state: &mut TracerState, pkt: Option<&PitInst>, finish: bool) {
    if finish {
        println!("--- Instruction Mix ---");
        println!("Load Accesses: {} ({:.2}%)", state.loadinsts,
                 100.0 * (state.loadinsts as f64) / (state.insts as f64));
        println!("Store Accesses: {} ({:.2}%)", state.storeinsts,
                 100.0 * (state.storeinsts as f64) / (state.insts as f64));
        println!("Mem Insts: {} ({:.2}%)", state.meminsts,
                 100.0 * (state.meminsts as f64) / (state.insts as f64));
        println!("Control Instructions: {} ({:.2}%)", state.ctrlinsts,
                 100.0 * (state.ctrlinsts as f64) / (state.insts as f64));
    } else {
        let p = pkt.unwrap();

        if p.inst.load() {
            state.loadinsts += 1;
        }

        if p.inst.store() {
            state.storeinsts += 1;
        }

        if p.inst.mem() {
            state.meminsts += 1;
        }

        if p.inst.branch() {
            state.ctrlinsts += 1;
        }
    }
}

fn dfg_gen(state: &mut TracerState, pkt: Option<&PitInst>, _: bool) {
}

fn fusion_profiler(state: &mut TracerState, pkt: Option<&PitInst>, finish: bool) {
    if finish {
        println!("--- Fusion Profiler ---");
        println!("Fused Pairs: {} ({:.2}%)", state.fusions,
                 200.0 * (state.fusions as f64) / (state.insts as f64));

        println!("Adjacent Load Fusions: {} ({:.2}%)", state.adjloads,
                 200.0 * (state.adjloads as f64) / (state.loadinsts as f64));
        println!("Far Loads: {} ({:.2}%)", state.farloads,
                 100.0 * (state.farloads as f64) / (state.loadinsts as f64));

        println!("Shift Adds: {}", state.shiftadds);
        println!("Shift Fusion: {}", state.shiftfusions);

        println!("Branch Fusion (Theoretical): {} ({:.2}%)", state.alubranch,
                 100.0 * (state.alubranch as f64) / (state.ctrlinsts as f64));

        println!("Branch Fusion w/ Const Idioms (Theoretical): {} ({:.2}%)", state.constbr,
                 100.0 * (state.constbr as f64) / (state.ctrlinsts as f64));

        println!("JALR Fusion (Theoretical): {} ({:.2}%)", state.alujalr,
                 100.0 * (state.alujalr as f64) / (state.ctrlinsts as f64));

        println!("Average Distance for ALUBR: {:.2}",
                 (state.alubranch_dist_tot as f64) / (state.alubranch as f64));
        return;
    }

    let inst = pkt.unwrap().inst;

    let lastinst = match state.lastinst {
        None => {
            state.lastinst = Some(*pkt.unwrap());
            return
        },
        Some(x) => x
    };

    state.alubranch_dist += 1;

    match (lastinst.inst, inst) {
        (SLLI{ rd, shamt: 32, .. },
         SRLI{ rd: rd2, rs1, shamt: 29..=32 }) if rd == rd2 &&
            rs1 == rd => {
                state.fusions += 1;
                state.shiftfusions += 1;
            },

        (SLLI{ rd, shamt: 48, .. },
         SRLI{ rd: rd2, rs1, shamt: 48 }) if rd == rd2 &&
            rs1 == rd => {
                state.fusions += 1;
                state.shiftfusions += 1;
            },

        (SLLIW{ rd, shamt: 16, .. },
         SRLIW{ rd: rd2, rs1, shamt: 16 } |
         SRAIW{ rd: rd2, rs1, shamt: 16 }) if rd == rd2 &&
            rs1 == rd => {
                state.fusions += 1;
                state.shiftfusions += 1;
            }

        (SLLI{ rd, shamt: 1..=4, .. } |
         SRLI{ rd, shamt: 29..=32, .. },
         ADD{ rd: rd2, rs1, rs2 }) if rd == rd2 &&
            (rs1 == rd || rs2 == rd) => {
                state.fusions += 1;
                state.shiftadds += 1;
            },

        (SRLI{ rd, shamt: 8, .. },
         ANDI{ rd: rd2, rs1, imm: 0xff }) if rd == rd2 &&
            rs1 == rd => {
                state.fusions += 1;
                state.shiftadds += 1;
            }

        // far load
        (ADD{ rd, .. } | ADDI{ rd, .. } | AUIPC{ rd, .. },
         LD{ rd: rd2, rs1, .. } |
         LW{ rd: rd2, rs1, .. } |
         LH{ rd: rd2, rs1, .. } |
         LB{ rd: rd2, rs1, .. }) if rd == rd2 &&
            rs1 == rd => {
                state.fusions += 1;
                state.farloads += 1;
            },

        // short-add
        (ADDW{ rd, .. },
         ANDI{ rd: rd2, rs1, imm: 255 } |
         ANDI{ rd: rd2, rs1, imm: 1 } |
         ZEXT_H{ rd: rd2, rs1 } |
         SEXT_H{ rd: rd2, rs1 }) if rd == rd2 &&
            rs1 == rd => (),

        (LUI{ rd, .. },
         ADDI{ rd: rd2, rs1, .. } |
         ADDIW{ rd: rd2, rs1, .. }) if rd == rd2 &&
            rs1 == rd => (),

        // TODO: Logic fusion (Done by Gemini)
        // oddadd & oddaddw: ANDI (imm == 1) + ADD/ADDW
        (ANDI { rd: rd1, imm: 1, .. },
         ADD { rd: rd2, rs1: rs1_2, rs2: rs2_2, .. } |
         ADDW { rd: rd2, rs1: rs1_2, rs2: rs2_2, .. })
            if rd1 == rd2 && (rd2 == rs1_2 || rd2 == rs2_2) => {
            state.fusions += 1;
        },

        // orh48: ANDI (imm == -256) + OR
        (ANDI { rd: rd1, imm: -256, .. },
         OR { rd: rd2, rs1: rs1_2, rs2: rs2_2, .. })
            if rd1 == rd2 && (rd2 == rs1_2 || rd2 == rs2_2) => {
            state.fusions += 1;
        },

        // mulw7: ANDI (imm == 127) + MULW
        (ANDI { rd: rd1, imm: 127, .. },
         MULW { rd: rd2, rs1: rs1_2, rs2: rs2_2, .. })
            if rd1 == rd2 && (rd2 == rs1_2 || rd2 == rs2_2) => {
            state.fusions += 1;
        },

        // Logic + ANDI (imm == 1)
        (ANDI { rd: rd1, .. } | AND { rd: rd1, .. } | ORI { rd: rd1, .. } |
         OR { rd: rd1, .. }   | XORI { rd: rd1, .. } | XOR { rd: rd1, .. } |
         ORC_B { rd: rd1, .. },
         ANDI { rd: rd2, rs1: rs1_2, imm: 1, .. })
            if rd1 == rd2 && rd2 == rs1_2 => {
            state.fusions += 1;
        },

        // Logic + ZEXT.H
        (ANDI { rd: rd1, .. } | AND { rd: rd1, .. } | ORI { rd: rd1, .. } |
         OR { rd: rd1, .. }   | XORI { rd: rd1, .. } | XOR { rd: rd1, .. } |
         ORC_B { rd: rd1, .. },
         ZEXT_H { rd: rd2, rs1: rs1_2, .. })
            if rd1 == rd2 && rd2 == rs1_2 => {
            state.fusions += 1;1;
        },

        // Load fusion (Pradyun-written code starts here again)
        (
            first @ (LD { .. } | LW { .. } | LH { .. } | LB { .. }),
            LB { rd: rd2, rs1: rs12, offset: off2 } |
            LH { rd: rd2, rs1: rs12, offset: off2 } |
            LW { rd: rd2, rs1: rs12, offset: off2 } |
            LD { rd: rd2, rs1: rs12, offset: off2 }
        ) if {
            // Destructure the shared fields from the first instruction
            let (rd, rs1, offset) = match first {
                LD { rd, rs1, offset } | LW { rd, rs1, offset } |
                LH { rd, rs1, offset } | LB { rd, rs1, offset } => (rd, rs1, offset),
                _ => unreachable!(),
            };

            // Determine the required stride based on the instruction type
            let stride = match first {
                LD { .. } => 8,
                LW { .. } => 4,
                LH { .. } => 2,
                LB { .. } => 1,
                _ => unreachable!(),
            };

            rd != rd2 && rs1 == rs12 && off2 == offset + stride
        } => {
            state.fusions += 1;
            state.adjloads += 1;
        }

        // Theoretical
        (ADDI{ rd, rs1: rs1e, .. } | SLTI{ rd, rs1: rs1e, .. }  | ANDI{ rd, rs1: rs1e, .. } | SRAI{ rd, rs1: rs1e, .. },
         // | ADD{ rd, .. } | SLT{ rd, .. } | AND{ rd, .. } | SUB{ rd, .. },
         BEQ { rs1, rs2, .. }  | BNE  { rs1, rs2, .. } |
         BLT  { rs1, rs2, .. } | BGE  { rs1, rs2, .. } | BLTU { rs1, rs2, .. } |
         BGEU { rs1, rs2, .. }) if rd == rs1 || rd == rs2 => {
            state.fusions += 1;
            state.alubranch += 1;

            state.alubranch_dist_tot += state.alubranch_dist;
            state.alubranch_dist = 0;

            if rs1e == 0 { state.constbr += 1; }
        }

        (AUIPC{ rd, .. } | LUI{ rd, .. } | ADDI{ rd, .. },
         JALR{ rd: rd2, rs1, .. })
            if rd == rs1 && rd == rd2 => {
            state.fusions += 1;
            state.alujalr += 1;
        }

        _ => {
            state.lastinst = Some(*pkt.unwrap());
            return;
        }
    }

    // if fusion successful, don't use lastinst/inst for another fusion
    state.lastinst = None;
}


const HANDLERS: &[fn(&mut TracerState, Option<&PitInst>, bool)] =
    &[
        inst_mix,
        dfg_gen,
        fusion_profiler
    ];

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("[err] no file supplied");
        return;
    }

    let trace = match File::open(&args[1]) {
        Err(_) => {
            eprintln!("[err] invalid tracefile");
            return;
        }
        Ok(f) => f,
    };
    let mut tracereader = BufReader::new(trace);

    // omitted from Xiangshan Spec due to package compat: KHV
    let target = Target::from_str("RV64IMAFDCZicsr_Zifencei_Zba_Zbb_Zbs_Zbkb")
        .unwrap().with_s_mode(true).with_privileged(true);

    let mut state = TracerState::default();

    loop {
        let mut ibuf = [0u8; 4];
        match tracereader.read_exact(&mut ibuf) {
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => break,

            Err(e) => {
                eprintln!("[err] encountered {} on parsing", e.kind());
                return;
            }

            Ok(_) => (),
        }

        let (inst, bytes) = match riscv_isa::decode_le_bytes(&ibuf, &target) {
            None => {
                eprintln!("[err] invalid instruction encountered");
                break;
            }

            Some(x) => x,
        };

        let pit_inst = match inst {
            _ if inst.mem() => {
                let mut maddr = [0u8; 8];

                match tracereader.read_exact(&mut maddr) {
                    Err(e) => {
                        eprintln!("[err] encountered {} on parsing a memaddr", e.kind());
                        return;
                    }

                    Ok(_) => PitInst{
                        inst: inst,
                        addr: Some(u64::from_le_bytes(maddr))
                    },
                }
            }


            UNIMP => {
                let opcode = u32::from_le_bytes(ibuf) & 0x7F;
                if matches!(opcode, 0x7 | 0x27 |0x57) {
                    state.vector += 1;
                } else {
                    println!("[inf] unsupported instr: 0x{:08x} @ order {}",
                             u32::from_le_bytes(ibuf), state.insts);
                }
                PitInst{ inst: inst, addr: None }
            },

            _ => PitInst{ inst: inst, addr: None },
        };

        state.insts += 1;
        if bytes == 2 { state.compressed += 1; }
        else if bytes != 4 {
            eprintln!("[err] invalid instruction encountered");
            return;
        }

        for f in HANDLERS {
            f(&mut state, Some(&pit_inst), false);
        }
    }

    println!("Total Decoded Instructions: {}", state.insts);
    println!("Compressed Instructions: {} ({:.2}%)", state.compressed,
             100.0 * (state.compressed as f64) / (state.insts as f64));
    println!("Vector Instructions: {} ({:.2}%)", state.vector,
             100.0 * (state.vector as f64) / (state.insts as f64));

    for f in HANDLERS {
        f(&mut state, None, true);
    }
}
