use clap::Parser;
use riscv_isa::{self, Instruction::*, Target};
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::str::FromStr;

#[derive(Parser)]
#[command(about = "SuperSonic Instruction Profiler")]
struct Args {
    /// PIT trace file to analyze
    tracefile: String,

    /// Enable DFG generation/analysis over a sliding window of WINSIZE instructions
    #[arg(short = 'I', long, value_name = "WINSIZE")]
    ilp_check: Option<u64>,

    /// # of instructions to prune on WINSIZE
    #[arg(short = 'P', long, value_name = "PRUNESIZE", default_value_t = 32)]
    prune_size: u64,

    /// Enable ASM dumping for DUMPLEN instructions
    #[arg(short = 'A', long, value_name = "DUMPLEN")]
    asm: Option<u64>,
}

use std::io::{BufReader, ErrorKind, Read};

#[derive(Clone, Copy)]
struct PitInst {
    inst: riscv_isa::Instruction,
    addr: Option<u64>,
}

#[derive(Default)]
struct TracerState {
    // global
    insts: u64,
    compressed: u64,
    vector: u64,

    // inst mix
    meminsts: u64,
    loadinsts: u64,
    storeinsts: u64,
    ctrlinsts: u64,
    floatinsts: u64,
    miscinsts: u64,
    divinsts: u64,
    fences: u64,

    // asm dumper
    asm_range: u64,

    // fusionCheck
    lastinst: Option<PitInst>,
    fusions: u64,
    logicfusion: u64,
    adjloads: u64,
    farloads: u64,
    alubranch: u64,
    alujalr: u64,
    constbr: u64,
    alubranch_dist: u64,
    alubranch_dist_tot: u64,

    // amo profiler
    lrct: u64,
    scct: u64,
    amoct: u64,
    aqct: u64,
    rlct: u64,
    aqrlct: u64,

    // dfg/ilp analyzer
    winsize: usize,
    prunesize: usize,
    dfg_window: VecDeque<PitInst>,
}

fn inst_mix(state: &mut TracerState, pkt: Option<&PitInst>, finish: bool) {
    if finish {
        println!("--- Instruction Mix ---");
        println!(
            "Load Accesses: {} ({:.2}%)",
            state.loadinsts,
            100.0 * (state.loadinsts as f64) / (state.insts as f64)
        );
        println!(
            "Store Accesses: {} ({:.2}%)",
            state.storeinsts,
            100.0 * (state.storeinsts as f64) / (state.insts as f64)
        );
        println!(
            "Mem Insts: {} ({:.2}%)",
            state.meminsts,
            100.0 * (state.meminsts as f64) / (state.insts as f64)
        );
        println!(
            "Control Instructions: {} ({:.2}%)",
            state.ctrlinsts,
            100.0 * (state.ctrlinsts as f64) / (state.insts as f64)
        );
        println!(
            "Float Instructions: {} ({:.2}%)",
            state.floatinsts,
            100.0 * (state.floatinsts as f64) / (state.insts as f64)
        );
        println!(
            "XS-MISC Instructions: {} ({:.2}%)",
            state.miscinsts,
            100.0 * (state.miscinsts as f64) / (state.insts as f64)
        );
        println!(
            "Div Instructions: {} ({:.2}%)",
            state.divinsts,
            100.0 * (state.divinsts as f64) / (state.insts as f64)
        );
        println!(
            "Fences: {} ({:.2}%)",
            state.fences,
            100.0 * (state.fences as f64) / (state.insts as f64)
        );
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

        if p.inst.float() {
            state.floatinsts += 1;
        }

        if p.inst.misc() {
            state.miscinsts += 1;
        }

        if p.inst.div() {
            state.divinsts += 1;
        }

        if matches!(p.inst, FENCE { .. }) {
            state.fences += 1;
        }
    }
}

fn amoprof(state: &mut TracerState, pkt: Option<&PitInst>, finish: bool) {
    if finish {
        let total = state.lrct + state.scct + state.amoct;
        println!("--- AMO Profiler ---");
        println!(
            "Atomic Instructions: {} ({:.2}%)",
            total,
            100.0 * (total as f64) / (state.insts as f64)
        );
        println!(
            "  LR: {} ({:.2}%)",
            state.lrct,
            100.0 * (state.lrct as f64) / (total as f64)
        );
        println!(
            "  SC: {} ({:.2}%)",
            state.scct,
            100.0 * (state.scct as f64) / (total as f64)
        );
        println!(
            "  AMO: {} ({:.2}%)",
            state.amoct,
            100.0 * (state.amoct as f64) / (total as f64)
        );
        println!(
            "  .aq: {} ({:.2}%)",
            state.aqct,
            100.0 * (state.aqct as f64) / (total as f64)
        );
        println!(
            "  .rl: {} ({:.2}%)",
            state.rlct,
            100.0 * (state.rlct as f64) / (total as f64)
        );
        println!(
            "  .aqrl: {} ({:.2}%)",
            state.aqrlct,
            100.0 * (state.aqrlct as f64) / (total as f64)
        );
        return;
    } else {
        match pkt.unwrap().inst {
            LR_W { aq, rl, .. } | LR_D { aq, rl, .. } => {
                state.lrct += 1;

                state.aqct += aq as u64;
                state.rlct += rl as u64;
                state.aqrlct += (aq * rl) as u64;
            }
            SC_W { aq, rl, .. } | SC_D { aq, rl, .. } => {
                state.scct += 1;

                state.aqct += aq as u64;
                state.rlct += rl as u64;
                state.aqrlct += (aq * rl) as u64;
            }
            AMOSWAP_W { aq, rl, .. }
            | AMOSWAP_D { aq, rl, .. }
            | AMOADD_W { aq, rl, .. }
            | AMOADD_D { aq, rl, .. }
            | AMOXOR_W { aq, rl, .. }
            | AMOXOR_D { aq, rl, .. }
            | AMOAND_W { aq, rl, .. }
            | AMOAND_D { aq, rl, .. }
            | AMOOR_W { aq, rl, .. }
            | AMOOR_D { aq, rl, .. }
            | AMOMIN_W { aq, rl, .. }
            | AMOMIN_D { aq, rl, .. }
            | AMOMAX_W { aq, rl, .. }
            | AMOMAX_D { aq, rl, .. }
            | AMOMINU_W { aq, rl, .. }
            | AMOMINU_D { aq, rl, .. }
            | AMOMAXU_W { aq, rl, .. }
            | AMOMAXU_D { aq, rl, .. } => {
                state.amoct += 1;

                state.aqct += aq as u64;
                state.rlct += rl as u64;
                state.aqrlct += (aq * rl) as u64;
            }
            _ => (),
        };
    }
}

fn dfg_gen(state: &mut TracerState, pkt: Option<&PitInst>, finish: bool) {
    if !finish {
        let isn = pkt.unwrap();

        if state.winsize > 0 && state.dfg_window.len() >= state.winsize {
            // 1: prune head of graph if too many insts
            // Q: what should pruning size be? 1 inst may be too expensive
            state.dfg_window.drain(..state.prunesize);
        }

        // 2: append new inst to graph
        state.dfg_window.push_back(*isn);
    }

    if finish || state.dfg_window.len() == state.winsize {
        // 3. do a DFG traversal here to find graph depth (and efffective ILP)
        // MVP: only use data connectivity, not instruction timings
        // Connection types: Int RF, Float RF, Control (to stores/misc/fence), Fence/AMO

        // store vtick (virtual tick) here
        let mut l_ctrl = 0;
        let mut l_ser = 0;
        let mut l_memser = 0;
        let mut l_mem = 0;
        let mut memticks: HashMap<u64, usize> = HashMap::new();
        let mut intregdep = [0usize; 32];
        let mut flregdep = [0usize; 32];

        let mut maxtick = 0;

        for pkt in &state.dfg_window {
            let inst = &pkt.inst;

            let mut vtick = 0;

            // find dependencies to determine earliest possible tick
            if let Some(rs1) = inst.get_rs1() {
                vtick = vtick.max(intregdep[rs1 as usize]);
            }

            if let Some(rs2) = inst.get_rs2() {
                vtick = vtick.max(intregdep[rs2 as usize]);
            }

            if let Some(frs1) = inst.get_frs1() {
                vtick = vtick.max(flregdep[frs1 as usize]);
            }

            if let Some(frs2) = inst.get_frs2() {
                vtick = vtick.max(flregdep[frs2 as usize]);
            }

            if let Some(frs3) = inst.get_frs3() {
                vtick = vtick.max(flregdep[frs3 as usize]);
            }

            // non-reg dependencies
            if inst.mem() {
                let vaddr = pkt.addr.unwrap();
                vtick = vtick.max(*memticks.get(&vaddr).unwrap_or(&0));

                if inst.store() {
                    vtick = vtick.max(l_ctrl); // stores drain after branches
                }

                vtick = vtick.max(l_memser);
            }

            if inst.misc() {
                // be VERY conservative, just do it after everything else.
                vtick = maxtick;
            }

            if matches!(inst, FENCE { .. }) {
                vtick = vtick.max(l_mem);
            }

            vtick = vtick.max(l_ser); // all instructions wait on serialization, for now

            vtick += 1; // earliest possible time of completion

            // record this tick if it is the "youngest" completed inst
            maxtick = maxtick.max(vtick);

            // record this tick into dependency sources
            if let Some(rd) = inst.get_rd() {
                intregdep[rd as usize] = vtick;
            }

            if let Some(frd) = inst.get_frd() {
                flregdep[frd as usize] = vtick;
            }

            if inst.branch() {
                l_ctrl = l_ctrl.max(vtick);
            }

            if inst.mem() {
                l_mem = l_mem.max(vtick);

                let vaddr = pkt.addr.unwrap();
                memticks.insert(vaddr, vtick);
            }

            if inst.misc() {
                l_ser = vtick;
            }

            if matches!(inst, FENCE { .. }) {
                l_memser = vtick;
            }
        }

        let wlen = state.dfg_window.len();
        println!(
            "Window @ {}: {wlen} insts, {maxtick} ticks, ILP {:.2}",
            state.insts,
            wlen as f64 / maxtick as f64
        );
    }
}

fn asm_dump(state: &mut TracerState, pkt: Option<&PitInst>, finish: bool) {
    if !finish && state.insts <= state.asm_range {
        print!("{}", pkt.unwrap().inst);
        match pkt.unwrap().addr {
            Some(x) => println!(" [0x{:08x}]", x),
            None => println!(""),
        };
    }
}

fn fusion_profiler(state: &mut TracerState, pkt: Option<&PitInst>, finish: bool) {
    if finish {
        println!("--- Fusion Profiler ---");
        println!(
            "Fused Pairs: {} ({:.2}%)",
            state.fusions,
            200.0 * (state.fusions as f64) / (state.insts as f64)
        );

        println!(
            "Adjacent Load Fusions: {} ({:.2}%)",
            state.adjloads,
            200.0 * (state.adjloads as f64) / (state.loadinsts as f64)
        );
        println!(
            "Far Loads: {} ({:.2}%)",
            state.farloads,
            100.0 * (state.farloads as f64) / (state.loadinsts as f64)
        );

        println!("Logic Fusions: {}", state.logicfusion);

        println!(
            "Branch Fusion (Theoretical): {} ({:.2}%)",
            state.alubranch,
            100.0 * (state.alubranch as f64) / (state.ctrlinsts as f64)
        );

        println!(
            "Branch Fusion w/ Const Idioms (Theoretical): {} ({:.2}%)",
            state.constbr,
            100.0 * (state.constbr as f64) / (state.ctrlinsts as f64)
        );

        println!(
            "JALR Fusion (Theoretical): {} ({:.2}%)",
            state.alujalr,
            100.0 * (state.alujalr as f64) / (state.ctrlinsts as f64)
        );

        println!(
            "Average Distance for ALUBR: {:.2}",
            (state.alubranch_dist_tot as f64) / (state.alubranch as f64)
        );
        return;
    }

    let inst = pkt.unwrap().inst;

    let lastinst = match state.lastinst {
        None => {
            state.lastinst = Some(*pkt.unwrap());
            return;
        }
        Some(x) => x,
    };

    state.alubranch_dist += 1;

    match (lastinst.inst, inst) {
        (
            SLLI { rd, shamt: 32, .. },
            SRLI {
                rd: rd2,
                rs1,
                shamt: 29..=32,
            },
        ) if rd == rd2 && rs1 == rd => {
            state.fusions += 1;
            state.logicfusion += 1;
        }

        (
            SLLI { rd, shamt: 48, .. },
            SRLI {
                rd: rd2,
                rs1,
                shamt: 48,
            },
        ) if rd == rd2 && rs1 == rd => {
            state.fusions += 1;
            state.logicfusion += 1;
        }

        (
            SLLIW { rd, shamt: 16, .. },
            SRLIW {
                rd: rd2,
                rs1,
                shamt: 16,
            }
            | SRAIW {
                rd: rd2,
                rs1,
                shamt: 16,
            },
        ) if rd == rd2 && rs1 == rd => {
            state.fusions += 1;
            state.logicfusion += 1;
        }

        (
            SLLI {
                rd, shamt: 1..=4, ..
            }
            | SRLI {
                rd, shamt: 29..=32, ..
            },
            ADD { rd: rd2, rs1, rs2 },
        ) if rd == rd2 && (rs1 == rd || rs2 == rd) => {
            state.fusions += 1;
            state.logicfusion += 1;
        }

        (
            SRLI { rd, shamt: 8, .. },
            ANDI {
                rd: rd2,
                rs1,
                imm: 0xff,
            },
        ) if rd == rd2 && rs1 == rd => {
            state.fusions += 1;
            state.logicfusion += 1;
        }

        // far load
        (
            ADD { rd, .. } | ADDI { rd, .. } | AUIPC { rd, .. },
            LD { rd: rd2, rs1, .. }
            | LW { rd: rd2, rs1, .. }
            | LH { rd: rd2, rs1, .. }
            | LB { rd: rd2, rs1, .. },
        ) if rd == rd2 && rs1 == rd => {
            state.fusions += 1;
            state.farloads += 1;
        }

        // short-add
        (
            ADDW { rd, .. },
            ANDI {
                rd: rd2,
                rs1,
                imm: 255,
            }
            | ANDI {
                rd: rd2,
                rs1,
                imm: 1,
            }
            | ZEXT_H { rd: rd2, rs1 }
            | SEXT_H { rd: rd2, rs1 },
        ) if rd == rd2 && rs1 == rd => {
            state.fusions += 1;
            state.logicfusion += 1;
        }

        (LUI { rd, .. }, ADDI { rd: rd2, rs1, .. } | ADDIW { rd: rd2, rs1, .. })
            if rd == rd2 && rs1 == rd =>
        {
            state.fusions += 1;
            state.logicfusion += 1;
        }

        // TODO: Logic fusion (Done by Gemini)
        // oddadd & oddaddw: ANDI (imm == 1) + ADD/ADDW
        (
            ANDI {
                rd: rd1, imm: 1, ..
            },
            ADD {
                rd: rd2,
                rs1: rs1_2,
                rs2: rs2_2,
                ..
            }
            | ADDW {
                rd: rd2,
                rs1: rs1_2,
                rs2: rs2_2,
                ..
            },
        ) if rd1 == rd2 && (rd2 == rs1_2 || rd2 == rs2_2) => {
            state.fusions += 1;
        }

        // orh48: ANDI (imm == -256) + OR
        (
            ANDI {
                rd: rd1, imm: -256, ..
            },
            OR {
                rd: rd2,
                rs1: rs1_2,
                rs2: rs2_2,
                ..
            },
        ) if rd1 == rd2 && (rd2 == rs1_2 || rd2 == rs2_2) => {
            state.fusions += 1;
            state.logicfusion += 1;
        }

        // mulw7: ANDI (imm == 127) + MULW
        (
            ANDI {
                rd: rd1, imm: 127, ..
            },
            MULW {
                rd: rd2,
                rs1: rs1_2,
                rs2: rs2_2,
                ..
            },
        ) if rd1 == rd2 && (rd2 == rs1_2 || rd2 == rs2_2) => {
            state.fusions += 1;
            state.logicfusion += 1;
        }

        // Logic + ANDI (imm == 1)
        (
            ANDI { rd: rd1, .. }
            | AND { rd: rd1, .. }
            | ORI { rd: rd1, .. }
            | OR { rd: rd1, .. }
            | XORI { rd: rd1, .. }
            | XOR { rd: rd1, .. }
            | ORC_B { rd: rd1, .. },
            ANDI {
                rd: rd2,
                rs1: rs1_2,
                imm: 1,
                ..
            },
        ) if rd1 == rd2 && rd2 == rs1_2 => {
            state.fusions += 1;
            state.logicfusion += 1;
        }

        // Logic + ZEXT.H
        (
            ANDI { rd: rd1, .. }
            | AND { rd: rd1, .. }
            | ORI { rd: rd1, .. }
            | OR { rd: rd1, .. }
            | XORI { rd: rd1, .. }
            | XOR { rd: rd1, .. }
            | ORC_B { rd: rd1, .. },
            ZEXT_H {
                rd: rd2,
                rs1: rs1_2,
                ..
            },
        ) if rd1 == rd2 && rd2 == rs1_2 => {
            state.fusions += 1;
            state.logicfusion += 1;
        }

        // Load fusion (Pradyun-written code starts here again)
        (
            first @ (LD { .. } | LW { .. } | LH { .. } | LB { .. }),
            LB {
                rd: rd2,
                rs1: rs12,
                offset: off2,
            }
            | LH {
                rd: rd2,
                rs1: rs12,
                offset: off2,
            }
            | LW {
                rd: rd2,
                rs1: rs12,
                offset: off2,
            }
            | LD {
                rd: rd2,
                rs1: rs12,
                offset: off2,
            },
        ) if {
            // Destructure the shared fields from the first instruction
            let (rd, rs1, offset) = match first {
                LD { rd, rs1, offset }
                | LW { rd, rs1, offset }
                | LH { rd, rs1, offset }
                | LB { rd, rs1, offset } => (rd, rs1, offset),
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
        } =>
        {
            state.fusions += 1;
            state.adjloads += 1;
        }

        // Theoretical
        (
            ADDI { rd, rs1: rs1e, .. }
            | SLTI { rd, rs1: rs1e, .. }
            | ANDI { rd, rs1: rs1e, .. }
            | SRAI { rd, rs1: rs1e, .. },
            // | ADD{ rd, .. } | SLT{ rd, .. } | AND{ rd, .. } | SUB{ rd, .. },
            BEQ { rs1, rs2, .. }
            | BNE { rs1, rs2, .. }
            | BLT { rs1, rs2, .. }
            | BGE { rs1, rs2, .. }
            | BLTU { rs1, rs2, .. }
            | BGEU { rs1, rs2, .. },
        ) if rd == rs1 || rd == rs2 => {
            state.fusions += 1;
            state.alubranch += 1;

            state.alubranch_dist_tot += state.alubranch_dist;
            state.alubranch_dist = 0;

            if rs1e == 0 {
                state.constbr += 1;
            }
        }

        (AUIPC { rd, .. } | LUI { rd, .. } | ADDI { rd, .. }, JALR { rd: rd2, rs1, .. })
            if rd == rs1 && rd == rd2 =>
        {
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

fn main() {
    let mut handlers: Vec<fn(&mut TracerState, Option<&PitInst>, bool)> =
        vec![amoprof, inst_mix, fusion_profiler];
    let args = Args::parse();

    let trace = match File::open(&args.tracefile) {
        Err(_) => {
            eprintln!("[err] invalid tracefile");
            return;
        }
        Ok(f) => f,
    };
    let mut tracereader = BufReader::new(trace);

    // omitted from Xiangshan Spec due to package compat: KHV
    let target = Target::from_str("RV64IMAFDCZicsr_Zifencei_Zba_Zbb_Zbs_Zbkb")
        .unwrap()
        .with_s_mode(true)
        .with_privileged(true);

    let asm_range = match args.asm {
        None => 0,
        Some(x) => {
            handlers.push(asm_dump);
            x
        }
    };

    let ilp_winsize = match args.ilp_check {
        None => 0,
        Some(x) => {
            handlers.push(dfg_gen);
            x
        }
    };

    let mut state = TracerState {
        winsize: ilp_winsize as usize,
        prunesize: args.prune_size as usize,
        asm_range: asm_range,
        ..Default::default()
    };

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

                    Ok(_) => PitInst {
                        inst: inst,
                        addr: Some(u64::from_le_bytes(maddr)),
                    },
                }
            }

            UNIMP => {
                let opcode = u32::from_le_bytes(ibuf) & 0x7F;
                if matches!(opcode, 0x7 | 0x27 | 0x57) {
                    state.vector += 1;
                } else {
                    println!(
                        "[inf] unsupported instr: 0x{:08x} @ order {}",
                        u32::from_le_bytes(ibuf),
                        state.insts
                    );
                }
                PitInst {
                    inst: inst,
                    addr: None,
                }
            }

            _ => PitInst {
                inst: inst,
                addr: None,
            },
        };

        state.insts += 1;
        if bytes == 2 {
            state.compressed += 1;
        } else if bytes != 4 {
            eprintln!("[err] invalid instruction encountered");
            return;
        }

        for f in &handlers {
            f(&mut state, Some(&pit_inst), false);
        }
    }

    println!("Total Decoded Instructions: {}", state.insts);
    println!(
        "Compressed Instructions: {} ({:.2}%)",
        state.compressed,
        100.0 * (state.compressed as f64) / (state.insts as f64)
    );
    println!(
        "Vector Instructions: {} ({:.2}%)",
        state.vector,
        100.0 * (state.vector as f64) / (state.insts as f64)
    );

    for f in &handlers {
        f(&mut state, None, true);
    }
}
