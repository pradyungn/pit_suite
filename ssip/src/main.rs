mod cache;
mod state;

use cache::{CacheModel, ReplacementPolicy};
use clap::Parser;
use riscv_isa::{self, Instruction::*, Target};
use state::{PitInst, TracerState};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, ErrorKind, Read, Write};
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
    #[arg(short = 'X', long, value_name = "PRUNESIZE", default_value_t = 32)]
    prune_size: u64,

    /// Enable ASM dumping for DUMPLEN instructions
    #[arg(short = 'A', long, value_name = "DUMPLEN")]
    asm: Option<u64>,

    /// Enable D-cache simulation; optional value is SIZE,LINE_SIZE,WAYS[,POLICY]
    #[arg(
        long,
        value_name = "SIZE,LINE_SIZE,WAYS[,POLICY]",
        num_args = 0..=1,
        default_missing_value = "32768,64,8,tree-plru",
        require_equals = true
    )]
    dcache: Option<String>,

    /// Number of decoded instructions to process as warmup before resetting all profiler stats
    #[arg(long, value_name = "INSTS", default_value_t = 0)]
    warmup: u64,

    /// Verbose DFG output (prints per-instruction dependencies for the first window)
    #[arg(short = 'V', long)]
    verbose: bool,

    /// If trace has PC annotations, collect that too
    #[arg(short = 'P', long)]
    pc_annot: bool,

    /// Dump branch trace to path path
    #[arg(short = 'B', long)]
    branch_trace: Option<String>,
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
            "Trap Instructions: {} ({:.2}%)",
            state.trapinsts,
            100.0 * (state.trapinsts as f64) / (state.insts as f64)
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

        if p.inst.trap() {
            state.trapinsts += 1;
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
        if state.verbose {
            dfg_traverse_verbose(state);
            state.verbose = false;
        } else {
            dfg_traverse(state);
        }
    }
}

fn dfg_traverse(state: &TracerState) {
    let mut l_ctrl = 0usize;
    let mut l_ser = 0usize;
    let mut l_memser = 0usize;
    let mut l_mem = 0usize;
    let mut memticks: HashMap<u64, usize> = HashMap::new();
    let mut intregdep = [0usize; 32];
    let mut flregdep = [0usize; 32];
    let mut maxtick = 0;

    for pkt in &state.dfg_window {
        let inst = &pkt.inst;
        let mut vtick = 0;

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

        if inst.mem() {
            let vaddr = pkt.addr.unwrap();
            vtick = vtick.max(*memticks.get(&vaddr).unwrap_or(&0));
            if inst.store() {
                vtick = vtick.max(l_ctrl);
            }
            vtick = vtick.max(l_memser);
        }
        if inst.misc() {
            vtick = maxtick;
        }
        if matches!(inst, FENCE { .. }) {
            vtick = vtick.max(l_mem);
        }
        vtick = vtick.max(l_ser);
        vtick += 1;
        maxtick = maxtick.max(vtick);

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

fn dfg_traverse_verbose(state: &TracerState) {
    let none: usize = usize::MAX;
    let mut l_ctrl = (0usize, none);
    let mut l_ser = (0usize, none);
    let mut l_memser = (0usize, none);
    let mut l_mem = (0usize, none);
    let mut memticks: HashMap<u64, (usize, usize)> = HashMap::new();
    let mut intregdep = [(0usize, none); 32];
    let mut flregdep = [(0usize, none); 32];
    let mut maxtick = 0;

    for (i, pkt) in state.dfg_window.iter().enumerate() {
        let inst = &pkt.inst;
        let mut vtick = 0;
        let mut deps: Vec<(usize, &str)> = Vec::new();

        if let Some(rs1) = inst.get_rs1() {
            let (tick, prod) = intregdep[rs1 as usize];
            if tick > vtick {
                vtick = tick;
            }
            if prod != none {
                deps.push((prod, "rs1"));
            }
        }
        if let Some(rs2) = inst.get_rs2() {
            let (tick, prod) = intregdep[rs2 as usize];
            if tick > vtick {
                vtick = tick;
            }
            if prod != none {
                deps.push((prod, "rs2"));
            }
        }
        if let Some(frs1) = inst.get_frs1() {
            let (tick, prod) = flregdep[frs1 as usize];
            if tick > vtick {
                vtick = tick;
            }
            if prod != none {
                deps.push((prod, "frs1"));
            }
        }
        if let Some(frs2) = inst.get_frs2() {
            let (tick, prod) = flregdep[frs2 as usize];
            if tick > vtick {
                vtick = tick;
            }
            if prod != none {
                deps.push((prod, "frs2"));
            }
        }
        if let Some(frs3) = inst.get_frs3() {
            let (tick, prod) = flregdep[frs3 as usize];
            if tick > vtick {
                vtick = tick;
            }
            if prod != none {
                deps.push((prod, "frs3"));
            }
        }

        if inst.mem() {
            let vaddr = pkt.addr.unwrap();
            if let Some(&(tick, prod)) = memticks.get(&vaddr) {
                if tick > vtick {
                    vtick = tick;
                }
                if prod != none {
                    deps.push((prod, "mem"));
                }
            }
            if inst.store() {
                if l_ctrl.0 > vtick {
                    vtick = l_ctrl.0;
                }
                if l_ctrl.1 != none {
                    deps.push((l_ctrl.1, "ctrl"));
                }
            }
            if l_memser.0 > vtick {
                vtick = l_memser.0;
            }
            if l_memser.1 != none {
                deps.push((l_memser.1, "memser"));
            }
        }
        if inst.misc() {
            vtick = maxtick;
            deps.push((i.wrapping_sub(1), "serialize"));
        }
        if matches!(inst, FENCE { .. }) {
            if l_mem.0 > vtick {
                vtick = l_mem.0;
            }
            if l_mem.1 != none {
                deps.push((l_mem.1, "fence"));
            }
        }
        if l_ser.0 > vtick {
            vtick = l_ser.0;
        }
        if l_ser.1 != none {
            deps.push((l_ser.1, "ser"));
        }

        vtick += 1;
        maxtick = maxtick.max(vtick);

        let dep_str: Vec<String> = deps
            .iter()
            .map(|(prod, kind)| format!("#{prod}({kind})"))
            .collect();
        println!("  [{i}] {inst} @ tick {vtick} <- [{}]", dep_str.join(", "));

        if let Some(rd) = inst.get_rd() {
            intregdep[rd as usize] = (vtick, i);
        }
        if let Some(frd) = inst.get_frd() {
            flregdep[frd as usize] = (vtick, i);
        }
        if inst.branch() {
            if vtick > l_ctrl.0 {
                l_ctrl = (vtick, i);
            }
        }
        if inst.mem() {
            if vtick > l_mem.0 {
                l_mem = (vtick, i);
            }
            let vaddr = pkt.addr.unwrap();
            memticks.insert(vaddr, (vtick, i));
        }
        if inst.misc() {
            l_ser = (vtick, i);
        }
        if matches!(inst, FENCE { .. }) {
            l_memser = (vtick, i);
        }
    }

    let wlen = state.dfg_window.len();
    println!(
        "Window @ {}: {wlen} insts, {maxtick} ticks, ILP {:.2}",
        state.insts,
        wlen as f64 / maxtick as f64
    );
}

fn asm_dump(state: &mut TracerState, pkt: Option<&PitInst>, finish: bool) {
    if !finish && state.insts <= state.asm_range {
        if state.pc_annot && state.insts == 1 {
            println!("start @ 0x{:08x}", state.pc);
        }

        print!("{}", pkt.unwrap().inst);

        if pkt.unwrap().compressed {
            print!(" [c]");
        }

        match pkt.unwrap().addr {
            Some(x) => {
                if pkt.unwrap().inst.mem() {
                    println!(" [0x{:08x}]", x)
                } else {
                    println!(" [0x{:08x} -> 0x{:08x}]", state.pc, x)
                }
            }
            None => println!(""),
        };
    }
}

fn dcache_profiler(state: &mut TracerState, pkt: Option<&PitInst>, finish: bool) {
    if finish {
        if let Some(dcache) = &state.dcache {
            dcache.print_stats("DCache");
        }
        return;
    }

    let pkt = pkt.unwrap();
    if !(pkt.inst.load() || pkt.inst.store()) {
        return;
    }

    if let (Some(dcache), Some(addr)) = (state.dcache.as_mut(), pkt.addr) {
        dcache.access(addr);
    }
}

fn branch_trace_dump(state: &mut TracerState, pkt: Option<&PitInst>, finish: bool) {
    if finish {
        println!("[inf] branch trace dumped");
    } else {
        // trace fmt: pc, jump addr, cond, indir, call, ret
        let inst = pkt.unwrap().inst;
        if !inst.branch() {
            return;
        };

        let jmp_pc = pkt.unwrap().addr.unwrap();
        let branch_trace_fp = state.branch_trace_fp.as_mut().unwrap();

        match inst {
            JALR { rd, rs1, .. } => {
                writeln!(
                    branch_trace_fp,
                    "{:08x},{:08x},{},{},{},{}",
                    state.pc,
                    jmp_pc,
                    0,
                    1,
                    (rd == 1 || rd == 5) as u8,
                    ((rs1 == 1 || rs1 == 5) && rd != rs1) as u8
                )
                .expect("[err] failed to write to branch trace");
            }

            JAL { rd, .. } => {
                writeln!(
                    branch_trace_fp,
                    "{:08x},{:08x},{},{},{},{}",
                    state.pc,
                    jmp_pc,
                    0,
                    0,
                    (rd == 1 || rd == 5) as u8,
                    0
                )
                .expect("[err] failed to write to branch trace");
            }

            // check for taken branches
            _ if (state.pc + (if pkt.unwrap().compressed { 2 } else { 4 })) != jmp_pc => {
                writeln!(
                    branch_trace_fp,
                    "{:08x},{:08x},{},{},{},{}",
                    state.pc, jmp_pc, 1, 0, 0, 0
                )
                .expect("[err] failed to write to branch trace");
            }

            _ => (),
        }
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

        if state.alubranch_dists.is_empty() {
            println!("ALUBR Distance: no samples");
        } else {
            let mut sorted = state.alubranch_dists.clone();
            sorted.sort_unstable();
            let pct = |p: f64| -> f64 {
                let n = sorted.len();
                let pos = p * (n - 1) as f64;
                let lo = pos.floor() as usize;
                let hi = pos.ceil() as usize;
                let frac = pos - lo as f64;
                sorted[lo] as f64 + frac * (sorted[hi] as f64 - sorted[lo] as f64)
            };
            println!(
                "ALUBR Distance (n={}): Q1={:.2}, median={:.2}, Q3={:.2}",
                sorted.len(),
                pct(0.25),
                pct(0.50),
                pct(0.75)
            );
            println!(
                "ALUBR Distance: mean={:.2}, 5-pct={:.2}",
                (sorted.iter().sum::<u64>() as f64) / (sorted.len() as f64),
                pct(0.05)
            );
        }
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

            if state.alubranch > 1 {
                state.alubranch_dists.push(state.alubranch_dist);
            }

            state.alubranch_dist = 0;

            if rs1e == 0 {
                state.constbr += 1;
            }

            // let cur = pkt.unwrap();
            // let comp_tag = match (lastinst.compressed, cur.compressed) {
            //     (true, true) => "both compressed",
            //     (true, false) => "first compressed",
            //     (false, true) => "second compressed",
            //     (false, false) => "neither compressed",
            // };
            // println!("[fusion] {} | {} ({})", lastinst.inst, inst, comp_tag);
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

fn dump_stats(
    state: &mut TracerState,
    handlers: &[fn(&mut TracerState, Option<&PitInst>, bool)],
    label: Option<&str>,
) {
    if let Some(label) = label {
        println!("=== {label} ===");
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

    for f in handlers {
        f(state, None, true);
    }
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

    let dcache = match args.dcache.as_deref() {
        None => None,
        Some(spec) => match CacheModel::from_spec(spec, ReplacementPolicy::TreePlru) {
            Ok(cache) => {
                handlers.push(dcache_profiler);
                Some(cache)
            }
            Err(e) => {
                eprintln!("[err] invalid dcache config: {e}");
                return;
            }
        },
    };

    if args.branch_trace.is_some() && !args.pc_annot {
        eprintln!("[err] cannot dump branch traces without a PC-annotated PIT dump");
        return;
    }

    let branch_trace_fp = match args.branch_trace.as_deref() {
        None => None,
        Some(path) => match File::create(path) {
            Ok(fp) => {
                handlers.push(branch_trace_dump);
                Some(fp)
            }
            Err(e) => {
                eprintln!("[err] invalid branch trace path: {e}");
                return;
            }
        },
    };

    // read out the starting PC on annotated traces
    let start_pc = if args.pc_annot {
        let mut start_pcbuf = [0u8; 8];

        match tracereader.read_exact(&mut start_pcbuf) {
            Err(e) => {
                eprintln!("[err] encountered {} on parsing start pc", e.kind());
                return;
            }

            Ok(_) => u64::from_le_bytes(start_pcbuf),
        }
    } else {
        0u64
    };

    let mut state = TracerState {
        pc_annot: args.pc_annot,
        pc: start_pc,

        winsize: ilp_winsize as usize,
        prunesize: args.prune_size as usize,
        verbose: args.verbose,
        asm_range: asm_range,
        dcache,
        branch_trace_fp,
        ..Default::default()
    };
    let mut trace_insts = 0u64;
    let mut warmup_complete = args.warmup == 0;

    loop {
        let mut ibuf = [0u8; 4];
        match tracereader.read_exact(&mut ibuf) {
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => break,

            Err(e) => {
                eprintln!("[err] encountered {} on parsing", e.kind());
                break;
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
                        compressed: bytes == 2,
                    },
                }
            }

            _ if args.pc_annot && (inst.branch() || inst.trap()) => {
                let mut npc = [0u8; 8];

                match tracereader.read_exact(&mut npc) {
                    Err(e) => {
                        eprintln!("[err] encountered {} on parsing a pc annotation", e.kind());
                        return;
                    }

                    Ok(_) => PitInst {
                        inst: inst,
                        addr: Some(u64::from_le_bytes(npc)),
                        compressed: bytes == 2,
                    },
                }
            }

            UNIMP => {
                let opcode = u32::from_le_bytes(ibuf) & 0x7F;
                if matches!(opcode, 0x7 | 0x27 | 0x57) {
                    state.vector += 1;
                } else if opcode == 0xb {
                    let mut npcbuf = [0u8; 8];

                    tracereader.read_exact(&mut npcbuf)
                        .expect("[err] faulted while parsing a trap pc");

                    let npc = u64::from_le_bytes(npcbuf);

                    if state.insts < state.asm_range {
                        println!(
                            "redirect: 0x{:08x} -> 0x{:08x}",
                            state.pc, npc
                        );
                    }

                    state.pc = npc;
                    continue;
                } {
                    println!(
                        "[inf] unsupported instr: 0x{:08x} @ order {}",
                        u32::from_le_bytes(ibuf),
                        trace_insts
                    );
                }
                PitInst {
                    inst: inst,
                    addr: None,
                    compressed: bytes == 2,
                }
            }

            _ => PitInst {
                inst: inst,
                addr: None,
                compressed: bytes == 2,
            },
        };

        trace_insts += 1;
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

        if args.pc_annot {
            if inst.branch() || inst.trap() {
                state.pc = pit_inst.addr.unwrap();
            } else {
                state.pc += bytes as u64;
            }
        }

        if args.warmup > 0 && trace_insts == args.warmup {
            dump_stats(&mut state, &handlers, Some("Warmup Statistics"));
            state.reset_for_measurement(args.verbose);
            warmup_complete = true;
        }
    }

    if args.warmup > 0 && !warmup_complete {
        dump_stats(&mut state, &handlers, Some("Warmup Statistics"));
        eprintln!("[inf] trace ended before warmup completed; no measurement statistics emitted");
        return;
    }

    if args.warmup > 0 && state.insts == 0 {
        println!("=== Measurement Statistics ===");
        println!("[inf] no instructions after warmup");
        return;
    }

    let final_label = if args.warmup > 0 {
        Some("Measurement Statistics")
    } else {
        None
    };
    dump_stats(&mut state, &handlers, final_label);
}
