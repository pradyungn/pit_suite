use crate::cache::CacheModel;
use std::collections::VecDeque;
use std::fs::File;

#[derive(Clone, Copy)]
pub struct PitInst {
    pub inst: riscv_isa::Instruction,
    pub addr: Option<u64>,
    pub compressed: bool,
}

#[derive(Default)]
pub struct TracerState {
    // global
    pub insts: u64,
    pub compressed: u64,
    pub vector: u64,

    pub pc_annot: bool,
    pub pc: u64,

    // inst mix
    pub meminsts: u64,
    pub loadinsts: u64,
    pub storeinsts: u64,
    pub ctrlinsts: u64,
    pub trapinsts: u64,
    pub floatinsts: u64,
    pub miscinsts: u64,
    pub divinsts: u64,
    pub fences: u64,

    // asm dumper
    pub asm_range: u64,

    // fusionCheck
    pub lastinst: Option<PitInst>,
    pub fusions: u64,
    pub logicfusion: u64,
    pub adjloads: u64,
    pub farloads: u64,
    pub alubranch: u64,
    pub alujalr: u64,
    pub constbr: u64,
    pub alubranch_dist: u64,
    pub alubranch_dists: Vec<u64>,

    // amo profiler
    pub lrct: u64,
    pub scct: u64,
    pub amoct: u64,
    pub aqct: u64,
    pub rlct: u64,
    pub aqrlct: u64,

    // dfg/ilp analyzer
    pub winsize: usize,
    pub prunesize: usize,
    pub verbose: bool,
    pub dfg_window: VecDeque<PitInst>,

    // cache models
    pub dcache: Option<CacheModel>,

    // branch trace dumper
    pub branch_trace_fp: Option<File>,
}

impl TracerState {
    pub fn reset_for_measurement(&mut self, verbose: bool) {
        let winsize = self.winsize;
        let prunesize = self.prunesize;
        let asm_range = self.asm_range;
        let pc_annot = self.pc_annot;
        let pc = self.pc;
        let mut dcache = self.dcache.take();
        let branch_trace_fp = self.branch_trace_fp.take();

        if let Some(dcache) = dcache.as_mut() {
            dcache.reset_stats();
        }

        *self = Self {
            pc_annot,
            pc,
            winsize,
            prunesize,
            verbose,
            asm_range,
            dcache,
            branch_trace_fp,
            ..Default::default()
        };
    }
}
