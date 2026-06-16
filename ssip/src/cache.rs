#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ReplacementPolicy {
    Lru,
    TreePlru,
}

impl ReplacementPolicy {
    fn parse(value: &str) -> Result<Self, String> {
        let trimmed = value.trim();
        match trimmed.to_ascii_lowercase().as_str() {
            "lru" => Ok(Self::Lru),
            "tree-plru" => Ok(Self::TreePlru),
            _ => Err(format!(
                "invalid replacement policy `{trimmed}`; expected `lru` or `tree-plru`"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Lru => "lru",
            Self::TreePlru => "tree-plru",
        }
    }
}

#[derive(Clone, Copy)]
struct CacheConfig {
    size: usize,
    line_size: usize,
    ways: usize,
    sets: usize,
    replacement: ReplacementPolicy,
}

#[derive(Clone, Copy, Default)]
struct CacheLine {
    valid: bool,
    tag: u64,
    last_used: u64,
}

#[derive(Default)]
struct CacheStats {
    accesses: u64,
    hits: u64,
    misses: u64,
}

struct CacheSet {
    lines: Vec<CacheLine>,
    replacement: ReplacementState,
}

enum ReplacementState {
    Lru,
    TreePlru { bits: Vec<bool> },
}

impl CacheSet {
    fn new(ways: usize, replacement: ReplacementPolicy) -> Self {
        let replacement = match replacement {
            ReplacementPolicy::Lru => ReplacementState::Lru,
            ReplacementPolicy::TreePlru => ReplacementState::TreePlru {
                bits: vec![false; ways - 1],
            },
        };

        Self {
            lines: vec![CacheLine::default(); ways],
            replacement,
        }
    }

    fn touch(&mut self, way: usize, timestamp: u64) {
        self.lines[way].last_used = timestamp;

        match &mut self.replacement {
            ReplacementState::Lru => (),
            ReplacementState::TreePlru { bits } => tree_plru_touch(bits, self.lines.len(), way),
        }
    }

    fn victim(&self) -> usize {
        if let Some(invalid) = self.lines.iter().position(|line| !line.valid) {
            return invalid;
        }

        match &self.replacement {
            ReplacementState::Lru => {
                self.lines
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, line)| line.last_used)
                    .unwrap()
                    .0
            }
            ReplacementState::TreePlru { bits } => tree_plru_victim(bits, self.lines.len()),
        }
    }
}

pub struct CacheModel {
    cfg: CacheConfig,
    sets: Vec<CacheSet>,
    stats: CacheStats,
    timestamp: u64,
}

impl CacheModel {
    pub fn new(
        size: usize,
        line_size: usize,
        ways: usize,
        replacement: ReplacementPolicy,
    ) -> Result<Self, String> {
        if size == 0 {
            return Err("size must be non-zero".to_string());
        }
        if line_size == 0 {
            return Err("line size must be non-zero".to_string());
        }
        if ways == 0 {
            return Err("associativity must be non-zero".to_string());
        }
        if replacement == ReplacementPolicy::TreePlru && !ways.is_power_of_two() {
            return Err("tree-plru associativity must be a power of two".to_string());
        }
        if size % line_size != 0 {
            return Err("size must be a multiple of line size".to_string());
        }

        let lines = size / line_size;
        if lines == 0 {
            return Err("cache must contain at least one line".to_string());
        }
        if lines % ways != 0 {
            return Err("number of lines must be a multiple of associativity".to_string());
        }

        let sets = lines / ways;
        let cfg = CacheConfig {
            size,
            line_size,
            ways,
            sets,
            replacement,
        };

        Ok(Self {
            cfg,
            sets: (0..sets)
                .map(|_| CacheSet::new(ways, replacement))
                .collect(),
            stats: CacheStats::default(),
            timestamp: 0,
        })
    }

    pub fn from_spec(spec: &str, default_replacement: ReplacementPolicy) -> Result<Self, String> {
        let parts: Vec<&str> = spec.split(',').map(str::trim).collect();
        if !(3..=4).contains(&parts.len()) {
            return Err("expected SIZE,LINE_SIZE,WAYS[,POLICY]".to_string());
        }

        let size = parse_cache_usize(parts[0], "size")?;
        let line_size = parse_cache_usize(parts[1], "line size")?;
        let ways = parse_cache_usize(parts[2], "associativity")?;
        let replacement = if parts.len() == 4 {
            ReplacementPolicy::parse(parts[3])?
        } else {
            default_replacement
        };

        Self::new(size, line_size, ways, replacement)
    }

    pub fn reset_stats(&mut self) {
        self.stats = CacheStats::default();
    }

    pub fn access(&mut self, addr: u64) -> bool {
        self.stats.accesses += 1;
        self.timestamp += 1;

        let line_addr = addr / (self.cfg.line_size as u64);
        let set_idx = (line_addr % (self.cfg.sets as u64)) as usize;
        let tag = line_addr / (self.cfg.sets as u64);
        let set = &mut self.sets[set_idx];

        if let Some(hit_idx) = set
            .lines
            .iter()
            .position(|line| line.valid && line.tag == tag)
        {
            self.stats.hits += 1;
            set.touch(hit_idx, self.timestamp);
            return true;
        }

        self.stats.misses += 1;
        let replace_idx = set.victim();
        set.lines[replace_idx] = CacheLine {
            valid: true,
            tag,
            last_used: 0,
        };
        set.touch(replace_idx, self.timestamp);

        false
    }

    pub fn print_stats(&self, name: &str) {
        println!("--- {name} ---");
        println!(
            "Config: {} B, {} B lines, {}-way, {} sets, {} replacement",
            self.cfg.size,
            self.cfg.line_size,
            self.cfg.ways,
            self.cfg.sets,
            self.cfg.replacement.name()
        );
        println!("Accesses: {}", self.stats.accesses);

        let hit_rate = pct(self.stats.hits, self.stats.accesses);
        let miss_rate = pct(self.stats.misses, self.stats.accesses);
        println!("Hits: {} ({hit_rate:.2}%)", self.stats.hits);
        println!("Misses: {} ({miss_rate:.2}%)", self.stats.misses);
    }
}

fn tree_plru_touch(bits: &mut [bool], ways: usize, way: usize) {
    let mut node = 0;
    let mut base_way = 0;
    let mut span = ways;

    while node < bits.len() {
        let half = span / 2;
        let split = base_way + half;

        if way < split {
            // Bit values point toward the victim subtree. Touching the left
            // subtree makes the right subtree the next PLRU candidate.
            bits[node] = true;
            node = 2 * node + 1;
        } else {
            // Touching the right subtree makes the left subtree the next PLRU
            // candidate.
            bits[node] = false;
            node = 2 * node + 2;
            base_way = split;
        }

        span = half;
    }
}

fn tree_plru_victim(bits: &[bool], ways: usize) -> usize {
    let mut node = 0;
    let mut victim = 0;
    let mut span = ways;

    while node < bits.len() {
        let half = span / 2;

        if bits[node] {
            victim += half;
            node = 2 * node + 2;
        } else {
            node = 2 * node + 1;
        }

        span = half;
    }

    victim
}

fn parse_cache_usize(value: &str, name: &str) -> Result<usize, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(format!("{name} is empty"));
    }

    trimmed
        .parse::<usize>()
        .map_err(|_| format!("invalid {name}: `{trimmed}`"))
}

fn pct(part: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        100.0 * (part as f64) / (total as f64)
    }
}
