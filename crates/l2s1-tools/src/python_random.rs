//! CPython `random.Random` integer-seed selection for existing frozen datasets.
//! Only the `getrandbits`, `_randbelow`, `sample`, and `shuffle` paths are used.

const N: usize = 624;
const M: usize = 397;

pub struct PythonRandom {
    state: [u32; N],
    index: usize,
}

impl PythonRandom {
    pub fn new(seed: u64) -> Self {
        let mut value = Self {
            state: [0; N],
            index: N,
        };
        value.init_genrand(19_650_218);
        let mut key = vec![seed as u32];
        if seed >> 32 != 0 {
            key.push((seed >> 32) as u32);
        }
        let mut i = 1;
        let mut j = 0;
        for _ in 0..N.max(key.len()) {
            let previous = value.state[i - 1];
            value.state[i] = (value.state[i]
                ^ (previous ^ (previous >> 30)).wrapping_mul(1_664_525))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                value.state[0] = value.state[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..N - 1 {
            let previous = value.state[i - 1];
            value.state[i] = (value.state[i]
                ^ (previous ^ (previous >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                value.state[0] = value.state[N - 1];
                i = 1;
            }
        }
        value.state[0] = 0x8000_0000;
        value
    }

    fn init_genrand(&mut self, seed: u32) {
        self.state[0] = seed;
        for i in 1..N {
            self.state[i] = 1_812_433_253_u32
                .wrapping_mul(self.state[i - 1] ^ (self.state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            for i in 0..N {
                let next = (i + 1) % N;
                let y = (self.state[i] & 0x8000_0000) | (self.state[next] & 0x7fff_ffff);
                self.state[i] =
                    self.state[(i + M) % N] ^ (y >> 1) ^ if y & 1 != 0 { 0x9908_b0df } else { 0 };
            }
            self.index = 0;
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^= value >> 18;
        value
    }

    pub fn randbelow(&mut self, upper: usize) -> usize {
        assert!(upper > 0);
        let bits = usize::BITS - upper.leading_zeros();
        loop {
            let value = if bits <= 32 {
                (self.next_u32() >> (32 - bits)) as usize
            } else {
                let low = self.next_u32() as u64;
                let high_bits = bits - 32;
                let high = (self.next_u32() >> (32 - high_bits)) as u64;
                (low | (high << 32)) as usize
            };
            if value < upper {
                return value;
            }
        }
    }

    pub fn sample_indices(&mut self, population: usize, count: usize) -> Vec<usize> {
        assert!(count <= population);
        let mut setsize = 21;
        if count > 5 {
            let mut power = 1;
            while power < 3 * count {
                power *= 4;
            }
            setsize += power;
        }
        let mut output = Vec::with_capacity(count);
        if population <= setsize {
            let mut pool = (0..population).collect::<Vec<_>>();
            for i in 0..count {
                let pick = self.randbelow(population - i);
                output.push(pool[pick]);
                pool[pick] = pool[population - i - 1];
            }
        } else {
            let mut seen = std::collections::HashSet::new();
            for _ in 0..count {
                let mut pick = self.randbelow(population);
                while !seen.insert(pick) {
                    pick = self.randbelow(population);
                }
                output.push(pick);
            }
        }
        output
    }

    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            items.swap(i, self.randbelow(i + 1));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PythonRandom;

    #[test]
    fn matches_cpython_integer_seed_sampling() {
        let mut rng = PythonRandom::new(20260921);
        assert_eq!(
            rng.sample_indices(1900, 10),
            [1783, 495, 75, 390, 159, 1694, 93, 1777, 1469, 322]
        );
        let mut values = (0..10).collect::<Vec<_>>();
        rng.shuffle(&mut values);
        assert_eq!(values, [7, 4, 1, 0, 9, 2, 8, 3, 5, 6]);
    }
}
