/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

/*

 The logic here is different here than the C++ version, resulting in
 a 2x speed increase. Nothing magic, the main change is to not
 store the probability, since it is deterministically determined
 based on the true/false counts. Instead of doing the calculation,
 we just lookup the 16-bit value in a lookup table to get the
 corresponding probabiity.

 The only corner case is that in the case of 255 true and 1
 false, the C++ version decides to set the probability to 0 for the next
 true value, which is different than the formula ((a << 8) / ( a + b )).

 To handle this, we use 0 as a special value to indicate this corner case,
 which has a value of 0 in the lookup table. On subsequent calls,
 we make sure that we immediately transition back to (255,1) before
 executing any further logic.

 Always have to be careful with lookup tables, since the cache hit can
 be worse than the calculations it is saving. However, in the case
 of the probability calculation, it is saving a divide or table based
 multiply by the reciprocal, which is also a lookup table of similar size.

 The normalization is a 256 byte lookup table, so it shouldn't have any
 horrible cache implications.
*/
pub struct Branch {
    counts: u16,
}

impl Default for Branch {
    fn default() -> Branch {
        Branch::new()
    }
}

// used to precalculate the probabilities
const fn problookup() -> [u8; 65536] {
    let mut retval = [0; 65536];
    let mut i = 1i32;
    while i < 65536 {
        let a = i >> 8;
        let b = i & 0xff;

        retval[i as usize] = ((a << 8) / (a + b)) as u8;
        i += 1;
    }

    return retval;
}

static PROB_LOOKUP: [u8; 65536] = problookup();

const fn truelookup() -> [u16; 256] {
    let mut retval = [0; 256];
    let mut i = 0i32;
    while i < 256 {
        retval[i as usize] = if i == 0 || i == 1 {
            // special case where we have many trues in a row
            0
        } else {
            ((((i << 8) as u32 + 0x100) >> 1) & 0xff00) as u16 | 0x81
        };
        i += 1;
    }

    retval
}

static NORMALIZE_TRUE: [u16; 256] = truelookup();

const fn falselookup() -> [u16; 256] {
    let mut retval = [0; 256];
    let mut i = 0i32;
    while i < 256 {
        retval[i as usize] = if i == 0 {
            // special case where counts is zero
            0x02ff
        } else if i == 1 {
            // special case where we have many falses in a row
            0xff01
        } else {
            ((1 + (i as u32)) >> 1) as u16 | 0x8100
        };

        i += 1;
    }

    retval
}

static NORMALIZE_FALSE: [u16; 256] = falselookup();

impl Branch {
    pub fn new() -> Self {
        Branch { counts: 0x0101 }
    }

    // used for debugging
    #[allow(dead_code)]
    pub fn get_u64(&self) -> u64 {
        let mut c = self.counts;
        if c == 0 {
            c = 0x01ff;
        }

        return ((PROB_LOOKUP[self.counts as usize] as u64) << 16) + c as u64;
    }

    #[inline(always)]
    pub fn get_probability(&self) -> u8 {
        // 0 is a special corner case which should return probability 0
        // since 0 is impossible to happen since the counts always start at 1
        PROB_LOOKUP[self.counts as usize]
    }

    #[inline(always)]
    pub fn record_and_update_true_obs(&mut self) {
        // do a wrapping subtraction so that we can catch both the case
        // where counts is zero (special case for many trues in a row), or
        // 0x**ff in which case we need to normalize
        // The adjustment to handle zero is handled in the lookup table
        if (self.counts & 0xff).wrapping_sub(1) < 0xfe {
            // non-overflow case is easy
            self.counts = self.counts.wrapping_add(1);
        } else {
            self.counts = NORMALIZE_TRUE[(self.counts >> 8) as usize];
        }
    }

    #[inline(always)]
    pub fn record_and_update_false_obs(&mut self) {
        // do a wrapping subtraction so that we can catch both the case
        // where counts is zero (special case for many trues in a row), or
        // 0xff** in which case we need to normalize.
        // The adjustment to handle zero is handled in the lookup table
        if self.counts.wrapping_sub(1) < 0xff00 {
            // non-overflow case is easy
            self.counts = self.counts.wrapping_add(0x100);
        } else {
            self.counts = NORMALIZE_FALSE[(self.counts & 0xff) as usize];
        }
    }
}

/// This is copied from the C++ implementation to ensure that the behavior is the same
#[cfg(test)]
struct CppBranchImplForTest {
    counts_: [u8; 2],
    probability_: u8,
}

#[cfg(test)]
impl CppBranchImplForTest {
    fn true_count(&self) -> u32 {
        return self.counts_[1] as u32;
    }
    fn false_count(&self) -> u32 {
        return self.counts_[0] as u32;
    }

    fn record_obs_and_update(&mut self, obs: bool) {
        let fcount = self.counts_[0] as u32;
        let tcount = self.counts_[1] as u32;

        let overflow = self.counts_[obs as usize] == 0xff;

        if overflow {
            // check less than 512
            let neverseen = self.counts_[!obs as usize] == 1;
            if neverseen {
                self.counts_[obs as usize] = 0xff;
                self.probability_ = if obs { 0 } else { 255 };
            } else {
                self.counts_[0] = ((1 + fcount) >> 1) as u8;
                self.counts_[1] = ((1 + tcount) >> 1) as u8;
                self.counts_[obs as usize] = 129;
                self.probability_ = self.optimize(self.counts_[0] as u32 + self.counts_[1] as u32);
            }
        } else {
            self.counts_[obs as usize] += 1;
            self.probability_ = self.optimize(fcount + tcount + 1);
        }
    }

    fn optimize(&self, sum: u32) -> u8 {
        let prob = (self.false_count() << 8) / sum;

        prob as u8
    }
}

/// run through all the possible combinations of counts and ensure that the probability is the same
#[test]
fn test_all_probabilities() {
    for i in 0..65536 {
        let mut old_f = CppBranchImplForTest {
            counts_: [(i >> 8) as u8, i as u8],
            probability_: 0,
        };

        if old_f.true_count() == 0 || old_f.false_count() == 0 {
            // ignore the special case where counts is zero which is invalid
            continue;
        }

        let mut new_f = Branch { counts: i as u16 };

        for _k in 0..10 {
            old_f.record_obs_and_update(false);
            new_f.record_and_update_false_obs();
            assert_eq!(old_f.probability_, new_f.get_probability());
        }

        let mut old_t = CppBranchImplForTest {
            counts_: [(i >> 8) as u8, i as u8],
            probability_: 0,
        };
        let mut new_t = Branch { counts: i as u16 };

        for _k in 0..10 {
            old_t.record_obs_and_update(true);
            new_t.record_and_update_true_obs();

            assert_eq!(old_t.probability_, new_t.get_probability());
        }
    }
}
