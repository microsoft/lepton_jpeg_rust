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
        // 0x00ff is a special corner case which should return probability 0
        // since 0x00ff is impossible to happen since the counts always start at 1
        PROB_LOOKUP[self.counts as usize]
    }

    #[inline(always)]
    pub fn record_and_update_bit(&mut self, bit: bool) {
        // rotation is use to update either the true or false counter
        // this allows the same code to be used without branching, which
        // makes the CPU happy (about 20% happier)
        let orig = self.counts.rotate_left(bit as u32 * 8);
        let (mut sum, o) = orig.overflowing_add(0x100);
        if o {
            // normalize, except is special case where we have 0xff same bits in a row
            // in which case we want to bias the probability to get better compression
            let mask = if orig == 0xff01 { 0xff00 } else { 0x8100 };
            sum = ((1 + (sum & 0xff)) >> 1) | mask;
        }

        self.counts = sum.rotate_left(bit as u32 * 8);
    }
}

#[test]
fn test_branch_update_false() {
    let mut b = Branch { counts: 0x0101 };
    b.record_and_update_bit(false);
    assert_eq!(b.counts, 0x0201);

    b.counts = 0x80ff;
    b.record_and_update_bit(false);
    assert_eq!(b.counts, 0x81ff);

    b.counts = 0xff01;
    b.record_and_update_bit(false);
    assert_eq!(b.counts, 0xff01);

    b.counts = 0xff02;
    b.record_and_update_bit(false);
    assert_eq!(b.counts, 0x8101);

    b.counts = 0xffff;
    b.record_and_update_bit(false);
    assert_eq!(b.counts, 0x8180);
}

#[test]
fn test_branch_update_true() {
    let mut b = Branch { counts: 0x0101 };
    b.record_and_update_bit(true);
    assert_eq!(b.counts, 0x0102);

    b.counts = 0xff80;
    b.record_and_update_bit(true);
    assert_eq!(b.counts, 0xff81);

    b.counts = 0x01ff;
    b.record_and_update_bit(true);
    assert_eq!(b.counts, 0x01ff);

    b.counts = 0x02ff;
    b.record_and_update_bit(true);
    assert_eq!(b.counts, 0x0181);

    b.counts = 0xffff;
    b.record_and_update_bit(true);
    assert_eq!(b.counts, 0x8081);
}

/// run through all the possible combinations of counts and ensure that the probability is the same
#[test]
fn test_all_probabilities() {
    /// This is copied from the C++ implementation to ensure that the behavior is the same
    struct OriginalImplForTest {
        counts: [u8; 2],
        probability: u8,
    }

    impl OriginalImplForTest {
        fn true_count(&self) -> u32 {
            return self.counts[1] as u32;
        }
        fn false_count(&self) -> u32 {
            return self.counts[0] as u32;
        }

        fn record_obs_and_update(&mut self, obs: bool) {
            let fcount = self.counts[0] as u32;
            let tcount = self.counts[1] as u32;

            let overflow = self.counts[obs as usize] == 0xff;

            if overflow {
                // check less than 512
                let neverseen = self.counts[!obs as usize] == 1;
                if neverseen {
                    self.counts[obs as usize] = 0xff;
                    self.probability = if obs { 0 } else { 255 };
                } else {
                    self.counts[0] = ((1 + fcount) >> 1) as u8;
                    self.counts[1] = ((1 + tcount) >> 1) as u8;
                    self.counts[obs as usize] = 129;
                    self.probability = self.optimize(self.counts[0] as u32 + self.counts[1] as u32);
                }
            } else {
                self.counts[obs as usize] += 1;
                self.probability = self.optimize(fcount + tcount + 1);
            }
        }

        fn optimize(&self, sum: u32) -> u8 {
            let prob = (self.false_count() << 8) / sum;

            prob as u8
        }
    }

    for i in 0u16..=65535 {
        let mut old_f = OriginalImplForTest {
            counts: [(i >> 8) as u8, i as u8],
            probability: 0,
        };

        if old_f.true_count() == 0 || old_f.false_count() == 0 {
            // starting counts can't be zero (we use 0 as an internal special value for the new implementation for the edge case of many trues in a row)
            continue;
        }

        let mut new_f = Branch { counts: i as u16 };

        for _k in 0..10 {
            old_f.record_obs_and_update(false);
            new_f.record_and_update_bit(false);
            assert_eq!(old_f.probability, new_f.get_probability());
        }

        let mut old_t = OriginalImplForTest {
            counts: [(i >> 8) as u8, i as u8],
            probability: 0,
        };
        let mut new_t = Branch { counts: i as u16 };

        for _k in 0..10 {
            old_t.record_obs_and_update(true);
            new_t.record_and_update_bit(true);

            if old_t.probability == 0 {
                // there is a change of behavior here compared to the C++ version,
                // but because of the way split is calculated it doesn't result in an
                // overall change in the way that encoding is done, but it does simplify
                // one of the corner cases.
                assert_eq!(new_t.get_probability(), 1);
            } else {
                assert_eq!(old_t.probability, new_t.get_probability());
            }
        }
    }
}
