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
        // 0 is a special corner case which should return probability 0
        // since 0 is impossible to happen since the counts always start at 1
        PROB_LOOKUP[self.counts as usize]
    }

    #[inline(always)]
    pub fn record_and_update_true_obs(&mut self) {
        if self.counts == 0 {
            return; // no need to do anything since we are already as baised towards all trues as possible
        }

        if (self.counts & 0xff) != 0xff {
            // non-overflow case is easy
            self.counts += 1;
        } else {
            // special case where it is all trues
            if self.counts == 0x01ff {
                // corner case since the original implementation
                // insists on setting the probabily to zero,
                // although the probability calculation would
                // return 1.
                self.counts = 0;
            } else {
                self.counts = (((self.counts as u32 + 0x100) >> 1) & 0xff00) as u16 | 129;
            }
        }
    }

    #[inline(always)]
    pub fn record_and_update_false_obs(&mut self) {
        if self.counts == 0 {
            // handle corner case where prob was set badly
            self.counts = 0x02ff;
            return;
        }

        if (self.counts & 0xff00) != 0xff00 {
            // non-overflow case is easy
            self.counts += 0x100;
        } else {
            // special case where it is all falses
            if self.counts == 0xff01 {
            } else {
                self.counts = ((1 + (self.counts & 0xff) as u32) >> 1) as u16 | 0x8100;
            }
        }
    }
}
