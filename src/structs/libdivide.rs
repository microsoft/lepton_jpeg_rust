pub struct BranchFreeDividerU32 {
    magic: u32,
    more: u8,
}

impl BranchFreeDividerU32 {

    #[inline]
    fn mullhi(x: u32, y: u32) -> u32 {
        (u64::from(x) * u64::from(y)) >> 32
    }

    fn unsigned_branchfree_div_by(numer: u32, denom: BranchFreeDividerU32) -> u32
    {
        let numer = self;
        let q = Self::mullhi(denom.magic, numer);
        let t = ((numer - q) >> 1) + q;
        t.wrapping_shr(denom.more as u32)
    }
}

impl DividerInt for u32 {
    const SHIFT_MASK: u8 = SHIFT_MASK_32;
    const _BITS: u32 = 32;
    const SIGNED: bool = false;
    type Double = u64;
    type Unsigned = Self;
    type UnsignedDouble = Self::Double;

    fn internal_gen(self, branchfree: bool) -> Result<(Self, u8), DividerError> {
        let d = self;
        if d == 0 {
            return Err(DividerError::Zero);
        }

        let floor_log_2_d = (Self::_BITS as u32 - 1) - d.leading_zeros();

        // Power of 2
        Ok(if (d & (d - 1)) == 0 {
            // We need to subtract 1 from the shift value in case of an unsigned
            // branchfree divider because there is a hardcoded right shift by 1
            // in its division algorithm. Because of this we also need to add back
            // 1 in its recovery algorithm.
            (0, (floor_log_2_d - u32::from(branchfree)) as u8)
        } else {
            let (proposed_m, rem) = (1u64 << (floor_log_2_d + 32)).div_rem(&(d as u64));
            let mut proposed_m = proposed_m as u32;
            let rem = rem as u32;
            assert!(rem > 0 && rem < d);

            let e = d - rem;

            // This power works if e < 2**floor_log_2_d.
            let more = if !branchfree && (e < (1 << floor_log_2_d)) {
                // This power works
                floor_log_2_d as u8
            } else {
                // We have to use the general 33-bit algorithm.  We need to compute
                // (2**power) / d. However, we already have (2**(power-1))/d and
                // its remainder.  By doubling both, and then correcting the
                // remainder, we can compute the larger division.
                // don't care about overflow here - in fact, we expect it
                proposed_m = proposed_m.wrapping_add(proposed_m);
                let twice_rem = rem.wrapping_add(rem);
                if twice_rem >= d || twice_rem < rem {
                    proposed_m += 1;
                }
                (floor_log_2_d as u8) | ADD_MARKER
            };
            (1 + proposed_m, more)
            // result.more's shift should in general be ceil_log_2_d. But if we
            // used the smaller power, we subtract one from the shift because we're
            // using the smaller power. If we're using the larger power, we
            // subtract one from the shift because it's taken care of by the add
            // indicator. So floor_log_2_d happens to be correct in both cases.
        })
    }
}
