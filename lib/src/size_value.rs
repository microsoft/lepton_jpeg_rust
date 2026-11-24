/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

//! Checked conversions and limit checks for sizes read from or written to untrusted data.

use std::num::Saturating;

use crate::lepton_error::{ExitCode, LeptonError, Result};

#[cold]
#[track_caller]
fn size_out_of_range() -> LeptonError {
    let mut e = LeptonError::new(ExitCode::IntegerCastOverflow, "size out of range");
    e.add_context();
    e
}

/// A size or length, either exact or computed with `Saturating` arithmetic.
pub(crate) trait SizeValue: Sized {
    fn to_usize(self) -> Result<usize>;

    #[track_caller]
    fn to_u32(self) -> Result<u32> {
        u32::try_from(self.to_usize()?).map_err(|_| size_out_of_range())
    }
}

// `From` can't be used: there is no `usize -> Saturating<usize>` conversion, and a blanket
// `TryInto` impl would conflict with the `Saturating` impl below.
macro_rules! exact_size_value {
    ($($t:ty),*) => {$(
        impl SizeValue for $t {
            #[track_caller]
            fn to_usize(self) -> Result<usize> {
                usize::try_from(self).map_err(|_| size_out_of_range())
            }
        }
    )*};
}

// Exact integers: errors instead of truncating
exact_size_value!(usize, u32, u64);

// Saturating `+` and `*` stop at exactly usize::MAX, which no real size can reach.
// Don't subtract or divide saturated values, since that can bring them back under a limit.
impl SizeValue for Saturating<usize> {
    #[track_caller]
    fn to_usize(self) -> Result<usize> {
        if self.0 == usize::MAX {
            return Err(size_out_of_range());
        }
        Ok(self.0)
    }
}

/// Returns the size if it is within the limit.
#[track_caller]
pub(crate) fn check_size(size: impl SizeValue, limit: usize, name: &str) -> Result<usize> {
    let size = size.to_usize()?;
    if size > limit {
        let mut e = LeptonError::new(
            ExitCode::BadLeptonFile,
            format!("{name} size {size} exceeds limit {limit}"),
        );
        e.add_context();
        return Err(e);
    }
    Ok(size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saturated_size_is_rejected() {
        let huge = Saturating(usize::MAX / 2) * Saturating(3);
        assert!(check_size(huge, usize::MAX, "test").is_err());
        assert_eq!(
            check_size(Saturating(4usize) * Saturating(5), 20, "test").unwrap(),
            20
        );
        assert!(check_size(21usize, 20, "test").is_err());
    }

    #[test]
    fn to_u32_rejects_truncation() {
        // computed via u64 so the test is portable to 32-bit targets
        let too_big = u64::from(u32::MAX) + 1;
        assert!(too_big.to_u32().is_err());
        assert_eq!(u64::from(u32::MAX).to_u32().unwrap(), u32::MAX);
    }
}
