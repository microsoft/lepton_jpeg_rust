/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See License.txt in the project root for license information.
 *  This software incorporates material from third parties. See Notices.txt for details.
 *----------------------------------------/----------------------------------------------------*/

pub struct ProbabilityTablesCoefficientContext {
    pub best_prior: i32, // lakhani or aavrg depending on coefficient number
    pub num_non_zeros_bin: u8,
    pub best_prior_bit_len: u8,
}

impl Default for ProbabilityTablesCoefficientContext {
    fn default() -> ProbabilityTablesCoefficientContext {
        ProbabilityTablesCoefficientContext::new()
    }
}

impl ProbabilityTablesCoefficientContext {
    pub fn new() -> Self {
        return ProbabilityTablesCoefficientContext {
            best_prior: 0,
            num_non_zeros_bin: 0,
            best_prior_bit_len: 0,
        };
    }
}
