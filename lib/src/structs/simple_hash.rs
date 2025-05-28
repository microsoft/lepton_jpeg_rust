/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

#![allow(dead_code)]

use std::num::Wrapping;

/// used for debugging when there are divergences between encoder and decoder
pub struct SimpleHash {
    hash: u64,
}

pub trait SimpleHashProvider {
    fn get_u64(&self) -> u64;
}

impl SimpleHashProvider for i32 {
    fn get_u64(&self) -> u64 {
        return *self as u64;
    }
}

impl SimpleHashProvider for u32 {
    fn get_u64(&self) -> u64 {
        return *self as u64;
    }
}

impl SimpleHashProvider for u64 {
    fn get_u64(&self) -> u64 {
        return *self;
    }
}

impl SimpleHash {
    pub fn new() -> Self {
        return SimpleHash { hash: 0 };
    }

    pub fn hash<T: SimpleHashProvider>(&mut self, v: T) {
        self.hash = (Wrapping(self.hash) * Wrapping(13u64) + Wrapping(v.get_u64())).0;
    }

    pub fn get(&self) -> u32 {
        return self.hash as u32;
    }
}
