/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use crate::consts::{JPegDecodeStatus, JPegType};
use crate::lepton_error::{err_exit_code, AddContext, ExitCode};
use crate::{LeptonError, Result};

use crate::structs::jpeg_header::{HuffCodes, JPegHeader};

/// used to keep track of position while encoding or decoding a jpeg
pub struct JpegPositionState {
    /// current component
    cmp: usize,

    /// current minimum coded unit (a fraction of dpos)
    mcu: i32,

    /// index of component
    csc: usize,

    /// offset within mcu
    sub: i32,

    /// current block position in image for this component
    dpos: i32,

    /// number of blocks left until reset interval
    rstw: i32,

    /// tracks long zero byte runs in progressive images
    pub eobrun: u16,

    /// if the previous value was also an eobrun then this is used to make sure
    /// that we don't have two non-maximum value runs in a row that we wouldn't be
    /// able to recode exactly the same way
    pub prev_eobrun: u16,
}

impl JpegPositionState {
    pub fn new(jf: &JPegHeader, mcu: i32) -> Self {
        let cmp = jf.cs_cmp[0];
        let mcumul = jf.cmp_info[cmp].sfv * jf.cmp_info[cmp].sfh;

        let state = JpegPositionState {
            cmp,
            mcu,
            csc: 0,
            sub: 0,
            dpos: mcu * mcumul,
            rstw: if jf.rsti != 0 {
                jf.rsti - (mcu % jf.rsti)
            } else {
                0
            },
            eobrun: 0,
            prev_eobrun: 0,
        };
        return state;
    }

    pub fn get_mcu(&self) -> i32 {
        self.mcu
    }
    pub fn get_dpos(&self) -> i32 {
        self.dpos
    }
    pub fn get_cmp(&self) -> usize {
        self.cmp
    }

    pub fn get_cumulative_reset_markers(&self, jf: &JPegHeader) -> i32 {
        if self.rstw != 0 {
            self.get_mcu() / jf.rsti
        } else {
            0
        }
    }

    pub fn reset_rstw(&mut self, jf: &JPegHeader) {
        self.rstw = jf.rsti;

        // eobruns don't span reset intervals
        self.prev_eobrun = 0;
    }

    /// calculates next position (non interleaved)
    fn next_mcu_pos_noninterleaved(&mut self, jf: &JPegHeader) -> JPegDecodeStatus {
        // increment position
        self.dpos += 1;

        let cmp_info = &jf.cmp_info[self.cmp];

        // fix for non interleaved mcu - horizontal
        if cmp_info.bch != cmp_info.nch && self.dpos % cmp_info.bch == cmp_info.nch {
            self.dpos += cmp_info.bch - cmp_info.nch;
        }

        // fix for non interleaved mcu - vertical
        if cmp_info.bcv != cmp_info.ncv && self.dpos / cmp_info.bch == cmp_info.ncv {
            self.dpos = cmp_info.bc;
        }

        // now we've updated dpos, update the current MCU to be a fraction of that
        if jf.jpeg_type == JPegType::Sequential {
            self.mcu = self.dpos / (cmp_info.sfv * cmp_info.sfh);
        }

        // check position
        if self.dpos >= cmp_info.bc {
            return JPegDecodeStatus::ScanCompleted;
        } else if jf.rsti > 0 {
            self.rstw -= 1;
            if self.rstw == 0 {
                return JPegDecodeStatus::RestartIntervalExpired;
            }
        }

        return JPegDecodeStatus::DecodeInProgress;
    }

    /// calculates next position for MCU
    pub fn next_mcu_pos(&mut self, jf: &JPegHeader) -> JPegDecodeStatus {
        // if there is just one component, go the simple route
        if jf.cs_cmpc == 1 {
            return self.next_mcu_pos_noninterleaved(jf);
        }

        let mut sta = JPegDecodeStatus::DecodeInProgress; // status
        let local_mcuh = jf.mcuh;
        let mut local_mcu = self.mcu;
        let mut local_cmp = self.cmp;

        // increment all counts where needed
        self.sub += 1;
        let mut local_sub = self.sub;
        if local_sub >= jf.cmp_info[local_cmp].mbs {
            self.sub = 0;
            local_sub = 0;

            self.csc += 1;

            if self.csc >= jf.cs_cmpc {
                self.csc = 0;
                self.cmp = jf.cs_cmp[0];
                local_cmp = self.cmp;

                self.mcu += 1;

                local_mcu = self.mcu;

                if local_mcu >= jf.mcuc {
                    sta = JPegDecodeStatus::ScanCompleted;
                } else if jf.rsti > 0 {
                    self.rstw -= 1;
                    if self.rstw == 0 {
                        sta = JPegDecodeStatus::RestartIntervalExpired;
                    }
                }
            } else {
                self.cmp = jf.cs_cmp[self.csc];
                local_cmp = self.cmp;
            }
        }

        let sfh = jf.cmp_info[local_cmp].sfh;
        let sfv = jf.cmp_info[local_cmp].sfv;

        // get correct position in image ( x & y )
        if sfh > 1 {
            // to fix mcu order
            let mcu_over_mcuh = local_mcu / local_mcuh;
            let sub_over_sfv = local_sub / sfv;
            let mcu_mod_mcuh = local_mcu - (mcu_over_mcuh * local_mcuh);
            let sub_mod_sfv = local_sub - (sub_over_sfv * sfv);
            let mut local_dpos = (mcu_over_mcuh * sfh) + sub_over_sfv;

            local_dpos *= jf.cmp_info[local_cmp].bch;
            local_dpos += (mcu_mod_mcuh * sfv) + sub_mod_sfv;

            self.dpos = local_dpos;
        } else if sfv > 1 {
            // simple calculation to speed up things if simple fixing is enough
            self.dpos = (local_mcu * jf.cmp_info[local_cmp].mbs) + local_sub;
        } else {
            // no calculations needed without subsampling
            self.dpos = self.mcu;
        }

        return sta;
    }

    /// skips the eobrun, calculates next position
    pub fn skip_eobrun(&mut self, jf: &JPegHeader) -> Result<JPegDecodeStatus> {
        assert!(jf.cs_cmpc == 1, "this code only works for non-interleved");

        if (self.eobrun) == 0 {
            return Ok(JPegDecodeStatus::DecodeInProgress);
        }

        // compare rst wait counter if needed
        if jf.rsti > 0 {
            if i32::from(self.eobrun) > self.rstw {
                return err_exit_code(
                    ExitCode::UnsupportedJpeg,
                    "skip_eobrun: eob run extends passed end of reset interval",
                )
                .context();
            } else {
                self.rstw -= i32::from(self.eobrun);
            }
        }

        fn checked_add(a: i32, b: i32) -> Result<i32> {
            a.checked_add(b)
                .ok_or_else(|| LeptonError::new(ExitCode::UnsupportedJpeg, "integer overflow"))
        }

        let cmp_info = &jf.cmp_info[self.cmp];

        // fix for non interleaved mcu - horizontal
        if cmp_info.bch != cmp_info.nch {
            self.dpos = checked_add(
                self.dpos,
                (((self.dpos % cmp_info.bch) + i32::from(self.eobrun)) / cmp_info.nch)
                    * (cmp_info.bch - cmp_info.nch),
            )
            .context()?;
        }

        // fix for non interleaved mcu - vertical
        if cmp_info.bcv != cmp_info.ncv && self.dpos / cmp_info.bch >= cmp_info.ncv {
            self.dpos =
                checked_add(self.dpos, (cmp_info.bcv - cmp_info.ncv) * cmp_info.bch).context()?;
        }

        // skip blocks
        self.dpos = checked_add(self.dpos, i32::from(self.eobrun)).context()?;

        // reset eobrun
        self.eobrun = 0;

        // check position to see if we are done decoding
        if self.dpos == cmp_info.bc {
            Ok(JPegDecodeStatus::ScanCompleted)
        } else if self.dpos > cmp_info.bc {
            err_exit_code(
                ExitCode::UnsupportedJpeg,
                "skip_eobrun: position extended passed block count",
            )
            .context()
        } else if jf.rsti > 0 && self.rstw == 0 {
            Ok(JPegDecodeStatus::RestartIntervalExpired)
        } else {
            Ok(JPegDecodeStatus::DecodeInProgress)
        }
    }

    /// checks to see if the we have optimal eob runs (each eobrun is as large as it legally can be) otherwise
    /// we will not know how to reencode the file since the encoder always assumes EOB runs as large as possible
    pub fn check_optimal_eobrun(
        &mut self,
        is_current_block_empty: bool,
        hc: &HuffCodes,
    ) -> Result<()> {
        // if we got an empty block, make sure that the previous zero run was as high as it could be
        // otherwise we won't reencode the file in the same way
        if is_current_block_empty {
            if self.prev_eobrun > 0 && self.prev_eobrun < hc.max_eob_run - 1 {
                return err_exit_code(
                    ExitCode::UnsupportedJpeg,
                    format!("non optimial eobruns not supported (could have encoded up to {0} zero runs in a row, but only did {1} followed by {2}",
                        hc.max_eob_run, self.prev_eobrun + 1,
                        self.eobrun + 1).as_str());
            }
        }

        self.prev_eobrun = self.eobrun;

        Ok(())
    }
}
