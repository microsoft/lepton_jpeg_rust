/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use anyhow::{Context, Result};

use default_boxed::DefaultBoxed;

use std::cmp;
use std::io::Read;

use crate::consts::{LOG_TABLE_256, RASTER_TO_ALIGNED, UNZIGZAG_49};
use crate::helpers::{err_exit_code, here, u16_bit_length};
use crate::lepton_error::ExitCode;

use crate::metrics::Metrics;
use crate::structs::{
    block_based_image::AlignedBlock, block_based_image::BlockBasedImage, model::Model,
    model::ModelPerColor, neighbor_summary::NeighborSummary, probability_tables::ProbabilityTables,
    probability_tables_set::ProbabilityTablesSet, quantization_tables::QuantizationTables,
    row_spec::RowSpec, truncate_components::*, vpx_bool_reader::VPXBoolReader,
};

use super::block_context::BlockContext;

// reads stream from reader and populates image_data with the decoded data

#[inline(never)] // don't inline so that the profiler can get proper data
pub fn lepton_decode_row_range<R: Read>(
    pts: &ProbabilityTablesSet,
    qt: &[QuantizationTables],
    trunc: &TruncateComponents,
    image_data: &mut [BlockBasedImage],
    reader: &mut R,
    min_y: i32,
    max_y: i32,
    is_last_thread: bool,
    full_file_compression: bool,
) -> Result<Metrics> {
    let component_size_in_blocks = trunc.get_component_sizes_in_blocks();
    let max_coded_heights = trunc.get_max_coded_heights();

    let mut is_top_row = Vec::new();
    let mut num_non_zeros = Vec::new();

    // Init helper structures
    for i in 0..image_data.len() {
        is_top_row.push(true);

        let num_non_zeros_length = (image_data[i].get_block_width() << 1) as usize;

        let mut num_non_zero_list = Vec::new();
        num_non_zero_list.resize(num_non_zeros_length, NeighborSummary::new());

        num_non_zeros.push(num_non_zero_list);
    }

    let mut model = Model::default_boxed();
    let mut bool_reader = VPXBoolReader::new(reader)?;

    let mut decode_index = 0;

    loop {
        let cur_row = RowSpec::get_row_spec_from_index(
            decode_index,
            &image_data[..],
            trunc.mcu_count_vertical,
            &max_coded_heights,
        );
        decode_index += 1;

        if cur_row.done {
            break;
        }

        if cur_row.luma_y >= max_y && !(is_last_thread && full_file_compression) {
            break;
        }

        if cur_row.skip {
            continue;
        }

        if cur_row.luma_y < min_y {
            continue;
        }

        decode_row_wrapper(
            &mut model,
            &mut bool_reader,
            pts,
            &mut image_data[cur_row.component],
            &qt[cur_row.component],
            &mut num_non_zeros[cur_row.component],
            &mut is_top_row[..],
            &component_size_in_blocks[..],
            cur_row.component,
            cur_row.curr_y,
        )
        .context(here!())?;
    }
    Ok(bool_reader.drain_stats())
}

#[inline(never)] // don't inline so that the profiler can get proper data
fn decode_row_wrapper<R: Read>(
    model: &mut Model,
    bool_reader: &mut VPXBoolReader<R>,
    pts: &ProbabilityTablesSet,
    image_data: &mut BlockBasedImage,
    qt: &QuantizationTables,
    num_non_zeros: &mut Vec<NeighborSummary>,
    is_top_row: &mut [bool],
    component_size_in_blocks: &[i32],
    component: usize,
    curr_y: i32,
) -> Result<()> {
    let mut context = image_data.off_y(curr_y);

    let block_width = image_data.get_block_width();
    if is_top_row[component] {
        is_top_row[component] = false;
        decode_row(
            model,
            bool_reader,
            &qt,
            &pts.corner[component],
            &pts.top[component],
            &pts.top[component],
            image_data,
            &mut context,
            num_non_zeros,
            component_size_in_blocks[component],
        )
        .context(here!())?;
    } else if block_width > 1 {
        let _bt = component;
        decode_row(
            model,
            bool_reader,
            &qt,
            &pts.mid_left[component],
            &pts.middle[component],
            &pts.mid_right[component],
            image_data,
            &mut context,
            num_non_zeros,
            component_size_in_blocks[component],
        )
        .context(here!())?;
    } else {
        assert!(block_width == 1, "block_width == 1");
        decode_row(
            model,
            bool_reader,
            &qt,
            &pts.width_one[component],
            &pts.width_one[component],
            &pts.width_one[component],
            image_data,
            &mut context,
            num_non_zeros,
            component_size_in_blocks[component],
        )
        .context(here!())?;
    }

    Ok(())
}

fn decode_row<R: Read>(
    model: &mut Model,
    bool_reader: &mut VPXBoolReader<R>,
    qt: &QuantizationTables,
    left_model: &ProbabilityTables,
    middle_model: &ProbabilityTables,
    right_model: &ProbabilityTables,
    image_data: &mut BlockBasedImage,
    block_context: &mut BlockContext,
    num_non_zeros: &mut [NeighborSummary],
    component_size_in_blocks: i32,
) -> Result<()> {
    let block_width = image_data.get_block_width();
    if block_width > 0 {
        parse_token::<R, false>(
            model,
            bool_reader,
            image_data,
            block_context,
            num_non_zeros,
            qt,
            left_model,
        )
        .context(here!())?;
        let offset = block_context.next(true);

        if offset >= component_size_in_blocks {
            return Ok(()); // no sure if this is an error
        }
    }

    for _jpeg_x in 1..block_width - 1 {
        if middle_model.is_all_present() {
            parse_token::<R, true>(
                model,
                bool_reader,
                image_data,
                block_context,
                num_non_zeros,
                qt,
                middle_model,
            )
            .context(here!())?;
        } else {
            parse_token::<R, false>(
                model,
                bool_reader,
                image_data,
                block_context,
                num_non_zeros,
                qt,
                middle_model,
            )
            .context(here!())?;
        }

        let offset = block_context.next(true);

        if offset >= component_size_in_blocks {
            return Ok(()); // no sure if this is an error
        }
    }

    if block_width > 1 {
        if right_model.is_all_present() {
            parse_token::<R, true>(
                model,
                bool_reader,
                image_data,
                block_context,
                num_non_zeros,
                qt,
                right_model,
            )
            .context(here!())?;
        } else {
            parse_token::<R, false>(
                model,
                bool_reader,
                image_data,
                block_context,
                num_non_zeros,
                qt,
                right_model,
            )
            .context(here!())?;
        }

        block_context.next(false);
    }
    Ok(())
}

#[inline(never)] // don't inline so that the profiler can get proper data
fn parse_token<R: Read, const ALL_PRESENT: bool>(
    model: &mut Model,
    bool_reader: &mut VPXBoolReader<R>,
    image_data: &mut BlockBasedImage,
    context: &mut BlockContext,
    num_non_zeros: &mut [NeighborSummary],
    qt: &QuantizationTables,
    pt: &ProbabilityTables,
) -> Result<()> {
    debug_assert!(pt.is_all_present() == ALL_PRESENT);

    let model_per_color = model.get_per_color(pt);

    let num_non_zeros_7x7 = model_per_color
        .read_non_zero_7x7_count(
            bool_reader,
            pt.calc_non_zero_counts_context_7x7::<ALL_PRESENT>(context, num_non_zeros),
        )
        .context(here!())?;

    if num_non_zeros_7x7 > 49 {
        return err_exit_code(ExitCode::StreamInconsistent, "numNonzeros7x7 > 49");
    }

    let (above_left, above, left) = context.get_neighbors::<ALL_PRESENT>(image_data, pt);

    let mut output = AlignedBlock::default();

    let mut eob_x: u8 = 0;
    let mut eob_y: u8 = 0;
    let mut num_non_zeros_left_7x7: u8 = num_non_zeros_7x7;

    let best_priors =
        pt.calc_coefficient_context_7x7_aavg_block::<ALL_PRESENT>(&left, &above, &above_left);

    for zz in 0..49 {
        if num_non_zeros_left_7x7 == 0 {
            break;
        }

        let best_prior_bit_length = u16_bit_length(best_priors[zz] as u16);

        let coef = model_per_color
            .read_coef(
                bool_reader,
                zz.into(),
                ProbabilityTables::num_non_zeros_to_bin(num_non_zeros_left_7x7) as usize,
                best_prior_bit_length as usize,
            )
            .context(here!())?;

        if coef != 0 {
            let coord = UNZIGZAG_49[zz];
            debug_assert!(
                (coord & 7) > 0 && (coord >> 3) > 0,
                "this does the DC and the lower 7x7 AC"
            );

            let b_x = coord & 7;
            let b_y = coord >> 3;

            eob_x = cmp::max(eob_x, b_x);
            eob_y = cmp::max(eob_y, b_y);
            num_non_zeros_left_7x7 -= 1;

            output.set_coefficient(zz, coef);
        }
    }

    decode_edge::<R, ALL_PRESENT>(
        model_per_color,
        bool_reader,
        &left,
        &above,
        &mut output,
        qt,
        pt,
        num_non_zeros_7x7,
        eob_x,
        eob_y,
    )?;

    let predicted_dc = pt.adv_predict_dc_pix::<ALL_PRESENT>(&output, qt, context, num_non_zeros);

    let coef = model
        .read_dc(
            bool_reader,
            pt.get_color_index(),
            predicted_dc.uncertainty,
            predicted_dc.uncertainty2,
        )
        .context(here!())?;

    output.set_dc(ProbabilityTables::adv_predict_or_unpredict_dc(
        coef,
        true,
        predicted_dc.predicted_dc,
    ) as i16);

    let here = context.neighbor_context_here(num_non_zeros);
    here.set_num_non_zeros(num_non_zeros_7x7);

    here.set_horizontal(
        &predicted_dc.advanced_predict_dc_pixels_sans_dc,
        qt.get_quantization_table(),
        output.get_dc(),
    );

    here.set_vertical(
        &predicted_dc.advanced_predict_dc_pixels_sans_dc,
        qt.get_quantization_table(),
        output.get_dc(),
    );

    image_data.append_block(output);

    Ok(())
}

#[inline(never)] // don't inline so that the profiler can get proper data
fn decode_edge<R: Read, const ALL_PRESENT: bool>(
    model_per_color: &mut ModelPerColor,
    bool_reader: &mut VPXBoolReader<R>,
    left: &AlignedBlock,
    above: &AlignedBlock,
    here_mut: &mut AlignedBlock,
    qt: &QuantizationTables,
    pt: &ProbabilityTables,
    num_non_zeros_7x7: u8,
    eob_x: u8,
    eob_y: u8,
) -> Result<()> {
    decode_one_edge::<R, ALL_PRESENT, true>(
        model_per_color,
        bool_reader,
        left,
        above,
        here_mut,
        qt,
        pt,
        num_non_zeros_7x7,
        eob_x,
    )?;
    decode_one_edge::<R, ALL_PRESENT, false>(
        model_per_color,
        bool_reader,
        left,
        above,
        here_mut,
        qt,
        pt,
        num_non_zeros_7x7,
        eob_y,
    )?;
    Ok(())
}

fn decode_one_edge<R: Read, const ALL_PRESENT: bool, const HORIZONTAL: bool>(
    model_per_color: &mut ModelPerColor,
    bool_reader: &mut VPXBoolReader<R>,
    left: &AlignedBlock,
    above: &AlignedBlock,
    here_mut: &mut AlignedBlock,
    qt: &QuantizationTables,
    pt: &ProbabilityTables,
    num_non_zeros_7x7: u8,
    est_eob: u8,
) -> Result<()> {
    let mut num_non_zeros_edge = model_per_color
        .read_non_zero_edge_count::<R, HORIZONTAL>(bool_reader, est_eob, num_non_zeros_7x7)
        .context(here!())?;

    if num_non_zeros_edge > 7 {
        return err_exit_code(ExitCode::StreamInconsistent, "StreamInconsistent");
    }

    let aligned_block_offset;
    let log_edge_step;
    let delta;
    let mut zig15offset;

    if HORIZONTAL {
        log_edge_step = LOG_TABLE_256[(RASTER_TO_ALIGNED[2] - RASTER_TO_ALIGNED[1]) as usize];
        aligned_block_offset = RASTER_TO_ALIGNED[1];
        delta = 1;
        zig15offset = 0;
    } else {
        log_edge_step = LOG_TABLE_256[(RASTER_TO_ALIGNED[16] - RASTER_TO_ALIGNED[8]) as usize];
        aligned_block_offset = RASTER_TO_ALIGNED[8];
        delta = 8;
        zig15offset = 7;
    }

    let mut coord = delta;

    for lane in 0..7 {
        if num_non_zeros_edge == 0 {
            break;
        }

        let ptcc8 = pt.calc_coefficient_context8_lak::<ALL_PRESENT, HORIZONTAL>(
            qt,
            coord,
            here_mut,
            above,
            left,
            num_non_zeros_edge,
        );

        let coef =
            model_per_color.read_edge_coefficient(bool_reader, qt, coord, zig15offset, &ptcc8)?;

        if coef != 0 {
            num_non_zeros_edge -= 1;
        }

        here_mut.set_coefficient(
            aligned_block_offset as usize + (lane << log_edge_step),
            coef,
        );

        coord += delta;
        zig15offset += 1;
    }

    Ok(())
}
