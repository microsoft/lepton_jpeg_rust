/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See License.txt in the project root for license information.
 *  This software incorporates material from third parties. See Notices.txt for details.
 *----------------------------------------/----------------------------------------------------*/

use anyhow::{Context, Result};

use std::cmp;
use std::io::Write;

use crate::helpers::*;
use crate::lepton_error::ExitCode;
use crate::{consts::*, here};

use crate::structs::{
    block_based_image::BlockBasedImage, block_context::BlockContext, model::Model,
    neighbor_summary::NeighborSummary, probability_tables::ProbabilityTables,
    probability_tables_set::ProbabilityTablesSet, quantization_tables::QuantizationTables,
    row_spec::RowSpec, truncate_components::*, vpx_bool_writer::VPXBoolWriter,
};

use default_boxed::DefaultBoxed;

pub fn lepton_encode_row_range<W: Write>(
    pts: &ProbabilityTablesSet,
    quantization_tables: &[QuantizationTables],
    image_data: &[BlockBasedImage],
    writer: &mut W,
    _thread_id: i32,
    colldata: &TruncateComponents,
    min_y: i32,
    max_y: i32,
    is_last_thread: bool,
    full_file_compression: bool,
) -> Result<()> {
    let mut model = Model::default_boxed();
    let mut bool_writer = VPXBoolWriter::new(writer)?;

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

    let component_size_in_blocks = colldata.get_component_sizes_in_blocks();
    let max_coded_heights = colldata.get_max_coded_heights();

    let mut encode_index = 0;
    loop {
        let cur_row = RowSpec::get_row_spec_from_index(
            encode_index,
            image_data,
            colldata.mcu_count_vertical,
            &max_coded_heights,
        );
        encode_index += 1;

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

        // Advance to next row to cache expended block data for current row. Should be called before getting block context.
        let bt = cur_row.component;

        let mut block_context = image_data[bt].off_y(cur_row.curr_y);

        let block_width = image_data[bt].get_block_width();

        if is_top_row[bt] {
            is_top_row[bt] = false;
            process_row(
                &mut model,
                &mut bool_writer,
                &image_data[bt],
                &quantization_tables[bt],
                &pts.corner[bt],
                &pts.top[bt],
                &pts.top[bt],
                colldata,
                &mut block_context,
                &mut num_non_zeros[bt][..],
                block_width,
                component_size_in_blocks[bt],
            )
            .context(here!())?;
        } else if block_width > 1 {
            process_row(
                &mut model,
                &mut bool_writer,
                &image_data[bt],
                &quantization_tables[bt],
                &pts.mid_left[bt],
                &pts.middle[bt],
                &pts.mid_right[bt],
                colldata,
                &mut block_context,
                &mut num_non_zeros[bt][..],
                block_width,
                component_size_in_blocks[bt],
            )
            .context(here!())?;
        } else {
            assert!(block_width == 1, "block_width == 1");
            process_row(
                &mut model,
                &mut bool_writer,
                &image_data[bt],
                &quantization_tables[bt],
                &pts.width_one[bt],
                &pts.width_one[bt],
                &pts.width_one[bt],
                colldata,
                &mut block_context,
                &mut num_non_zeros[bt][..],
                block_width,
                component_size_in_blocks[bt],
            )
            .context(here!())?;
        }
    }

    if is_last_thread && full_file_compression {
        let test = RowSpec::get_row_spec_from_index(
            encode_index,
            image_data,
            colldata.mcu_count_vertical,
            &max_coded_heights,
        );

        assert!(
            test.skip && test.done,
            "Row spec test: cmp {0} luma {1} item {2} skip {3} done {4}",
            test.component,
            test.luma_y,
            test.curr_y,
            test.skip,
            test.done
        );
    }

    bool_writer.finish().context(here!())?;

    Ok(())
}

fn process_row<W: Write>(
    model: &mut Model,
    bool_writer: &mut VPXBoolWriter<W>,
    image_data: &BlockBasedImage,
    qt: &QuantizationTables,
    left_model: &ProbabilityTables,
    middle_model: &ProbabilityTables,
    right_model: &ProbabilityTables,
    _colldata: &TruncateComponents,
    state: &mut BlockContext,
    num_non_zeros: &mut [NeighborSummary],
    block_width: i32,
    component_size_in_block: i32,
) -> Result<()> {
    if block_width > 0 {
        state
            .neighbor_context_here(num_non_zeros)
            .set_num_non_zeros(state.here(image_data).get_count_of_non_zeros_7x7());

        serialize_tokens::<W, false>(
            state,
            qt,
            left_model,
            model,
            image_data,
            num_non_zeros,
            bool_writer,
        )
        .context(here!())?;
        let offset = state.next(true);

        if offset >= component_size_in_block {
            return Ok(());
        }
    }

    for _jpeg_x in 1..block_width - 1 {
        state
            .neighbor_context_here(num_non_zeros)
            .set_num_non_zeros(state.here(image_data).get_count_of_non_zeros_7x7());

        // shortcut all the checks for the presence of left/right components by passing a constant generic parameter
        if middle_model.is_all_present() {
            serialize_tokens::<W, true>(
                state,
                qt,
                middle_model,
                model,
                image_data,
                num_non_zeros,
                bool_writer,
            )
            .context(here!())?;
        } else {
            serialize_tokens::<W, false>(
                state,
                qt,
                middle_model,
                model,
                image_data,
                num_non_zeros,
                bool_writer,
            )
            .context(here!())?;
        }

        let offset = state.next(true);

        if offset >= component_size_in_block {
            return Ok(());
        }
    }

    if block_width > 1 {
        state
            .neighbor_context_here(num_non_zeros)
            .set_num_non_zeros(state.here(image_data).get_count_of_non_zeros_7x7());

        serialize_tokens::<W, false>(
            state,
            qt,
            right_model,
            model,
            image_data,
            num_non_zeros,
            bool_writer,
        )
        .context(here!())?;
        state.next(false);
    }
    Ok(())
}

fn serialize_tokens<W: Write, const ALL_PRESENT: bool>(
    context: &mut BlockContext,
    qt: &QuantizationTables,
    pt: &ProbabilityTables,
    model: &mut Model,
    image_data: &BlockBasedImage,
    num_non_zeros: &mut [NeighborSummary],
    bool_writer: &mut VPXBoolWriter<W>,
) -> Result<()> {
    let num_non_zeros_7x7 = context.non_zeros_here(&num_non_zeros);

    model
        .write_non_zero_7x7_count(
            bool_writer,
            pt.get_color_index(),
            pt.calc_non_zero_counts_context_7x7::<ALL_PRESENT>(context, num_non_zeros),
            num_non_zeros_7x7,
        )
        .context(here!())?;

    let mut eob_x = 0;
    let mut eob_y = 0;
    let mut num_non_zeros_left_7x7 = num_non_zeros_7x7;

    let block = context.here(image_data);

    #[cfg(feature = "detailed_tracing")]
    println!(
        "block {0}:{1:x}",
        context.get_here_index(),
        block.get_hash()
    );

    for zig49 in 0..49 {
        if num_non_zeros_left_7x7 == 0 {
            break;
        }

        let coord = UNZIGZAG_49[zig49];

        let ptcc7x7 = pt.calc_coefficient_context_7x7_aavg::<ALL_PRESENT>(
            image_data,
            coord.into(),
            context,
            num_non_zeros_left_7x7,
        );

        // this should work in all cases but doesn't utilize that the zig49 is related
        let coef = block.get_coefficient(zig49);

        model
            .write_coef(
                bool_writer,
                pt.get_color_index(),
                coef,
                coord as usize,
                zig49,
                ptcc7x7.num_non_zeros_bin as usize,
                ptcc7x7.best_prior_bit_len as usize,
            )
            .context(here!())?;

        if coef != 0 {
            num_non_zeros_left_7x7 -= 1;

            let bx = coord & 7;
            let by = coord >> 3;

            assert!(bx > 0 && by > 0, "this does the DC and the lower 7x7 AC");

            eob_x = cmp::max(eob_x, bx);
            eob_y = cmp::max(eob_y, by);
        }
    }

    encode_edge::<W, ALL_PRESENT>(
        context,
        image_data,
        model,
        bool_writer,
        qt,
        pt,
        num_non_zeros_7x7,
        eob_x,
        eob_y,
    )
    .context(here!())?;

    let predicted_val =
        pt.adv_predict_dc_pix::<ALL_PRESENT>(image_data, qt, context, &num_non_zeros);

    let avg_predicted_dc = ProbabilityTables::adv_predict_or_unpredict_dc(
        block.get_dc(),
        false,
        predicted_val.predicted_dc.into(),
    );

    if block.get_dc() as i32
        != ProbabilityTables::adv_predict_or_unpredict_dc(
            avg_predicted_dc as i16,
            true,
            predicted_val.predicted_dc.into(),
        )
    {
        return err_exit_code(ExitCode::CoefficientOutOfRange, "BlockDC mismatch");
    }

    // do DC
    model
        .write_dc(
            bool_writer,
            pt.get_color_index(),
            avg_predicted_dc as i16,
            predicted_val.uncertainty,
            predicted_val.uncertainty2,
        )
        .context(here!())?;

    let here = context.neighbor_context_here(num_non_zeros);

    here.set_horizontal(
        &predicted_val.advanced_predict_dc_pixels_sans_dc,
        qt.get_quantization_table(),
        block.get_dc(),
    );

    here.set_vertical(
        &predicted_val.advanced_predict_dc_pixels_sans_dc,
        qt.get_quantization_table(),
        block.get_dc(),
    );

    Ok(())
}

fn encode_edge<W: Write, const ALL_PRESENT: bool>(
    context: &BlockContext,
    image_data: &BlockBasedImage,
    model: &mut Model,
    bool_writer: &mut VPXBoolWriter<W>,
    qt: &QuantizationTables,
    pt: &ProbabilityTables,
    num_non_zeros_7x7: u8,
    eob_x: u8,
    eob_y: u8,
) -> Result<()> {
    encode_one_edge::<W, ALL_PRESENT, true>(
        context,
        image_data,
        model,
        bool_writer,
        qt,
        pt,
        num_non_zeros_7x7,
        eob_x,
    )
    .context(here!())?;
    encode_one_edge::<W, ALL_PRESENT, false>(
        context,
        image_data,
        model,
        bool_writer,
        qt,
        pt,
        num_non_zeros_7x7,
        eob_y,
    )
    .context(here!())?;
    Ok(())
}

fn count_non_zero(v: i16) -> u8 {
    if v == 0 {
        0
    } else {
        1
    }
}

fn encode_one_edge<W: Write, const ALL_PRESENT: bool, const HORIZONTAL: bool>(
    block_context: &BlockContext,
    image_data: &BlockBasedImage,
    model: &mut Model,
    bool_writer: &mut VPXBoolWriter<W>,
    qt: &QuantizationTables,
    pt: &ProbabilityTables,
    num_non_zeros_7x7: u8,
    est_eob: u8,
) -> Result<()> {
    let block = block_context.here(image_data);

    let mut num_non_zeros_edge;

    if HORIZONTAL {
        num_non_zeros_edge = count_non_zero(block.get_coefficient_raster(1))
            + count_non_zero(block.get_coefficient_raster(2))
            + count_non_zero(block.get_coefficient_raster(3))
            + count_non_zero(block.get_coefficient_raster(4))
            + count_non_zero(block.get_coefficient_raster(5))
            + count_non_zero(block.get_coefficient_raster(6))
            + count_non_zero(block.get_coefficient_raster(7));
    } else {
        num_non_zeros_edge = count_non_zero(block.get_coefficient_raster(1 * 8))
            + count_non_zero(block.get_coefficient_raster(2 * 8))
            + count_non_zero(block.get_coefficient_raster(3 * 8))
            + count_non_zero(block.get_coefficient_raster(4 * 8))
            + count_non_zero(block.get_coefficient_raster(5 * 8))
            + count_non_zero(block.get_coefficient_raster(6 * 8))
            + count_non_zero(block.get_coefficient_raster(7 * 8));
    }

    model
        .write_non_zero_edge_count::<W, HORIZONTAL>(
            bool_writer,
            pt.get_color_index(),
            est_eob,
            num_non_zeros_7x7,
            num_non_zeros_edge,
        )
        .context(here!())?;

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
            image_data,
            qt,
            coord,
            block_context,
            num_non_zeros_edge,
        );

        let coef = block.get_coefficient((aligned_block_offset + (lane << log_edge_step)) as usize);

        model
            .write_edge_coefficient(bool_writer, qt, pt, coef, coord, zig15offset, &ptcc8)
            .context(here!())?;

        if coef != 0 {
            num_non_zeros_edge -= 1;
        }

        coord += delta;
        zig15offset += 1;
    }

    Ok(())
}
