/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use anyhow::{Context, Result};

use std::cmp;
use std::io::Write;

use crate::consts::UNZIGZAG_49;
use crate::enabled_features::EnabledFeatures;
use crate::helpers::*;
use crate::lepton_error::ExitCode;

use crate::metrics::Metrics;
use crate::structs::{
    block_based_image::AlignedBlock, block_based_image::BlockBasedImage,
    block_context::BlockContext, model::Model, model::ModelPerColor,
    neighbor_summary::NeighborSummary, probability_tables::ProbabilityTables,
    probability_tables_set::ProbabilityTablesSet, quantization_tables::QuantizationTables,
    row_spec::RowSpec, truncate_components::*, vpx_bool_writer::VPXBoolWriter,
};

use default_boxed::DefaultBoxed;

use super::block_context::NeighborData;

#[inline(never)] // don't inline so that the profiler can get proper data
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
    features: &EnabledFeatures,
) -> Result<Metrics> {
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
                features,
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
                features,
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
                features,
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

    Ok(bool_writer.drain_stats())
}

#[inline(never)] // don't inline so that the profiler can get proper data
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
    features: &EnabledFeatures,
) -> Result<()> {
    if block_width > 0 {
        serialize_tokens::<W, false>(
            state,
            qt,
            left_model,
            model,
            image_data,
            num_non_zeros,
            bool_writer,
            features,
        )
        .context(here!())?;
        let offset = state.next(true);

        if offset >= component_size_in_block {
            return Ok(());
        }
    }

    for _jpeg_x in 1..block_width - 1 {
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
                features,
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
                features,
            )
            .context(here!())?;
        }

        let offset = state.next(true);

        if offset >= component_size_in_block {
            return Ok(());
        }
    }

    if block_width > 1 {
        if right_model.is_all_present() {
            serialize_tokens::<W, true>(
                state,
                qt,
                right_model,
                model,
                image_data,
                num_non_zeros,
                bool_writer,
                features,
            )
            .context(here!())?;
        } else {
            serialize_tokens::<W, false>(
                state,
                qt,
                right_model,
                model,
                image_data,
                num_non_zeros,
                bool_writer,
                features,
            )
            .context(here!())?;
        }

        state.next(false);
    }
    Ok(())
}

#[inline(never)] // don't inline so that the profiler can get proper data
fn serialize_tokens<W: Write, const ALL_PRESENT: bool>(
    context: &mut BlockContext,
    qt: &QuantizationTables,
    pt: &ProbabilityTables,
    model: &mut Model,
    image_data: &BlockBasedImage,
    num_non_zeros: &mut [NeighborSummary],
    bool_writer: &mut VPXBoolWriter<W>,
    features: &EnabledFeatures,
) -> Result<()> {
    debug_assert!(ALL_PRESENT == pt.is_all_present());

    let block = context.here(image_data);

    let neighbors =
        context.get_neighbor_data::<ALL_PRESENT>(image_data, context, num_non_zeros, pt);

    #[cfg(feature = "detailed_tracing")]
    trace!(
        "block {0}:{1:x}",
        context.get_here_index(),
        block.get_hash()
    );

    let ns = write_coefficients::<ALL_PRESENT, W>(
        pt,
        &neighbors,
        block,
        model,
        bool_writer,
        qt,
        features,
    )?;

    *context.neighbor_context_here(num_non_zeros) = ns;

    Ok(())
}

pub fn write_coefficients<const ALL_PRESENT: bool, W: Write>(
    pt: &ProbabilityTables,
    neighbors_data: &NeighborData,
    here: &AlignedBlock,
    model: &mut Model,
    bool_writer: &mut VPXBoolWriter<W>,
    qt: &QuantizationTables,
    features: &EnabledFeatures,
) -> Result<NeighborSummary> {
    let model_per_color = model.get_per_color(pt);

    let num_non_zeros_7x7 = here.get_count_of_non_zeros_7x7();

    let predicted_num_non_zeros_7x7 =
        pt.calc_non_zero_counts_context_7x7::<ALL_PRESENT>(neighbors_data);

    model_per_color
        .write_non_zero_7x7_count(bool_writer, predicted_num_non_zeros_7x7, num_non_zeros_7x7)
        .context(here!())?;

    let mut eob_x = 0;
    let mut eob_y = 0;
    let mut num_non_zeros_left_7x7 = num_non_zeros_7x7;

    let best_priors = pt.calc_coefficient_context_7x7_aavg_block::<ALL_PRESENT>(
        &neighbors_data.left,
        &neighbors_data.above,
        &neighbors_data.above_left,
    );

    for zig49 in 0..49 {
        if num_non_zeros_left_7x7 == 0 {
            break;
        }

        let coord = UNZIGZAG_49[zig49];

        let best_prior_bit_length = u16_bit_length(best_priors[coord as usize] as u16);

        // this should work in all cases but doesn't utilize that the zig49 is related
        let coef = here.get_coefficient(coord as usize);

        model_per_color
            .write_coef(
                bool_writer,
                coef,
                zig49,
                ProbabilityTables::num_non_zeros_to_bin_7x7(num_non_zeros_left_7x7) as usize,
                best_prior_bit_length as usize,
            )
            .context(here!())?;

        if coef != 0 {
            num_non_zeros_left_7x7 -= 1;

            let bx = coord & 7;
            let by = coord >> 3;

            debug_assert!(bx > 0 && by > 0, "this does the DC and the lower 7x7 AC");

            eob_x = cmp::max(eob_x, bx);
            eob_y = cmp::max(eob_y, by);
        }
    }
    encode_edge::<W, ALL_PRESENT>(
        neighbors_data.left,
        neighbors_data.above,
        here,
        model_per_color,
        bool_writer,
        qt,
        pt,
        num_non_zeros_7x7,
        eob_x as u8,
        eob_y as u8,
    )
    .context(here!())?;

    let predicted_val = pt.adv_predict_dc_pix::<ALL_PRESENT>(&here, qt, neighbors_data, features);

    let avg_predicted_dc = ProbabilityTables::adv_predict_or_unpredict_dc(
        here.get_dc(),
        false,
        predicted_val.predicted_dc.into(),
    );

    if here.get_dc() as i32
        != ProbabilityTables::adv_predict_or_unpredict_dc(
            avg_predicted_dc as i16,
            true,
            predicted_val.predicted_dc.into(),
        )
    {
        return err_exit_code(ExitCode::CoefficientOutOfRange, "BlockDC mismatch");
    }
    model
        .write_dc(
            bool_writer,
            pt.get_color_index(),
            avg_predicted_dc as i16,
            predicted_val.uncertainty,
            predicted_val.uncertainty2,
        )
        .context(here!())?;

    let neighbor_summary = NeighborSummary::calculate_neighbor_summary(
        &predicted_val.advanced_predict_dc_pixels_sans_dc,
        qt,
        here,
        num_non_zeros_7x7,
        features,
    );

    Ok(neighbor_summary)
}

#[inline(never)] // don't inline so that the profiler can get proper data
fn encode_edge<W: Write, const ALL_PRESENT: bool>(
    left: &AlignedBlock,
    above: &AlignedBlock,
    here: &AlignedBlock,
    model_per_color: &mut ModelPerColor,
    bool_writer: &mut VPXBoolWriter<W>,
    qt: &QuantizationTables,
    pt: &ProbabilityTables,
    num_non_zeros_7x7: u8,
    eob_x: u8,
    eob_y: u8,
) -> Result<()> {
    encode_one_edge::<W, ALL_PRESENT, true>(
        left,
        above,
        here,
        model_per_color,
        bool_writer,
        qt,
        pt,
        num_non_zeros_7x7,
        eob_x,
    )
    .context(here!())?;
    encode_one_edge::<W, ALL_PRESENT, false>(
        left,
        above,
        here,
        model_per_color,
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
    left: &AlignedBlock,
    above: &AlignedBlock,
    block: &AlignedBlock,
    model_per_color: &mut ModelPerColor,
    bool_writer: &mut VPXBoolWriter<W>,
    qt: &QuantizationTables,
    pt: &ProbabilityTables,
    num_non_zeros_7x7: u8,
    est_eob: u8,
) -> Result<()> {
    let mut num_non_zeros_edge;

    if HORIZONTAL {
        num_non_zeros_edge = count_non_zero(block.get_coefficient(1))
            + count_non_zero(block.get_coefficient(2))
            + count_non_zero(block.get_coefficient(3))
            + count_non_zero(block.get_coefficient(4))
            + count_non_zero(block.get_coefficient(5))
            + count_non_zero(block.get_coefficient(6))
            + count_non_zero(block.get_coefficient(7));
    } else {
        num_non_zeros_edge = count_non_zero(block.get_coefficient(1 * 8))
            + count_non_zero(block.get_coefficient(2 * 8))
            + count_non_zero(block.get_coefficient(3 * 8))
            + count_non_zero(block.get_coefficient(4 * 8))
            + count_non_zero(block.get_coefficient(5 * 8))
            + count_non_zero(block.get_coefficient(6 * 8))
            + count_non_zero(block.get_coefficient(7 * 8));
    }

    model_per_color
        .write_non_zero_edge_count::<W, HORIZONTAL>(
            bool_writer,
            est_eob,
            num_non_zeros_7x7,
            num_non_zeros_edge,
        )
        .context(here!())?;

    let delta;
    let mut zig15offset;

    if HORIZONTAL {
        delta = 1;
        zig15offset = 0;
    } else {
        delta = 8;
        zig15offset = 7;
    }

    let mut coord = delta;
    for _lane in 0..7 {
        if num_non_zeros_edge == 0 {
            break;
        }

        let ptcc8 = pt.calc_coefficient_context8_lak::<ALL_PRESENT, HORIZONTAL>(
            qt,
            coord,
            &block,
            &above,
            &left,
            num_non_zeros_edge,
        );

        let coef = block.get_coefficient(coord);

        model_per_color
            .write_edge_coefficient(bool_writer, qt, coef, coord, zig15offset, &ptcc8)
            .context(here!())?;

        if coef != 0 {
            num_non_zeros_edge -= 1;
        }

        coord += delta;
        zig15offset += 1;
    }

    Ok(())
}

#[test]
fn roundtrip_corner() {
    let left = AlignedBlock::new([1; 64]);
    let above = AlignedBlock::new([1; 64]);
    let here = AlignedBlock::new([1; 64]);

    // verified output from a previous run. If this fails, then you've made a change
    // to the binary format of the compressor and probably shouldn't do that since then
    // the compressor and decompressor won't be compatible anymore.
    let verified_output: [Vec<u8>; 4] = [
        vec![
            1, 128, 1, 102, 54, 143, 149, 75, 100, 30, 184, 13, 62, 85, 134, 168, 117, 148, 252,
            32, 53, 87, 179, 120, 125, 251, 194, 248,
        ],
        vec![
            85, 139, 3, 242, 215, 114, 100, 42, 160, 13, 179, 101, 77, 204, 5, 66, 122, 247, 177,
            221, 131, 158, 52, 180, 180, 49, 30, 0,
        ],
        vec![
            85, 139, 3, 242, 215, 114, 100, 42, 160, 13, 179, 101, 77, 204, 5, 66, 122, 247, 177,
            220, 253, 1, 185, 41, 255, 54, 120, 0,
        ],
        vec![
            85, 139, 3, 242, 215, 114, 100, 42, 160, 13, 179, 101, 77, 204, 5, 66, 122, 247, 177,
            220, 253, 1, 188, 188, 140, 140, 145, 0,
        ],
    ];

    roundtrip_read_write_coefficients_all(
        &left,
        &above,
        &here,
        &verified_output,
        &EnabledFeatures::compat_lepton_vector_read(),
    );
}

/// test large coefficients that could overflow unpredictably if there are changes to the
/// way the math operations are performed (for example overflow or bitness)
#[test]
fn roundtrip_corner_large_coef() {
    let left = AlignedBlock::new([1023; 64]);
    let above = AlignedBlock::new([1023; 64]);
    let here = AlignedBlock::new([1023; 64]);

    // verified output from a previous run. If this fails, then you've made a change
    // to the binary format of the compressor and probably shouldn't do that since then
    // the compressor and decompressor won't be compatible anymore.
    let verified_output: [Vec<u8>; 4] = [
        vec![
            1, 128, 1, 255, 147, 255, 255, 255, 226, 127, 248, 15, 255, 255, 251, 47, 254, 139,
            255, 255, 254, 71, 255, 95, 255, 255, 254, 135, 255, 101, 255, 255, 253, 111, 255, 239,
            255, 255, 219, 127, 255, 255, 255, 255, 231, 255, 255, 217, 127, 255, 255, 255, 255,
            255, 255, 227, 255, 255, 213, 255, 255, 255, 255, 255, 255, 255, 255, 255, 253, 63,
            255, 251, 63, 255, 255, 255, 255, 255, 255, 255, 255, 215, 255, 254, 247, 255, 255,
            255, 255, 255, 255, 223, 255, 255, 91, 255, 255, 255, 255, 255, 255, 255, 255, 253, 95,
            255, 242, 255, 255, 255, 255, 255, 255, 255, 255, 255, 213, 255, 254, 243, 255, 255,
            255, 255, 255, 255, 255, 241, 255, 255, 151, 255, 255, 255, 255, 255, 255, 253, 31,
            255, 227, 191, 255, 255, 255, 255, 255, 255, 255, 255, 167, 255, 248, 191, 255, 255,
            255, 255, 255, 255, 254, 223, 255, 228, 127, 255, 255, 255, 255, 255, 253, 143, 255,
            186, 255, 255, 255, 255, 255, 254, 247, 255, 212, 127, 255, 255, 255, 255, 255, 255,
            233, 255, 248, 159, 255, 255, 255, 255, 255, 253, 47, 254, 139, 255, 255, 255, 255,
            255, 255, 255, 255, 240, 255, 238, 127, 255, 255, 255, 255, 255, 255, 255, 219, 254,
            235, 255, 225, 255, 203, 21, 236, 255, 39, 253, 107, 255, 201, 127, 255, 254, 215, 255,
            255, 255, 248, 255, 251, 207, 255, 255, 255, 213, 43, 13, 3, 255, 243, 127, 255, 250,
            175, 255, 255, 235, 127, 229, 181, 96,
        ],
        vec![
            85, 143, 255, 255, 75, 255, 255, 185, 255, 255, 253, 15, 255, 254, 75, 255, 255, 222,
            255, 255, 224, 255, 255, 252, 15, 255, 250, 191, 255, 254, 227, 255, 253, 255, 255,
            255, 83, 255, 254, 47, 255, 255, 27, 255, 252, 191, 255, 255, 227, 127, 255, 113, 255,
            255, 255, 255, 255, 255, 166, 255, 255, 255, 255, 207, 255, 255, 255, 255, 255, 250,
            127, 255, 200, 127, 255, 55, 255, 245, 191, 255, 255, 255, 255, 254, 87, 255, 255, 255,
            223, 255, 255, 255, 255, 255, 242, 255, 255, 255, 255, 255, 255, 253, 191, 255, 255,
            255, 255, 225, 127, 255, 255, 255, 255, 255, 245, 255, 255, 255, 255, 248, 63, 255,
            255, 191, 255, 255, 255, 255, 255, 227, 127, 255, 255, 255, 255, 255, 31, 255, 255,
            255, 255, 220, 127, 255, 255, 255, 255, 255, 240, 255, 255, 255, 255, 214, 255, 255,
            255, 255, 255, 239, 255, 255, 255, 254, 219, 255, 255, 255, 255, 254, 255, 255, 255,
            255, 255, 101, 255, 255, 255, 255, 255, 255, 143, 255, 255, 255, 251, 239, 246, 223,
            255, 255, 255, 255, 255, 255, 181, 255, 255, 255, 255, 233, 255, 255, 255, 255, 184,
            254, 131, 255, 255, 41, 227, 122, 30, 131, 239, 127, 203, 255, 251, 159, 255, 255, 232,
            255, 255, 255, 255, 131, 255, 183, 255, 102, 250, 31, 204, 59, 255, 255, 255, 197, 255,
            248, 31, 255, 135, 254, 205, 252,
        ],
        vec![
            85, 143, 255, 255, 75, 255, 255, 185, 255, 255, 253, 15, 255, 254, 75, 255, 255, 222,
            255, 255, 224, 255, 255, 252, 15, 255, 250, 191, 255, 254, 227, 255, 253, 255, 255,
            255, 83, 255, 254, 47, 255, 255, 27, 255, 252, 191, 255, 255, 227, 127, 255, 113, 255,
            255, 255, 255, 255, 255, 166, 255, 255, 255, 255, 207, 255, 255, 255, 255, 255, 250,
            127, 255, 200, 127, 255, 55, 255, 245, 191, 255, 255, 255, 255, 254, 87, 255, 255, 255,
            223, 255, 255, 255, 255, 255, 242, 255, 255, 255, 255, 255, 255, 253, 191, 255, 255,
            255, 255, 225, 127, 255, 255, 255, 255, 255, 245, 255, 255, 255, 255, 248, 63, 255,
            255, 191, 255, 255, 255, 255, 255, 227, 127, 255, 255, 255, 255, 255, 31, 255, 255,
            255, 255, 220, 127, 255, 255, 255, 255, 255, 240, 255, 255, 255, 255, 214, 255, 255,
            255, 255, 255, 239, 255, 255, 255, 254, 219, 255, 255, 255, 255, 254, 255, 255, 255,
            255, 255, 101, 255, 255, 255, 255, 255, 255, 143, 255, 255, 255, 251, 239, 246, 223,
            255, 255, 255, 255, 255, 255, 181, 255, 255, 255, 255, 233, 255, 255, 255, 255, 184,
            254, 131, 255, 255, 41, 227, 122, 31, 255, 199, 73, 169, 31, 255, 255, 255, 218, 255,
            254, 163, 255, 250, 15, 255, 255, 225, 127, 255, 255, 251, 186, 156, 54, 31, 255, 245,
            255, 255, 253, 175, 255, 255, 250, 127, 247, 215, 240, 0,
        ],
        vec![
            85, 143, 255, 255, 75, 255, 255, 185, 255, 255, 253, 15, 255, 254, 75, 255, 255, 222,
            255, 255, 224, 255, 255, 252, 15, 255, 250, 191, 255, 254, 227, 255, 253, 255, 255,
            255, 83, 255, 254, 47, 255, 255, 27, 255, 252, 191, 255, 255, 227, 127, 255, 113, 255,
            255, 255, 255, 255, 255, 166, 255, 255, 255, 255, 207, 255, 255, 255, 255, 255, 250,
            127, 255, 200, 127, 255, 55, 255, 245, 191, 255, 255, 255, 255, 254, 87, 255, 255, 255,
            223, 255, 255, 255, 255, 255, 242, 255, 255, 255, 255, 255, 255, 253, 191, 255, 255,
            255, 255, 225, 127, 255, 255, 255, 255, 255, 245, 255, 255, 255, 255, 248, 63, 255,
            255, 191, 255, 255, 255, 255, 255, 227, 127, 255, 255, 255, 255, 255, 31, 255, 255,
            255, 255, 220, 127, 255, 255, 255, 255, 255, 240, 255, 255, 255, 255, 214, 255, 255,
            255, 255, 255, 239, 255, 255, 255, 254, 219, 255, 255, 255, 255, 254, 255, 255, 255,
            255, 255, 101, 255, 255, 255, 255, 255, 255, 143, 255, 255, 255, 251, 239, 246, 223,
            255, 255, 255, 255, 255, 255, 181, 255, 255, 255, 255, 233, 255, 255, 255, 255, 184,
            254, 131, 255, 255, 41, 227, 122, 31, 255, 199, 73, 169, 31, 255, 255, 255, 218, 255,
            254, 163, 255, 250, 15, 255, 255, 225, 127, 21, 228, 229, 37, 127, 255, 255, 242, 63,
            252, 231, 255, 175, 255, 74, 17, 0,
        ],
    ];

    roundtrip_read_write_coefficients_all(
        &left,
        &above,
        &here,
        &verified_output,
        &EnabledFeatures::compat_lepton_scalar_read(),
    );
}

/// tests a pattern where all the coefficients are unique to make sure we don't mix up anything
#[test]
fn roundtrip_corner_unique() {
    let mut arr = [0; 64];
    for i in 0..64 {
        arr[i] = i as i16;
    }

    let left = AlignedBlock::new(arr);
    let above = AlignedBlock::new(arr.map(|x| x + 64));
    let here = AlignedBlock::new(arr.map(|x| x + 128));

    // verified output from a previous run. If this fails, then you've made a change
    // to the binary format of the compressor and probably shouldn't do that since then
    // the compressor and decompressor won't be compatible anymore.
    let verified_output: [Vec<u8>; 4] = [
        vec![
            1, 128, 1, 253, 183, 255, 229, 162, 154, 205, 253, 215, 28, 66, 15, 176, 179, 123, 131,
            243, 6, 232, 194, 254, 138, 197, 171, 191, 205, 252, 127, 255, 249, 158, 254, 183, 175,
            255, 255, 255, 255, 22, 127, 215, 127, 115, 255, 255, 255, 213, 199, 246, 79, 253, 63,
            255, 61, 191, 143, 245, 255, 255, 255, 255, 224, 33, 242, 31, 254, 1, 175, 255, 255,
            255, 160, 59, 205, 35, 63, 255, 255, 255, 210, 183, 222, 127, 227, 255, 255, 248, 141,
            59, 175, 255, 255, 255, 243, 133, 250, 95, 255, 253, 255, 255, 250, 228, 123, 186, 119,
            255, 255, 231, 3, 233, 159, 135, 255, 255, 80, 223, 91, 182, 63, 254, 187, 94, 112,
            126, 191, 255, 235, 207, 228, 247, 255, 255, 254, 108, 148, 183, 245, 207, 255, 255,
            252, 36, 52, 23, 254, 191, 255, 255, 223, 97, 242, 189, 67, 195, 176, 32, 39, 168, 1,
            129, 144, 0, 83, 0, 128, 13, 252, 104, 15, 39, 245, 136, 51, 255, 255, 244, 224, 101,
            121, 32, 41, 255, 255, 254, 16, 20, 46, 129, 193, 144, 157, 0, 44, 143, 0, 0, 3, 254,
            17, 245, 236, 97, 191, 255, 97, 181, 224, 0,
        ],
        vec![
            85, 143, 255, 255, 255, 255, 149, 191, 169, 72, 255, 255, 235, 31, 229, 137, 255, 255,
            225, 37, 222, 212, 255, 255, 243, 137, 236, 245, 255, 255, 70, 127, 41, 71, 255, 209,
            143, 192, 30, 127, 246, 216, 250, 6, 207, 255, 131, 79, 191, 62, 231, 255, 120, 31,
            149, 127, 233, 254, 173, 93, 68, 15, 253, 29, 27, 71, 255, 128, 107, 255, 120, 139, 18,
            157, 127, 255, 199, 105, 146, 127, 183, 255, 77, 67, 28, 127, 255, 250, 13, 247, 175,
            255, 253, 253, 124, 89, 166, 119, 253, 199, 90, 159, 233, 95, 53, 174, 15, 94, 121,
            191, 41, 124, 242, 247, 211, 225, 121, 255, 248, 111, 85, 95, 173, 255, 228, 81, 201,
            127, 241, 254, 90, 26, 75, 223, 235, 51, 139, 126, 231, 166, 144, 6, 6, 64, 1, 76, 2,
            0, 55, 241, 160, 60, 159, 214, 32, 207, 255, 255, 211, 129, 149, 228, 128, 167, 232,
            141, 157, 162, 134, 183, 3, 128, 4, 20, 16, 0, 0, 255, 255, 200, 13, 122, 231, 241,
            249, 108, 162, 144,
        ],
        vec![
            85, 143, 255, 255, 108, 111, 2, 91, 255, 239, 119, 224, 20, 127, 250, 246, 244, 106,
            255, 252, 135, 88, 74, 127, 252, 121, 11, 72, 223, 253, 88, 27, 175, 71, 251, 140, 59,
            239, 47, 249, 153, 185, 239, 238, 127, 211, 27, 203, 191, 246, 254, 156, 60, 158, 231,
            248, 61, 241, 63, 254, 1, 175, 251, 132, 87, 4, 103, 255, 251, 144, 115, 111, 247, 255,
            181, 58, 94, 255, 255, 242, 169, 230, 111, 255, 247, 239, 227, 209, 151, 197, 226, 197,
            172, 126, 201, 237, 13, 204, 111, 254, 253, 171, 221, 253, 251, 15, 236, 219, 255, 248,
            126, 49, 191, 173, 255, 204, 137, 130, 255, 227, 249, 68, 189, 143, 127, 64, 6, 172,
            203, 223, 246, 152, 0, 14, 224, 128, 0, 13, 121, 128, 0, 255, 255, 253, 92, 3, 20, 195,
            192, 73, 126, 220, 3, 192, 255, 207, 192, 255, 255, 255, 244, 176, 46, 131, 130, 162,
            241, 162, 0, 57, 181, 0, 0, 3, 254, 18, 71, 238, 146, 207, 255, 131, 194, 125, 176, 0,
        ],
        vec![
            85, 143, 255, 255, 239, 241, 233, 233, 255, 255, 188, 247, 147, 91, 255, 251, 224, 184,
            127, 127, 255, 102, 158, 239, 135, 255, 125, 191, 90, 47, 255, 159, 159, 99, 147, 255,
            98, 47, 140, 78, 255, 172, 191, 188, 126, 231, 254, 168, 222, 93, 255, 183, 244, 225,
            228, 247, 63, 193, 239, 137, 255, 240, 13, 127, 220, 34, 184, 35, 63, 255, 220, 131,
            155, 127, 191, 254, 199, 133, 181, 255, 255, 226, 135, 197, 63, 255, 239, 217, 132,
            145, 31, 111, 182, 79, 52, 125, 43, 210, 79, 127, 215, 157, 102, 213, 112, 122, 115,
            47, 206, 243, 255, 235, 149, 66, 191, 27, 255, 102, 31, 173, 255, 175, 235, 174, 157,
            189, 253, 0, 26, 179, 47, 127, 218, 96, 0, 59, 130, 0, 0, 53, 230, 0, 3, 255, 255, 245,
            112, 12, 83, 15, 1, 37, 251, 112, 15, 3, 255, 63, 3, 255, 111, 8, 192, 116, 44, 176,
            48, 0, 36, 160, 128, 0, 7, 255, 254, 64, 107, 211, 63, 143, 75, 91, 209, 0,
        ],
    ];

    roundtrip_read_write_coefficients_all(
        &left,
        &above,
        &here,
        &verified_output,
        &EnabledFeatures::compat_lepton_vector_read(),
    );
}

#[cfg(test)]
fn roundtrip_read_write_coefficients_all(
    left: &AlignedBlock,
    above: &AlignedBlock,
    here: &AlignedBlock,
    verified_output: &[Vec<u8>; 4],
    features: &EnabledFeatures,
) {
    for (&(left_present, above_present), vec) in
        [(false, false), (true, false), (false, true), (true, true)]
            .iter()
            .zip(verified_output.iter())
    {
        roundtrip_read_write_coefficients(
            left_present,
            above_present,
            left,
            above,
            here,
            vec,
            features,
        );
    }
}

/// randomizes the branches of the model so that we don't start with a
/// state where all the branches are in the same state. This is important
/// to catch any misaligment in the model state between reading and writing.
#[cfg(test)]
fn make_random_model() -> Box<Model> {
    let mut model = Model::default_boxed();
    let mut count = 1;
    model.walk(|x| {
        x.set_count(count);
        count = count.wrapping_add(1);
    });
    model
}

#[cfg(test)]
fn roundtrip_read_write_coefficients(
    left_present: bool,
    above_present: bool,
    left: &AlignedBlock,
    above: &AlignedBlock,
    here: &AlignedBlock,
    verified_output: &Vec<u8>,
    features: &EnabledFeatures,
) {
    use crate::structs::idct::run_idct;
    use crate::structs::lepton_decoder::read_coefficients;
    use crate::structs::neighbor_summary::NEIGHBOR_DATA_EMPTY;
    use crate::structs::vpx_bool_reader::VPXBoolReader;
    use std::io::Cursor;

    let pt = ProbabilityTables::new(0, left_present, above_present);

    let mut write_model = make_random_model();

    let mut buffer = Vec::new();

    let mut bool_writer = VPXBoolWriter::new(&mut buffer).unwrap();

    let qt = QuantizationTables::new_from_table(&[1; 64]);

    // calculate the neighbor values. Normally this is done by recycling the previous results
    // but since we are testing a one-off here, manually calculate the values
    let above_neighbor = if above_present {
        let idct_above = run_idct::<true>(above, qt.get_quantization_table_transposed());

        NeighborSummary::calculate_neighbor_summary(
            &idct_above,
            &qt,
            &above,
            above.get_count_of_non_zeros_7x7(),
            &features,
        )
    } else {
        NEIGHBOR_DATA_EMPTY
    };

    let left_neighbor = if left_present {
        let idct_left = run_idct::<true>(left, qt.get_quantization_table_transposed());

        NeighborSummary::calculate_neighbor_summary(
            &idct_left,
            &qt,
            &left,
            left.get_count_of_non_zeros_7x7(),
            &features,
        )
    } else {
        NEIGHBOR_DATA_EMPTY
    };

    let neighbors = NeighborData {
        above: &above,
        left: &left,
        above_left: &above,
        neighbor_context_above: &above_neighbor,
        neighbor_context_left: &left_neighbor,
    };

    // use the version with ALL_PRESENT is both above and left neighbors are present
    let ns_read = if left_present && above_present {
        write_coefficients::<true, _>(
            &pt,
            &neighbors,
            &here,
            &mut write_model,
            &mut bool_writer,
            &qt,
            &features,
        )
    } else {
        write_coefficients::<false, _>(
            &pt,
            &neighbors,
            &here,
            &mut write_model,
            &mut bool_writer,
            &qt,
            &features,
        )
    }
    .unwrap();

    bool_writer.finish().unwrap();

    let mut read_model = make_random_model();
    let mut bool_reader = VPXBoolReader::new(Cursor::new(&buffer)).unwrap();

    // use the version with ALL_PRESENT is both above and left neighbors are present
    let (output, ns_write) = if left_present && above_present {
        read_coefficients::<true, _>(
            &pt,
            &neighbors,
            &mut read_model,
            &mut bool_reader,
            &qt,
            &features,
        )
    } else {
        read_coefficients::<false, _>(
            &pt,
            &neighbors,
            &mut read_model,
            &mut bool_reader,
            &qt,
            &features,
        )
    }
    .unwrap();

    assert_eq!(ns_write.get_num_non_zeros(), ns_read.get_num_non_zeros());
    assert_eq!(ns_write.get_horizontal(), ns_read.get_horizontal());
    assert_eq!(ns_write.get_vertical(), ns_read.get_vertical());
    assert_eq!(output.get_block(), here.get_block());
    if verified_output.len() != 0 {
        assert_eq!(verified_output[..], buffer[..]);
    }
    assert_eq!(write_model.model_checksum(), read_model.model_checksum());

    println!("{:?}", &buffer[..]);
}
